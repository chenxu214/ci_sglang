"""ZBAL dispatch_ffn_combine (FuseEP) MoE path in pool-only symmetric space: expert weights and the
resident workspace live in the zbal pool, every other tensor stays on the native allocator.
Pool sizing needs ``SGLANG_ZBAL_DFC_MOE_LAYERS``; weight layouts live in ``zbal.zbal_moe_repack``.
The zbal-side imports (zbal / Buffer / zbal_moe_repack) must stay lazy inside functions"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.runtime_context import get_parallel, get_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)

# Alignment of the total pool size
_POOL_ALIGN = 2 * 1024 * 1024

# Slack over the exact ledger: covers the 512 B heap granule, the per-tensor 2 MiB budget rounding
# and the mempool's own block rounding, none of which the byte formulas can see.
_POOL_SLACK_MIN = 128 * 1024 * 1024

# gmm1/gmm2 weights required by zbal FuseEP
_FUSE_EP_PREFIXES = frozenset({"w13", "w2"})

# ABI per ModelSlim MoE scheme. The pool has to exist before create_weights allocates the weights,
# so the scheme class (fixed by the checkpoint's quant description) is the only flavor information
# available at sizing time; prepare_weights later checks it against the real weight dtype.
_SCHEME_LAYOUTS = {
    "ModelSlimMXFP8MoEScheme": "fp8",
    "ModelSlimW8A8Int8MoE": "int8",
    "ModelSlimW4A8MXFP4MoE": "w4a8",
    "ModelSlimW4A4MXFP4MoE": "w4a8",
}

# Quant methods without ModelSlim schemes (e.g. the A5 MXFP4 and block-FP8 MoE methods) declare
# their ABI directly as ``zbal_weight_layouts = (w13, w2)`` on the method class.
_KNOWN_LAYOUTS = frozenset(_SCHEME_LAYOUTS.values())


def _declared_layouts(layer: FusedMoE) -> tuple | None:
    """(w13, w2) ABI declared by the layer's quant method, or None when it declares none."""
    method = getattr(layer, "quant_method", None)
    layouts = getattr(method, "zbal_weight_layouts", None)
    if layouts is None:
        return None
    layouts = tuple(layouts)
    if len(layouts) != 2 or any(layout not in _KNOWN_LAYOUTS for layout in layouts):
        logger.error(
            "ZBAL FuseEP: layer_id=%s %s.zbal_weight_layouts=%s must be a (w13, w2) pair of %s",
            getattr(layer, "layer_id", "?"),
            type(method).__name__,
            layouts,
            ", ".join(sorted(_KNOWN_LAYOUTS)),
        )
        raise ValueError("ZBAL FuseEP: invalid zbal_weight_layouts declaration on the quant method")
    return layouts


def _scheme_layouts(layer: FusedMoE) -> tuple:
    """(w13, w2) ABI of one layer: the quant method's declaration, else its two ModelSlim schemes."""
    declared = _declared_layouts(layer)
    if declared is not None:
        return declared
    layouts = []
    for attr in ("w13_scheme", "w2_scheme"):
        scheme = getattr(layer, attr, None)
        name = type(scheme).__name__ if scheme is not None else "?"
        layout = _SCHEME_LAYOUTS.get(name)
        if layout is None:
            logger.error(
                "ZBAL FuseEP: layer_id=%s %s scheme %s has no pool ABI mapping (known: %s)",
                getattr(layer, "layer_id", "?"),
                attr,
                name,
                ", ".join(sorted(_SCHEME_LAYOUTS)),
            )
            raise ValueError(f"ZBAL FuseEP: unsupported MoE scheme for pool sizing: {name}")
        layouts.append(layout)
    return tuple(layouts)


class ZbalFuseEpState:
    """Process-wide FuseEP singleton owning the lazily created symmetric pool and resident Buffer."""

    _instance = None

    def __init__(self) -> None:
        # Cached on first read: forward() asks once per layer per step and the env is fixed for
        # the process lifetime.
        self.enabled: bool | None = None
        self.pool_inited = False
        self.buffer = None
        # (w13, w2) ABIs the pool was sized from.
        self.pool_layouts: tuple | None = None
        # Expert shape the pool was sized from; every MoE layer must match it.
        self.pool_shape_key: tuple | None = None
        # Per-forward prefill token bound frozen with the pool sizing: the pool, the workspace and
        # the runtime guard all read this snapshot.
        self.max_prefill_tokens: int | None = None
        # Budget breakdown logged at init and re-read by the first-forward occupancy check.
        self.ledger: dict | None = None
        # MoE modules whose weights went through the FuseEP ABI, for the layer-count check.
        self.prepared_layers: set = set()
        self.startup_checked = False

    @classmethod
    def instance(cls) -> "ZbalFuseEpState":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def use_fuseep(self) -> bool:
        """FuseEP switch: on when ``--moe-a2a-backend zbal_fuseep`` is selected.

        Cached on first read: forward() asks once per layer per step and the backend is fixed
        for the process lifetime.
        """
        if self.enabled is None:
            from sglang.srt.layers.moe.utils import get_moe_a2a_backend

            self.enabled = get_moe_a2a_backend().is_zbal_fuseep()
        return self.enabled

    def ensure_pool(self, layer: FusedMoE, layouts: tuple) -> None:
        """Create the symmetric pool and its resident Buffer once, before the weights exist.

        ``layouts`` are the (w13, w2) ABIs read from the layer's schemes; every later layer has to
        agree on them because the pool cannot grow.
        """
        if self.pool_inited:
            if layouts != self.pool_layouts:
                logger.error(
                    "ZBAL FuseEP: layer_id=%s weight flavors %s differ from the pool flavors %s; "
                    "the symmetric pool is sized for one ABI set",
                    getattr(layer, "layer_id", "?"),
                    layouts,
                    self.pool_layouts,
                )
                raise ValueError("ZBAL FuseEP requires one weight flavor set across MoE layers")
            return
        init_pool(layer, layouts)
        self.pool_inited = True
        self.pool_layouts = layouts
        self.buffer = _create_buffer(layer, layouts)


def use_zbal_fuseep() -> bool:
    """FuseEP switch: pool-only mode only when explicitly enabled (torch allocator untouched)."""
    return ZbalFuseEpState.instance().use_fuseep()


def _prefill_ceiling() -> int:
    """Per-forward prefill ceiling sglang sizes its own buffers with (0 when chunking is off)."""
    from sglang.srt.arg_groups.overrides import max_prefill_buffer_tokens

    return int(max_prefill_buffer_tokens(get_server_args()))


def _max_prefill_tokens() -> int:
    """Per-forward prefill token bound: the chunk ceiling, or ``--max-prefill-tokens`` without it."""
    return _prefill_ceiling() or int(get_server_args().max_prefill_tokens)


def _bound_tokens() -> int:
    """Frozen prefill bound shared by pool sizing, workspace sizing and the runtime guard."""
    state = ZbalFuseEpState.instance()
    if state.max_prefill_tokens is None:
        state.max_prefill_tokens = _max_prefill_tokens()
    return state.max_prefill_tokens


def _weights_key(layer: FusedMoE) -> tuple:
    """Shape key of one layer's expert weights: the pool is sized from one shape and every MoE
    layer must match it."""
    return (
        layer.num_local_experts,
        layer.hidden_size,
        layer.intermediate_size_per_partition,
    )


def _pool_bytes(layer: FusedMoE, layouts: tuple) -> int:
    """Exact pool ledger: per-layer ABI bytes for the real flavors × layers + transient + workspace."""
    moe_layers = envs.SGLANG_ZBAL_DFC_MOE_LAYERS.get()
    if moe_layers <= 0:
        logger.error(
            "ZBAL FuseEP: SGLANG_ZBAL_DFC_MOE_LAYERS is unset or non-positive (%s); set it to "
            "the number of MoE layers the model builds (MTP/nextn included)",
            moe_layers,
        )
        raise ValueError("ZBAL FuseEP requires SGLANG_ZBAL_DFC_MOE_LAYERS")

    # Fail fast: without the chunk gate the scheduler admits a lone request above
    # --max-prefill-tokens up to the context length, so no pool/workspace size is an upper bound
    # and the failure would only surface on the first long request.
    if _prefill_ceiling() <= 0:
        logger.error(
            "ZBAL FuseEP: chunked prefill is off (--chunked-prefill-size=%s), so one request may "
            "exceed --max-prefill-tokens up to the context length and the pool has no fixed "
            "per-forward bound to size from",
            get_server_args().chunked_prefill_size,
        )
        raise ValueError("ZBAL FuseEP requires --chunked-prefill-size > 0")

    bound = _bound_tokens()

    from zbal import zbal_moe_repack

    groups = tuple(
        zbal_moe_repack.moe_group_weight_bytes(
            layout,
            group=group,
            num_local_experts=layer.num_local_experts,
            hidden=layer.hidden_size,
            ffn=layer.intermediate_size_per_partition,
        )
        for layout, group in zip(layouts, ("w13", "w2"))
    )
    per_layer = sum(groups)
    state = ZbalFuseEpState.instance()
    # The budget assumes every MoE layer (MTP included) shares this shape.
    state.pool_shape_key = _weights_key(layer)
    weights = per_layer * moe_layers
    # int8 repacks into a new zN payload while the checkpoint-format source is still resident, so
    # the budget carries one extra layer of that flavor; MX payloads keep their layout and are
    # used in place (no second buffer). w4a8 still holds one layer's source e8m0 scale while the
    # padded scale is built, but that is ~1/16 of the payload and stays inside ``slack``.
    transient = per_layer if "int8" in layouts else 0
    workspace = _workspace_bytes(layer, layouts)
    slack = max(_POOL_SLACK_MIN, (weights + workspace) // 100)
    total = weights + transient + workspace + slack
    pool = (total + _POOL_ALIGN - 1) // _POOL_ALIGN * _POOL_ALIGN
    state.ledger = dict(
        layouts=layouts,
        moe_layers=moe_layers,
        per_layer=per_layer,
        groups=groups,
        weights=weights,
        transient=transient,
        workspace=workspace,
        slack=slack,
        pool=pool,
    )
    logger.info(
        "ZBAL FuseEP pool size: layouts=%s moe_layers=%s prefill_tokens=%s per_layer=%.3f GiB "
        "(w13=%.3f w2=%.3f) weights=%.3f GiB transient=%.3f GiB workspace=%.3f GiB "
        "slack=%.3f GiB pool=%.3f GiB",
        layouts,
        moe_layers,
        bound,
        per_layer / 2**30,
        groups[0] / 2**30,
        groups[1] / 2**30,
        weights / 2**30,
        transient / 2**30,
        workspace / 2**30,
        slack / 2**30,
        pool / 2**30,
    )
    return pool


def init_pool(layer: FusedMoE, layouts: tuple) -> None:
    """Initialize the pool-only symmetric pool before weights exist; the world group must already
    be published because the pool is sized per world rank."""
    import zbal

    if dist.is_available() and dist.is_initialized():
        world_size, rank_id = dist.get_world_size(), dist.get_rank()
    else:
        parallel = get_parallel()
        try:
            world_size = parallel.world_group.world_size
            rank_id = parallel.world_group.rank_in_group
        except (RuntimeError, ValueError):
            logger.error(
                "ZBAL FuseEP: world group is not published while the pool is being created, so "
                "world_size/rank_id are unavailable; pool init cannot guess them from tp"
            )
            raise

    device_id = torch.npu.current_device()
    pool_bytes = _pool_bytes(layer, layouts)
    ip_port = envs.SGLANG_ZBAL_BOOTSTRAP_URL.get() or ""
    logger.info(
        "ZBAL FuseEP pool-only init: world=%s device=%s rank=%s pool=%.3f GiB bootstrap=%s",
        world_size,
        device_id,
        rank_id,
        pool_bytes / 2**30,
        ip_port or "<default>",
    )
    ok = zbal.zbal_init_static_pool(
        int(world_size), int(device_id), int(rank_id), pool_bytes, ip_port=ip_port
    )
    if not ok:
        logger.error(
            "ZBAL FuseEP pool-only init failed: world=%s rank=%s pool_bytes=%s",
            world_size,
            rank_id,
            pool_bytes,
        )
        raise RuntimeError("zbal_init_static_pool failed; check pool size / bootstrap url")


@contextmanager
def _pool_scope():
    """Torch allocations in this scope go to the symmetric pool; only wrap peer-visible tensors."""
    import zbal # type: ignore

    with torch.npu.use_mem_pool(zbal.get_static_pool()):
        yield


def pool_scope(layer: FusedMoE):
    """Symmetric-pool scope for expert weight allocation, opened by the quant method itself.

    ``FusedMoE.create_weights`` is on the GPU/NPU shared path, so it must stay unaware of zbal:
    the caller is ``ZbalPooledWeightsMixin`` around the quant method's ``create_weights``
    (``ModelSlimFusedMoEMethod`` and the A5 MXFP4 method). When the backend is not
    ``zbal_fuseep`` this returns ``nullcontext()``, so the shared path only pays one cached
    branch and a no-op context.

    The checkpoint-format weights have to be pool-resident from the start for the MX ABIs: their
    layout already is the ABI payload, so repacking reuses them in place and the pool never holds
    a second copy. ABI sets whose payload must be rebuilt (int8) budget one extra layer instead.
    """
    if not use_zbal_fuseep():
        return nullcontext()
    ZbalFuseEpState.instance().ensure_pool(layer, _scheme_layouts(layer))
    return _pool_scope()


class ZbalPooledWeightsMixin:
    """Run a quant method's ``create_weights`` inside the FuseEP symmetric-pool scope.

    Mixed in as the first base of a quant method whose ``create_weights`` allocates the expert
    weights; that keeps ``FusedMoE.create_weights`` - a GPU/NPU shared path - free of zbal code.

    A method defined in the subclass body shadows any mixin method of the same name, so the
    wrapping happens in ``__init_subclass__``: the subclass's own implementation is replaced by a
    wrapper that opens the scope, which leaves the subclass body untouched. The scope is a no-op
    (``nullcontext``) unless ``--moe-a2a-backend zbal_fuseep`` is selected.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        create_weights = cls.__dict__.get("create_weights")
        if create_weights is None:
            return

        @functools.wraps(create_weights)
        def pooled_create_weights(self, layer, *args, **kwargs):
            with pool_scope(layer):
                return create_weights(self, layer, *args, **kwargs)

        cls.create_weights = pooled_create_weights


def _layer_tag(layer: FusedMoE) -> object:
    """Stable identity for the built-layer count (layer_id when present)."""
    return getattr(layer, "layer_id", id(layer))


def _resolve_scale_attr(layer: FusedMoE, prefix: str, scale_attr: str | None) -> str:
    """Name of the checkpoint scale parameter feeding the ABI.

    The ABI writes ``{prefix}_weight_scale``; the A5 MXFP4 method registers the checkpoint's
    ``{prefix}_weight_scale_inv`` instead, so accept both rather than teaching every call site.
    """
    if scale_attr is not None:
        return scale_attr
    candidates = (f"{prefix}_weight_scale", f"{prefix}_weight_scale_inv")
    for candidate in candidates:
        if getattr(layer, candidate, None) is not None:
            return candidate
    logger.error(
        "ZBAL FuseEP: layer_id=%s %s scale missing (looked for %s)",
        getattr(layer, "layer_id", "?"),
        prefix,
        " / ".join(candidates),
    )
    raise ValueError(f"ZBAL FuseEP: {prefix} weight scale not found")


def prepare_weights(
    layer: FusedMoE,
    prefix: str,
    *,
    layout: str | None = None,
    scale_attr: str | None = None,
) -> None:
    """Lay expert weights out for the FuseEP ABI (fp8: ND payload + padded e8m0 scale; int8: zN
    payload + per-channel int64 scale; w4a8: packed e2m1 payload + padded e8m0 scale).

    ``layout`` overrides the ABI when a packed-uint8 payload cannot reveal it by dtype; it must
    agree with the pool reservation. ``scale_attr`` names the checkpoint scale parameter when the
    method does not use the ABI's ``{prefix}_weight_scale``.
    """
    if prefix not in _FUSE_EP_PREFIXES:
        logger.error(
            "ZBAL FuseEP: layer_id=%s unsupported weight prefix %r (the ABI expects %s)",
            getattr(layer, "layer_id", "?"),
            prefix,
            ", ".join(sorted(_FUSE_EP_PREFIXES)),
        )
        raise ValueError(f"ZBAL FuseEP: unsupported weight prefix {prefix!r}")

    # Quant methods may run process_weights_after_loading more than once (prefixed sub-methods and
    # the MXFP4/block-FP8 hooks both do); the ABI tensors are already in place after the first pass.
    if prefix in getattr(layer, "_zbal_fuseep_prefixes", set()):
        logger.info(
            "ZBAL FuseEP: layer_id=%s %s already prepared, skipping",
            getattr(layer, "layer_id", "?"),
            prefix,
        )
        return

    # The pool is sized from one expert shape; heterogeneous layers (e.g. an MTP layer with a
    # different expert set) are not supported, so refuse them instead of mis-sizing the pool.
    key = _weights_key(layer)
    pool_key = ZbalFuseEpState.instance().pool_shape_key
    if pool_key is not None and key != pool_key:
        logger.error(
            "ZBAL FuseEP: layer_id=%s expert shape %s differs from the pool shape %s; "
            "all MoE layers (MTP/nextn included) must share one shape",
            getattr(layer, "layer_id", "?"),
            key,
            pool_key,
        )
        raise ValueError("ZBAL FuseEP requires identical expert shapes across MoE layers")

    # FuseEP weights cover routed experts only: shared experts must take their own path, fusing
    # them in breaks the E dimension the ABI expects.
    fused_shared = getattr(layer, "num_fused_shared_experts", 0)
    if fused_shared:
        logger.error(
            "ZBAL FuseEP: layer_id=%s has %s fused shared experts; launch with "
            "--disable-shared-experts-fusion",
            getattr(layer, "layer_id", "?"),
            fused_shared,
        )
        raise ValueError("ZBAL FuseEP: shared-expert fusion must be disabled")
    if layer.num_local_experts != layer.num_experts // layer.moe_ep_size:
        logger.error(
            "ZBAL FuseEP: layer_id=%s local_experts=%s but num_experts/ep=%s/%s",
            getattr(layer, "layer_id", "?"),
            layer.num_local_experts,
            layer.num_experts,
            layer.moe_ep_size,
        )
        raise ValueError("ZBAL FuseEP requires contiguous experts without redundancy")

    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        if layer.moe_ep_size != world:
            logger.error(
                "ZBAL FuseEP: layer_id=%s ep_size=%s must span world=%s (FuseEP exchange is WORLD-wide)",
                getattr(layer, "layer_id", "?"),
                layer.moe_ep_size,
                world,
            )
            raise ValueError("ZBAL FuseEP requires moe_ep_size == world_size")

    scale_attr = _resolve_scale_attr(layer, prefix, scale_attr)
    weight = getattr(layer, f"{prefix}_weight", None)
    scale = getattr(layer, scale_attr, None)
    if weight is None or scale is None:
        logger.error(
            "ZBAL FuseEP: layer_id=%s %s weights missing (weight=%s %s=%s)",
            getattr(layer, "layer_id", "?"),
            prefix,
            weight is not None,
            scale_attr,
            scale is not None,
        )
        raise ValueError(f"ZBAL FuseEP: {prefix} weight/scale not found")

    # Layout/packing is fully delegated to zbal's repack: this only picks the ABI, checks it against
    # the pool reservation and puts the final ABI tensors in the pool scope.
    from zbal import zbal_moe_repack

    state = ZbalFuseEpState.instance()
    if not state.pool_inited:
        state.ensure_pool(layer, _scheme_layouts(layer))
    expected = state.pool_layouts[0 if prefix == "w13" else 1]
    if layout is None:
        try:
            layout = zbal_moe_repack.infer_layout(weight)
        except ValueError:
            # A packed e2m1 payload carried as uint8 has an fp8 payload's itemsize, so the dtype
            # cannot pick the ABI; only a declared w4a8 reservation may take over here.
            if expected != zbal_moe_repack.LAYOUT_W4A8 or weight.dtype is not torch.uint8:
                raise
            layout = expected
    if layout != expected:
        logger.error(
            "ZBAL FuseEP: layer_id=%s %s weight dtype gives layout %s but the pool was sized for "
            "%s; the reservation would be wrong",
            getattr(layer, "layer_id", "?"),
            prefix,
            layout,
            expected,
        )
        raise ValueError("ZBAL FuseEP: weight flavor drifted from the pool reservation")
    # The w4a8 route hands the kernel a float4_e2m1fn_x2 tensor (the runtime dtype map has no uint8),
    # so give the ABI an fp4 view of the checkpoint bytes; the storage stays the same.
    if layout == zbal_moe_repack.LAYOUT_W4A8 and weight.dtype is torch.uint8:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None:
            logger.error(
                "ZBAL FuseEP: layer_id=%s %s uses the w4a8 ABI but this torch has no "
                "float4_e2m1fn_x2 dtype",
                getattr(layer, "layer_id", "?"),
                prefix,
            )
            raise ValueError("ZBAL FuseEP: w4a8 requires the packed float4 dtype")
        weight = weight.view(fp4_dtype)
    tag = f"layer_id={getattr(layer, 'layer_id', '?')} {prefix}"
    payload_shape, scale_shape = zbal_moe_repack.repacked_shapes(weight, scale, layout=layout)
    payload_dtype, scale_dtype = zbal_moe_repack.repacked_dtypes(weight, layout=layout)
    # MX ABIs keep the checkpoint layout, so the pool-resident source IS the payload: reuse it
    # instead of allocating a second buffer the budget does not carry.
    reusable = (
        layout != zbal_moe_repack.LAYOUT_INT8
        and weight.is_contiguous()
        and weight.dtype == payload_dtype
        and tuple(weight.shape) == tuple(payload_shape)
    )
    if reusable:
        payload = weight
    else:
        # The ABI payload must live in the symmetric pool: FuseEP prefetches weights through peers'
        # GVAs, and addresses outside the pool are invisible to peers.
        with _pool_scope():
            payload = torch.empty(payload_shape, dtype=payload_dtype, device=weight.device)
    # The packed scale always replaces the checkpoint scale (k-tile padded), so it is pooled too.
    with _pool_scope():
        packed_scale = torch.empty(scale_shape, dtype=scale_dtype, device=weight.device)
    zbal_moe_repack.repack_into(
        weight,
        scale,
        payload,
        packed_scale,
        layout=layout,
        offset=getattr(layer, f"{prefix}_weight_offset", None),
        label=tag,
    )
    state.prepared_layers.add(_layer_tag(layer))

    setattr(
        layer,
        f"{prefix}_weight",
        torch.nn.Parameter(payload, requires_grad=False),
    )
    setattr(
        layer,
        f"{prefix}_weight_scale",
        torch.nn.Parameter(packed_scale, requires_grad=False),
    )
    if scale_attr != f"{prefix}_weight_scale":
        # The checkpoint-name parameter must not keep the source scale alive inside the pool.
        setattr(
            layer,
            scale_attr,
            torch.nn.Parameter(packed_scale, requires_grad=False),
        )
    processed = getattr(layer, "_zbal_fuseep_prefixes", set())
    layer._zbal_fuseep_prefixes = processed | {prefix}
    # From here on forward() has to go through dispatch_ffn_combine: the weights are in the FuseEP
    # ABI, so any other MoE path would misread them.
    layer._use_zbal_fuseep = True
    logger.info(
        "ZBAL FuseEP: layer_id=%s %s layout=%s payload=%s scale=%s",
        getattr(layer, "layer_id", "?"),
        prefix,
        layout,
        tuple(payload.shape),
        tuple(packed_scale.shape),
    )
    # The checkpoint-format copy is dead once the ABI tensors exist; drop it and hand its native
    # blocks back before the next layer packs (the pool must not see either of them).
    del weight, scale
    torch.npu.empty_cache()


def _workspace_shape(layer: FusedMoE) -> tuple[int, ...]:
    """Inputs of the workspace query, with maxOutputSize at the worst case ``bound × EP × topk``."""
    bound = _bound_tokens()
    return (
        bound,
        layer.hidden_size,
        2 * layer.intermediate_size_per_partition,
        layer.num_local_experts,
        layer.moe_ep_size,
        layer.top_k,
        bound * layer.moe_ep_size * layer.top_k,
    )


def _quant_mode_of(layout: str) -> int:
    """DFC quant mode: the route only splits packed-4-bit (w4a8) weights from the 8-bit ones."""
    return 6 if layout == "w4a8" else 1


def _workspace_bytes(layer: FusedMoE, layouts: tuple) -> int:
    """Resident workspace bytes sized from the prefill budget, which also bounds every decode
    launch (decode reserves its own, smaller, row count at launch)."""
    from zbal import Buffer

    bs, h, ffn_dim, nle, ep, topk, max_output_size = _workspace_shape(layer)
    return Buffer.get_dfc_workspace_size(
        bs,
        h,
        ffn_dim,
        nle,
        ep,
        topk,
        max_output_size=max_output_size,
        mode="prefill",
        # w4a8 rows carry 4-bit weights, so their row/slot buffers differ from the 8-bit routes.
        quant_mode=_quant_mode_of(layouts[0]),
    )


def _create_buffer(layer: FusedMoE, layouts: tuple):
    """Create the FuseEP Buffer and allocate the resident workspace (called once by ``ensure_pool``)."""
    from zbal import Buffer

    group = get_parallel().moe_ep_group.device_group
    buf = Buffer(group, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=False)
    ws_bytes = _workspace_bytes(layer, layouts)
    buf.malloc_dfc_workspace(ws_bytes)
    return buf


def _check_startup_ledger() -> None:
    """First forward: reconcile the built layer count with the reservation and log occupancy.

    The pool is sized before the model is built (from ``SGLANG_ZBAL_DFC_MOE_LAYERS`` and the first
    layer's flavor), so the only place the reservation can be checked is after every MoE layer has
    gone through the ABI. A mismatch means the pool is either short (would have OOM'd) or
    over-reserved (wasted HBM), hence the hard error.
    """
    state = ZbalFuseEpState.instance()
    if state.startup_checked:
        return
    state.startup_checked = True
    import zbal # type: ignore

    used, total = zbal.static_pool_stats()
    peak = zbal.static_pool_peak()
    ledger = state.ledger or {}
    built = len(state.prepared_layers)
    moe_layers = ledger.get("moe_layers")
    waste = total - used
    logger.info(
        "ZBAL FuseEP pool occupancy: layout=%s built_layers=%s/%s used=%.3f GiB peak=%.3f GiB "
        "total=%.3f GiB waste=%.3f GiB (%.2f%%) weights_budget=%.3f GiB workspace=%.3f GiB",
        state.pool_layouts,
        built,
        moe_layers,
        used / 2**30,
        peak / 2**30,
        total / 2**30,
        waste / 2**30,
        (100.0 * waste / total if total else 0.0),
        ledger.get("weights", 0) / 2**30,
        ledger.get("workspace", 0) / 2**30,
    )
    if moe_layers is not None and built != moe_layers:
        logger.error(
            "ZBAL FuseEP: %s MoE layers went through the FuseEP ABI but "
            "SGLANG_ZBAL_DFC_MOE_LAYERS=%s; the pool reservation does not match the model",
            built,
            moe_layers,
        )
        raise ValueError("ZBAL FuseEP: built MoE layer count differs from SGLANG_ZBAL_DFC_MOE_LAYERS")


def forward(layer: FusedMoE, hidden_states: torch.Tensor, topk_output: TopKOutput) -> torch.Tensor:
    """Single-layer MoE forward: dispatch + GMM1 + SwiGLU + GMM2 + combine all inside the op."""
    if getattr(layer, "_zbal_fuseep_prefixes", None) != _FUSE_EP_PREFIXES:
        logger.error(
            "ZBAL FuseEP: layer_id=%s weights not prepared (%s); expected repacked expert weights",
            getattr(layer, "layer_id", "?"),
            getattr(layer, "_zbal_fuseep_prefixes", None),
        )
        raise ValueError("ZBAL FuseEP: expert weights were not laid out for FuseEP")
    _check_startup_ledger()

    tokens = hidden_states.shape[0]
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    # Scenario comes from the rank-synchronous extend flag, never from the token count: idle ranks
    # of a prefill step still launch prefill, and MLP sync refreshes the flag once per step.
    mode = "prefill" if get_is_extend_in_batch() else "decode"
    bound = _bound_tokens()
    if tokens > bound:
        logger.error(
            "ZBAL FuseEP: layer_id=%s local tokens=%s exceeds the reserved prefill bound %s "
            "(--chunked-prefill-size / --max-prefill-tokens); the workspace is sized from it",
            getattr(layer, "layer_id", "?"),
            tokens,
            bound,
        )
        raise ValueError("ZBAL FuseEP: local token count exceeds the reserved prefill bound")

    if tokens == 0:
        hidden_states = hidden_states.new_zeros((1, layer.hidden_size))
        topk_ids = torch.zeros((1, layer.top_k), dtype=torch.int32, device=hidden_states.device)
        topk_weights = torch.zeros((1, layer.top_k), dtype=torch.float32, device=hidden_states.device)

    rows = max(tokens, 1)
    max_output_size = (bound if mode == "prefill" else rows) * layer.moe_ep_size * layer.top_k

    state = ZbalFuseEpState.instance()
    # The route selects the row/slot payload widths and the w4a8 tiling key, so the launch has to
    # use the flavor the pool was sized for: w4a8 -> 6, the 8-bit routes -> 1 (fp8 keys off dtype).
    quant_mode = _quant_mode_of(state.pool_layouts[0])
    # The w4a8 epilogue applies the model's SwiGLU clamp (a deployment requirement for this route);
    # the 8-bit routes keep the previous behaviour (no clamp) until they are validated the same way.
    swiglu_limit = 0.0
    if quant_mode == 6:
        swiglu_limit = (
            getattr(getattr(layer, "moe_runner_config", None), "swiglu_limit", None) or 0.0
        )
    out, _ = state.buffer.dispatch_ffn_combine(
        x=hidden_states.contiguous(),
        topk_idx=topk_ids.to(torch.int32).contiguous(),
        weight1=layer.w13_weight,
        scale1=layer.w13_weight_scale,
        weight2=layer.w2_weight,
        scale2=layer.w2_weight_scale,
        topk_weights=topk_weights.float().contiguous(),
        max_output_size=max_output_size,
        num_experts=layer.num_experts,
        quant_mode=quant_mode,
        swiglu_limit=float(swiglu_limit),
        mode=mode,
    )
    return out[:tokens]
