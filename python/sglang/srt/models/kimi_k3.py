# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/model_executor/models/kimi_linear.py

from collections import deque
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, make_layers
from sglang.srt.utils.common import (
    BumpAllocator,
    add_prefix,
    get_int_env_var,
    set_weight_attrs,
)


class _NoopRotaryEmbedding(nn.Module):
    """Preserve Kimi MLA's skip_rope semantics on backends that call RoPE."""

    def forward(self, positions, query, key):
        return query, key


class KimiMLAAttention(DeepseekV2AttentionMLA):
    """DeepSeek MLA with K3's optional per-head output gate."""

    def __init__(self, *args, config: KimiLinearConfig, prefix: str = "", **kwargs):
        super().__init__(*args, config=config, prefix=prefix, **kwargs)
        if is_npu() and self.rotary_emb is None:
            self.rotary_emb = _NoopRotaryEmbedding()
        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
        self._output_gate = None
        if self.use_output_gate:
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=self.quant_config,
                prefix=add_prefix("g_proj", prefix),
                tp_rank=get_parallel().attn_tp_rank,
                tp_size=get_parallel().attn_tp_size,
            )

            # All MLA backends funnel their local head output through o_proj.
            # Wrapping that call preserves the optimized NPU attention path and
            # applies K3's gate at the exact point required by the reference.
            ungated_o_proj_forward = self.o_proj.forward

            def gated_o_proj_forward(x: torch.Tensor):
                assert self._output_gate is not None
                return ungated_o_proj_forward(x * self._output_gate)

            self.o_proj.forward = gated_o_proj_forward

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        **kwargs,
    ):
        if self.use_output_gate:
            self._output_gate = torch.sigmoid(self.g_proj(hidden_states)[0])

        # DeepSeek MLA's NPU prepare path consumes its q/kv latent projection
        # through the attention TP context. K3 does not use DeepSeek's decoder
        # LayerCommunicator, so publish the input explicitly (the same pattern
        # used by the standalone Kimi K2.5 Eagle MLA layer).
        attn_tp_context = get_attn_tp_context()
        attn_tp_context.set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, self.prepare_qkv_latent)
        )
        try:
            return super().forward(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                **kwargs,
            )
        finally:
            attn_tp_context.clear_attn_inputs()


class SituAndMul(nn.Module):
    def __init__(self, beta: float, linear_beta: Optional[float]) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        output_dtype = x.dtype
        gate = gate.float()
        up = up.float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(output_dtype)


class KimiMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        activation_situ_beta: float = 1.0,
        activation_situ_linear_beta: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
        )
        if hidden_act == "situ":
            self.act_fn = SituAndMul(
                beta=activation_situ_beta,
                linear_beta=activation_situ_linear_beta,
            )
        elif hidden_act == "silu":
            from sglang.srt.layers.activation import SiluAndMul

            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(gate_up))[0]


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_idx: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        moe_renormalize = config.moe_renormalize
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream

        if config.hidden_act not in {"silu", "situ"}:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=getattr(
                config, "routed_expert_hidden_size", config.hidden_size
            ),
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_idx,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            activation=config.hidden_act,
            prefix=add_prefix("experts", prefix),
        )

        self.routed_expert_hidden_size = getattr(
            config, "routed_expert_hidden_size", None
        )
        if self.routed_expert_hidden_size is not None:
            self.routed_expert_down_proj = ReplicatedLinear(
                config.hidden_size,
                self.routed_expert_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("routed_expert_down_proj", prefix),
            )
            self.routed_expert_up_proj = ReplicatedLinear(
                self.routed_expert_hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("routed_expert_up_proj", prefix),
            )
            self.routed_expert_norm = RMSNorm(
                self.routed_expert_hidden_size, eps=config.rms_norm_eps
            )

        self.topk = TopK(
            top_k=config.num_experts_per_token,
            renormalize=moe_renormalize,
            use_grouped_topk=True,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
            # and requires the output format to be standard. We use quant_config to determine the output format.
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )

        if self.num_shared_experts is not None:
            intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                activation_situ_beta=getattr(config, "activation_situ_beta", 1.0),
                activation_situ_linear_beta=getattr(
                    config, "activation_situ_linear_beta", None
                ),
            )

        # Host-offloaded expert weights: two modes.
        #
        # Decode: 20-slot (default, env-tunable) HBM LRU cache per layer. Sync
        # swap-in of topk-selected experts; topk_ids remapped to slot ids;
        # group_list + hidden_states rebuilt to slot order; kernel output
        # unpermuted back to dispatch order before combine.
        #
        # Prefill: full num_local_experts HBM buffer per layer. Async prefetch
        # of layer+N's full expert set on a side stream while the current layer
        # computes. After compute, the layer's buffer is freed immediately.
        self._offload_inited = False
        self._decode_cache_slots = get_int_env_var(
            "SGLANG_KIMI_DECODE_CACHE_SLOTS", 20
        )
        self._prefetch_layers = get_int_env_var("SGLANG_KIMI_PREFETCH_LAYERS", 10)
        self._expert_weight_names = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        )
        self._host_weights = {}
        self._orig_attrs = {}
        # Decode-mode state (allocated lazily in _switch_mode)
        self._decode_cache = {}
        self._slot_to_expert = [-1] * self._decode_cache_slots
        self._expert_to_slot = {}
        self._lru_deque = deque()
        # Prefill-mode state (allocated lazily in start_prefill)
        self._prefill_buffer = {}
        self._prefill_event = None
        self._prefill_loaded = False
        # Mode tracking: _mode is the persistent current mode; _current_mode
        # is set per-forward by KimiLinearModel before calling this layer.
        self._mode = None
        self._current_mode = None

    def _init_host_offload(self) -> None:
        """Pin host-resident expert weights and promote ``self.experts`` to a
        cache-aware subclass. Does NOT allocate HBM buffers; those are created
        lazily in ``_switch_mode``. Must run after ``process_weights_after_loading``.
        """
        target_device = torch.device("cuda", torch.cuda.current_device())
        for name in self._expert_weight_names:
            param = getattr(self.experts, name, None)
            if param is None:
                raise AttributeError(
                    f"experts layer has no weight '{name}'; cannot init host offload"
                )
            host_data = param.data
            if host_data.device.type != "cpu":
                raise RuntimeError(
                    f"KimiMoE host-offload expects expert weight '{name}' to "
                    f"be on CPU initially, but found device {host_data.device}. "
                    f"Ensure the weight loader keeps expert weights on CPU."
                )
            if not host_data.is_pinned():
                host_data = host_data.pin_memory()
            self._host_weights[name] = host_data
            self._orig_attrs[name] = param
        orig_params = {
            name: self._orig_attrs[name] for name in self._expert_weight_names
        }
        orig_num = self.experts.num_local_experts
        orig_ep = self.experts.moe_ep_rank

        # Build a cache-aware subclass inline (closure-scoped, no module-level
        # symbol). Properties gate num_local_experts / moe_ep_rank / each weight
        # on ``self._active_mode`` (None / "prefill" / "decode") so reads return
        # the right buffer without mutating instance attrs.
        base_cls = type(self.experts)
        weight_names = self._expert_weight_names

        def _weight_property(name: str):
            def getter(s):
                mode = getattr(s, "_active_mode", None)
                if mode == "prefill":
                    return s._prefill_buffer[name]
                elif mode == "decode":
                    return s._decode_cache[name]
                return s._orig_params[name]
            return property(getter)

        namespace = {
            "num_local_experts": property(
                lambda s: s._decode_cache_slots
                if getattr(s, "_active_mode", None) == "decode"
                else s._orig_num_local_experts
            ),
            "moe_ep_rank": property(
                lambda s: 0
                if getattr(s, "_active_mode", None) == "decode"
                else s._orig_moe_ep_rank
            ),
        }
        for name in weight_names:
            namespace[name] = _weight_property(name)

        cache_aware_cls = type(
            f"{base_cls.__name__}_KimiCacheAware",
            (base_cls,),
            namespace,
        )
        self.experts.__class__ = cache_aware_cls
        self.experts._active_mode = None
        self.experts._orig_params = orig_params
        self.experts._orig_num_local_experts = orig_num
        self.experts._orig_moe_ep_rank = orig_ep
        self.experts._decode_cache_slots = self._decode_cache_slots
        # _prefill_buffer / _decode_cache dicts are set lazily by _switch_mode;
        # initialize empty dicts so property getters don't KeyError on the
        # "else" (None) branch before the first mode switch.
        self.experts._prefill_buffer = {}
        self.experts._decode_cache = {}

        # Drop shadowing instance attrs so class-level properties take effect.
        self.experts.__dict__.pop("num_local_experts", None)
        self.experts.__dict__.pop("moe_ep_rank", None)

        self._orig_attrs["num_local_experts"] = orig_num
        self._orig_attrs["moe_ep_rank"] = orig_ep
        self._offload_inited = True

    # ------------------------------------------------------------------ #
    # Mode switching: allocate / free HBM buffers when mode changes.
    # ------------------------------------------------------------------ #

    def _switch_mode(self, new_mode: Optional[str]) -> None:
        """Switch between prefill (full-112 HBM buffer) and decode (20-slot
        LRU cache). Frees the old buffer and allocates the new one. No-op if
        ``new_mode`` equals the current ``self._mode``.
        """
        if self._mode == new_mode:
            return
        # Free old buffer.
        if self._mode == "prefill":
            self._free_prefill_buffer()
        elif self._mode == "decode":
            self._free_decode_cache()
        # Allocate new buffer (lazy: prefill buffer is filled by start_prefill;
        # decode cache is populated on-demand by _ensure_experts_cached).
        if new_mode == "prefill":
            self._alloc_prefill_buffer()
        elif new_mode == "decode":
            self._alloc_decode_cache()
        self._mode = new_mode

    def _alloc_decode_cache(self) -> None:
        target_device = torch.device("cuda", torch.cuda.current_device())
        for name in self._expert_weight_names:
            param = self._orig_attrs[name]
            cache = torch.empty(
                (self._decode_cache_slots, *param.data.shape[1:]),
                dtype=param.data.dtype,
                device=target_device,
            )
            self._decode_cache[name] = Parameter(cache, requires_grad=False)
        self.experts._decode_cache = self._decode_cache
        # Reset LRU bookkeeping.
        self._slot_to_expert = [-1] * self._decode_cache_slots
        self._expert_to_slot = {}
        self._lru_deque.clear()

    def _free_decode_cache(self) -> None:
        self._decode_cache.clear()
        self.experts._decode_cache = {}
        self._slot_to_expert = [-1] * self._decode_cache_slots
        self._expert_to_slot = {}
        self._lru_deque.clear()

    def _alloc_prefill_buffer(self) -> None:
        if self._prefill_buffer:
            return  # already allocated (e.g., by start_prefill)
        target_device = torch.device("cuda", torch.cuda.current_device())
        for name in self._expert_weight_names:
            param = self._orig_attrs[name]
            buf = torch.empty_like(param.data, device=target_device)
            self._prefill_buffer[name] = Parameter(buf, requires_grad=False)
        self.experts._prefill_buffer = self._prefill_buffer
        self._prefill_loaded = False
        self._prefill_event = None

    def _free_prefill_buffer(self) -> None:
        self._prefill_buffer.clear()
        self.experts._prefill_buffer = {}
        self._prefill_loaded = False
        self._prefill_event = None

    def free_prefill_buffer(self) -> None:
        """Public entry: free this layer's full-112 HBM buffer after prefill
        compute completes. Called by KimiLinearModel.forward per layer.
        """
        self._free_prefill_buffer()

    # ------------------------------------------------------------------ #
    # Prefill async prefetch (called by KimiLinearModel).
    # ------------------------------------------------------------------ #

    def start_prefill(self, stream: torch.cuda.Stream) -> None:
        """Async H2D copy of all num_local_experts weights for this layer on
        ``stream``. Records an event for later ``wait_prefill`` synchronization.
        """
        if not self._offload_inited:
            self._init_host_offload()
        if not self._prefill_buffer:
            self._alloc_prefill_buffer()
        if self._prefill_loaded:
            return
        with torch.cuda.stream(stream):
            for name in self._expert_weight_names:
                self._prefill_buffer[name].data.copy_(
                    self._host_weights[name], non_blocking=True
                )
        self._prefill_event = torch.cuda.Event()
        self._prefill_event.record(stream)

    def wait_prefill(self) -> None:
        """Block the default stream until the prefetch H2D copy for this layer
        has completed. Marks the buffer as ready for kernel use.
        """
        if self._prefill_event is not None:
            self._prefill_event.wait()
            self._prefill_event = None
        self._prefill_loaded = True

    # ------------------------------------------------------------------ #
    # Decode-mode LRU swap-in (unchanged logic, slots from env var).
    # ------------------------------------------------------------------ #

    def _ensure_experts_cached(self, local_topk_ids: torch.Tensor) -> torch.Tensor:
        """LRU swap-in: ensure unique local experts in ``local_topk_ids`` are
        in the HBM cache. Returns a remap tensor mapping local_expert_id -> slot_id
        (-1 for unmapped).
        """
        ids_cpu = local_topk_ids.view(-1).to("cpu", dtype=torch.int64).tolist()
        unique_ids = {int(x) for x in ids_cpu if x >= 0}
        if len(unique_ids) > self._decode_cache_slots:
            raise RuntimeError(
                f"KimiMoE host-offload cache: batch requests {len(unique_ids)} "
                f"unique experts but only {self._decode_cache_slots} slots exist. "
                f"batch_size=1 decode (16 unique experts) is the supported config."
            )
        for eid in unique_ids:
            if eid in self._expert_to_slot:
                slot = self._expert_to_slot[eid]
                self._lru_deque.remove(slot)
                self._lru_deque.appendleft(slot)
            else:
                free_slot = next(
                    (i for i, v in enumerate(self._slot_to_expert) if v == -1),
                    None,
                )
                if free_slot is None:
                    free_slot = self._lru_deque.pop()
                    old_eid = self._slot_to_expert[free_slot]
                    del self._expert_to_slot[old_eid]
                    self._slot_to_expert[free_slot] = -1
                for name in self._expert_weight_names:
                    self._decode_cache[name].data[free_slot].copy_(
                        self._host_weights[name][eid], non_blocking=False
                    )
                self._slot_to_expert[free_slot] = eid
                self._expert_to_slot[eid] = free_slot
                self._lru_deque.appendleft(free_slot)
        num_local_experts = self._orig_attrs["num_local_experts"]
        remap = torch.full(
            (num_local_experts,),
            -1,
            dtype=torch.int32,
            device=local_topk_ids.device,
        )
        for eid, slot in self._expert_to_slot.items():
            remap[eid] = slot
        return remap

    # ------------------------------------------------------------------ #
    # Decode-mode dispatch rebuild (group_list + hidden_states reorder).
    # ------------------------------------------------------------------ #

    def _rebuild_dispatch_for_slots(self, dispatch_output, remap: torch.Tensor):
        """Rebuild a DeepEPNormalDispatchOutput for the decode HBM cache.

        NPU ``npu_grouped_matmul`` iterates experts via ``group_list`` and
        indexes ``weight[expert_id]``. Our decode cache has limited slots, so we
        must:
        - Rebuild ``num_recv_tokens_per_expert`` to slot-ordered counts
        - Reorder ``hidden_states`` (and its scale) from original-local-id order
          to slot order
        - Remap ``topk_ids`` to slot ids (metadata for combine_input)

        Returns ``(new_dispatch, slot_perm)`` where ``slot_perm`` maps
        new_row_index (slot order) -> old_row_index (dispatch order); used to
        unpermute the kernel output before combine.
        """
        num_recv_per_expert = getattr(
            dispatch_output, "num_recv_tokens_per_expert", None
        )
        if num_recv_per_expert is None:
            old_ids = dispatch_output.topk_ids
            safe_ids = torch.where(old_ids == -1, 0, old_ids)
            new_ids = torch.where(
                old_ids == -1,
                old_ids,
                remap[safe_ids].to(old_ids.dtype),
            )
            return dispatch_output._replace(topk_ids=new_ids), None

        num_local = len(num_recv_per_expert)
        offsets = [0] * (num_local + 1)
        for i in range(num_local):
            offsets[i + 1] = offsets[i] + num_recv_per_expert[i]

        new_num_recv = [0] * self._decode_cache_slots
        perm_indices: list = []
        for slot_id in range(self._decode_cache_slots):
            orig_id = self._slot_to_expert[slot_id]
            if 0 <= orig_id < num_local:
                count = num_recv_per_expert[orig_id]
                new_num_recv[slot_id] = count
                start = offsets[orig_id]
                perm_indices.extend(range(start, start + count))

        device = dispatch_output.hidden_states.device
        slot_perm = torch.tensor(perm_indices, dtype=torch.int64, device=device)

        if slot_perm.numel() > 0:
            new_hidden_states = dispatch_output.hidden_states[slot_perm]
            old_scale = dispatch_output.hidden_states_scale
            new_scale = old_scale[slot_perm] if old_scale is not None else None
        else:
            new_hidden_states = dispatch_output.hidden_states[:0]
            old_scale = dispatch_output.hidden_states_scale
            new_scale = old_scale[:0] if old_scale is not None else None

        old_ids = dispatch_output.topk_ids
        safe_ids = torch.where(old_ids == -1, 0, old_ids)
        new_ids = torch.where(
            old_ids == -1,
            old_ids,
            remap[safe_ids].to(old_ids.dtype),
        )

        new_dispatch = dispatch_output._replace(
            hidden_states=new_hidden_states,
            hidden_states_scale=new_scale,
            topk_ids=new_ids,
            num_recv_tokens_per_expert=new_num_recv,
        )
        return new_dispatch, slot_perm

    # ------------------------------------------------------------------ #
    # Experts forward: prefill (full buffer, no remap) vs decode (cache).
    # ------------------------------------------------------------------ #

    def _experts_forward_prefill(
        self, routed_hidden_states: torch.Tensor, topk_output
    ) -> torch.Tensor:
        """Prefill path: full num_local_experts buffer already loaded by async
        prefetch. No remap, no rebuild, no unpermute — kernel sees the original
        full weight tensor + original topk_ids + original group_list.
        """
        if not self._offload_inited:
            self._init_host_offload()
        self.wait_prefill()
        self.experts._active_mode = "prefill"
        dispatch_output = self.experts.dispatcher.dispatch(
            hidden_states=routed_hidden_states, topk_output=topk_output
        )
        try:
            combine_input = self.experts.run_moe_core(dispatch_output)
        finally:
            self.experts._active_mode = None
        return self.experts.dispatcher.combine(combine_input=combine_input)

    def _experts_forward_decode(
        self, routed_hidden_states: torch.Tensor, topk_output
    ) -> torch.Tensor:
        """Decode path: limited-slot LRU cache, sync swap-in, rebuild dispatch
        output to slot order, kernel run on cache weights, unpermute output back
        to dispatch order before combine.
        """
        dispatch_output = self.experts.dispatcher.dispatch(
            hidden_states=routed_hidden_states, topk_output=topk_output
        )
        old_ids = dispatch_output.topk_ids
        remap = self._ensure_experts_cached(old_ids)
        remapped_dispatch, slot_perm = self._rebuild_dispatch_for_slots(
            dispatch_output, remap
        )
        self.experts._active_mode = "decode"
        try:
            combine_input = self.experts.run_moe_core(remapped_dispatch)
        finally:
            self.experts._active_mode = None
        if slot_perm is not None and slot_perm.numel() > 0:
            kernel_output = combine_input.hidden_states
            unpermuted = torch.empty_like(kernel_output)
            unpermuted[slot_perm] = kernel_output
            combine_input = combine_input._replace(hidden_states=unpermuted)
        return self.experts.dispatcher.combine(combine_input=combine_input)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        # Lazy init + mode switch (mode is set by KimiLinearModel per-forward).
        if not self._offload_inited:
            self._init_host_offload()
        if self._current_mode != self._mode:
            self._switch_mode(self._current_mode)

        experts_fn = (
            self._experts_forward_prefill
            if self._current_mode == "prefill"
            else self._experts_forward_decode
        )

        shared_output = None
        routed_hidden_states = hidden_states
        if self.routed_expert_hidden_size is not None:
            routed_hidden_states = self.routed_expert_down_proj(hidden_states)[0]

        if (
            self.alt_stream is not None
            and self.num_shared_experts is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            shared_output = self.shared_experts(hidden_states.clone())

            with torch.cuda.stream(self.alt_stream):
                router_logits, _ = self.gate(hidden_states)
                topk_output = self.topk(hidden_states, router_logits)
                final_hidden_states = experts_fn(
                    routed_hidden_states, topk_output
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            if self.num_shared_experts is not None and hidden_states.shape[0] > 0:
                shared_output = self.shared_experts(hidden_states)
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            final_hidden_states = experts_fn(
                routed_hidden_states, topk_output
            )

        if self.routed_expert_hidden_size is not None:
            final_hidden_states = self.routed_expert_norm(final_hidden_states)
            final_hidden_states = self.routed_expert_up_proj(final_hidden_states)[0]

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)


class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.attn_tp_size = get_parallel().attn_tp_size
        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.v_head_dim
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.use_full_rank_gate = config.linear_attn_config.get(
            "use_full_rank_gate", False
        )

        # TODO: support fusion with quant
        self.do_fuse_qkvbfg = quant_config is None and not self.use_full_rank_gate

        if self.do_fuse_qkvbfg:
            # Fuse: q, k, v, beta (column parallel) + f_a, g_a (replicated)
            self.qkvb_sizes = [
                projection_size,
                projection_size,
                projection_size,
                self.num_heads,
            ]
            self.fg_sizes = [self.head_dim, self.head_dim]

            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
                self.hidden_size,
                self.qkvb_sizes,  # Column parallel
                self.fg_sizes,  # Replicated: f_a, g_a
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkvbfg_a_proj",
            )
            self.split_sizes = [
                3 * projection_size // self.tp_size,  # qkv
                self.num_heads // self.tp_size,  # beta
                2 * self.head_dim,  # f_a, g_a
            ]
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(
                2, self.head_dim, projection_size, dtype=config.dtype
            )
        else:
            # Unfused path: separate QKVParallelLinear
            attn_tp_rank = get_parallel().attn_tp_rank
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                self.num_k_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=self.attn_tp_size,
                v_head_size=self.head_v_dim,
                prefix=f"{prefix}.qkv_proj",
            )

            self.f_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_a_proj",
            )

            self.f_b_proj = ColumnParallelLinear(
                self.head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_b_proj",
            )

            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.b_proj",
            )

            if self.use_full_rank_gate:
                self.g_proj = ColumnParallelLinear(
                    self.hidden_size,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_proj",
                )
            else:
                self.g_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_a_proj",
                )
                self.g_b_proj = ColumnParallelLinear(
                    self.head_dim,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_b_proj",
                )

        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.qkv_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )

        def load_a_log(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            # K3 stores A_log as [num_heads], while RadixLinearAttention keeps
            # the local shard as [1, 1, local_num_heads, 1].
            loaded_weight = loaded_weight.flatten()
            start = get_parallel().attn_tp_rank * self.local_num_heads
            local_weight = loaded_weight.narrow(0, start, self.local_num_heads)
            param.data.copy_(local_weight.reshape_as(param))

        set_weight_attrs(self.A_log, {"weight_loader": load_a_log})

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        conv_weights = self.qkv_conv1d.weight.squeeze(1)
        bias = self.qkv_conv1d.bias

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def forward_qkvbfg(self, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)

        # Compute beta, forget_gate, and g_proj_states
        beta = self.b_proj(hidden_states)[0]
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        if self.use_full_rank_gate:
            g_proj_states = self.g_proj(hidden_states)[0]
        else:
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward_qkvbfg_fused(self, hidden_states: torch.Tensor):
        # Single fused projection for all: qkv + beta + f_a + g_a
        fused_states = self.fused_qkvbfg_a_proj(hidden_states)

        qkv, beta, fg_a_states = torch.split(
            fused_states,
            self.split_sizes,
            dim=-1,
        )

        # use batch matmul to calculate forget_gate and g_proj_states
        forget_gate, g_proj_states = self.fused_fg_b_proj(
            fg_a_states.view(-1, 2, self.head_dim).transpose(0, 1)
        )

        return (
            qkv,
            beta,
            forget_gate,
            g_proj_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> None:
        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(
                hidden_states
            )
        else:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                hidden_states
            )

        # For prefill: raw gate is passed to chunk_kda_fwd, which fuses gate
        # activation with chunk_local_cumsum (kda_gate_chunk_cumsum kernel).
        # For decode: gate activation is handled inside fused_recurrent kernel.
        if not forward_batch.forward_mode.is_decode():
            forget_gate = forget_gate.unflatten(
                -1, (-1, self.head_dim)
            )  # [T, H*K] -> [T, H, K]
            # CUDA chunk KDA expects beta to be pre-sigmoided. The NPU
            # recurrent path fuses both sigmoid gates and therefore consumes
            # the raw beta projection, matching the decode path.
            if not is_npu():
                beta = beta.float().sigmoid()
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        norm_gate = g_proj_states.unflatten(
            -1, (-1, self.head_dim)
        )  # ... (h d) -> ... h d
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)  # 1 n h d -> n (h d)

        return self.o_proj(core_attn_out)[0]


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.alt_stream = alt_stream
        self.layer_idx = layer_idx

        self.is_moe = config.is_moe

        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiDeltaAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = KimiMLAAttention(
                layer_id=layer_idx,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                config=config,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                skip_rope=True,
            )

        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiMoE(
                config=config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.block_sparse_moe",
                alt_stream=self.alt_stream,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=getattr(config, "activation_situ_beta", 1.0),
                activation_situ_linear_beta=getattr(
                    config, "activation_situ_linear_beta", None
                ),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # K3 mixes the running prefix and one residual per attention block with
        # learned scalar projections.  These parameters exist in every K3
        # checkpoint and are not the ordinary two-add Transformer residuals.
        self.use_attn_residuals = (
            getattr(config, "attn_res_block_size", None) is not None
        )
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp_res_proj",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
            )

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def _forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        block_residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum = hidden_states
        if block_residual is None:
            block_residual = hidden_states.new_empty(
                hidden_states.shape[0], 0, hidden_states.shape[1]
            )

        if block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (block_residual, prefix_sum.unsqueeze(1)), dim=1
            )
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        hidden_states = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, block_residual


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Module,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply K3's learned softmax mixing over block residual streams."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    variance = values_float.square().mean(-1, keepdim=True)
    normalized = values_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    probabilities = (normalized * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probabilities, values_float).squeeze(1).to(values.dtype)


class KimiLinearModel(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream()
        self._prefetch_layers = get_int_env_var("SGLANG_KIMI_PREFETCH_LAYERS", 10)
        self._prefetch_stream = torch.cuda.Stream()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: KimiDecoderLayer(
                layer_idx=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.use_attn_residuals = (
                getattr(config, "attn_res_block_size", None) is not None
            )
            if self.use_attn_residuals:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = PPMissingLayer()
            self.use_attn_residuals = False

        world_size = get_parallel().tp_size
        assert (
            config.num_attention_heads % world_size == 0
        ), "num_attention_heads must be divisible by world_size"

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
            if self.use_attn_residuals:
                residual = hidden_states.new_empty(
                    hidden_states.shape[0], 0, hidden_states.shape[1]
                )
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        total_num_layers = self.end_layer - self.start_layer
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=device,
        )
        # TODO: capture aux hidden states
        aux_hidden_states = []
        is_prefill = forward_batch.forward_mode.is_prefill()
        N = self._prefetch_layers if is_prefill else 0

        # Set _current_mode on every MoE layer so KimiMoE.forward can switch
        # buffers (full-112 prefill vs 20-slot decode cache).
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if getattr(layer, "is_moe", False):
                layer.block_sparse_moe._current_mode = (
                    "prefill" if is_prefill else "decode"
                )

        # Pre-trigger async prefetch for the first N MoE layers so their
        # full-112 expert sets start loading before the compute loop begins.
        if is_prefill and N > 0:
            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                if not getattr(layer, "is_moe", False):
                    continue
                if i - self.start_layer >= N:
                    break
                layer.block_sparse_moe.start_prefill(self._prefetch_stream)

        for i in range(self.start_layer, self.end_layer):
            ctx = get_global_expert_distribution_recorder().with_current_layer(i)
            with ctx:
                layer = self.layers[i]
                moe = (
                    layer.block_sparse_moe
                    if getattr(layer, "is_moe", False)
                    else None
                )
                if is_prefill and N > 0 and moe is not None:
                    # Wait for this layer's prefetch to finish before compute.
                    moe.wait_prefill()
                    # Trigger prefetch for layer i+N so its H2D copy overlaps
                    # with this layer's compute.
                    target_i = i + N
                    if target_i < self.end_layer:
                        target_moe = (
                            self.layers[target_i].block_sparse_moe
                            if getattr(self.layers[target_i], "is_moe", False)
                            else None
                        )
                        if target_moe is not None:
                            target_moe.start_prefill(self._prefetch_stream)
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    zero_allocator=zero_allocator,
                )
                # After prefill compute, free this layer's full-112 buffer to
                # cap HBM usage at ~(N+1) concurrent buffers.
                if is_prefill and moe is not None:
                    moe.free_prefill_buffer()

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if self.use_attn_residuals:
                    hidden_states = _apply_attn_res(
                        hidden_states,
                        residual,
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    hidden_states = self.norm(hidden_states)
                elif residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class KimiK3ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Hugging Face exposes Kimi-K3 as a multimodal wrapper whose language
        # configuration lives under ``text_config``.  SRT only instantiates the
        # language model here, so unwrap it before constructing any layers.
        config = getattr(config, "text_config", config)
        if quant_config is not None and hasattr(quant_config, "quant_description"):
            # ModelSlim looks up schemes with SRT's language-only prefixes
            # (``model.*`` / ``lm_head``), while official K3 descriptions keep
            # the multimodal wrapper's ``language_model.*`` prefix. Normalize
            # those keys in memory; reduced debug descriptions are unchanged.
            quant_description = quant_config.quant_description
            if any(
                isinstance(name, str) and name.startswith("language_model.")
                for name in quant_description
            ):
                quant_config.quant_description = {
                    (
                        name.removeprefix("language_model.")
                        if isinstance(name, str)
                        else name
                    ): value
                    for name, value in quant_description.items()
                }
        self.config = config
        self.quant_config = quant_config
        self.model = KimiLinearModel(
            config, quant_config, prefix=maybe_prefix(prefix, "model")
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
            pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Kimi-K3 is distributed as a multimodal checkpoint.  This class only
        # owns the text model, whose tensors are nested below ``language_model``
        # in the checkpoint but live at the top level in this module.
        language_model_prefix = "language_model."
        multimodal_prefixes = ("vision_tower.", "mm_projector.")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # Fused path
            (".fused_qkvbfg_a_proj", ".q_proj", 0),
            (".fused_qkvbfg_a_proj", ".k_proj", 1),
            (".fused_qkvbfg_a_proj", ".v_proj", 2),
            (".fused_qkvbfg_a_proj", ".b_proj", 3),
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
            (".fused_fg_b_proj", ".f_b_proj", 0),
            (".fused_fg_b_proj", ".g_b_proj", 1),
            # Unfused path: separate qkv_proj (when do_fuse_qkvbfg=False)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # qkv conv fuse
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}

            if name.startswith(language_model_prefix):
                name = name[len(language_model_prefix) :]
            elif name.startswith(multimodal_prefixes):
                # Vision weights are consumed by the multimodal wrapper.  The
                # language-only SRT implementation must not try to load them.
                continue

            if name.startswith("model.layers."):
                layer_id = int(name.split(".")[2])
                if layer_id >= self.config.num_hidden_layers:
                    continue

            if "rotary_emb.inv_freq" in name:
                continue

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # DeepSeek MLA fuses q_a and kv_a into one replicated projection,
            # while K3 checkpoints keep the two FLOAT tensors separate.
            if ".self_attn.q_a_proj." in name or ".self_attn.kv_a_proj_with_mqa." in name:
                layer_id = int(name.split(".")[2])
                if not self.config.is_kda_layer(layer_id):
                    if ".self_attn.q_a_proj." in name:
                        fused_name = name.replace(
                            ".self_attn.q_a_proj.",
                            ".self_attn.fused_qkv_a_proj_with_mqa.",
                        )
                        output_offset = 0
                    else:
                        fused_name = name.replace(
                            ".self_attn.kv_a_proj_with_mqa.",
                            ".self_attn.fused_qkv_a_proj_with_mqa.",
                        )
                        output_offset = self.config.q_lora_rank
                    param = params_dict[fused_name]
                    param.data.narrow(
                        0, output_offset, loaded_weight.shape[0]
                    ).copy_(loaded_weight)
                    loaded_params.add(fused_name)
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (".experts." in name) and name not in params_dict:
                    continue
                # Check if this mapping targets a fused projection (only apply fusion check to fused params)
                if param_name in {".fused_qkvbfg_a_proj", ".fused_fg_b_proj"}:
                    layer_id = int(name.split(".")[2])
                    if not self.config.is_kda_layer(layer_id):
                        continue
                    layer = self.model.layers[layer_id].self_attn
                    # Only load to fused projection if fusion is enabled
                    if not getattr(layer, "do_fuse_qkvbfg", False):
                        continue
                if weight_name in {".q_proj", ".k_proj", ".v_proj"}:
                    layer_id = int(name.split(".")[2])
                    if not self.config.is_kda_layer(layer_id):
                        continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # if is_pp_missing_parameter(name, self):
                #     continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (param_name, weight_name, expert_id, shard_id) in enumerate(
                    expert_params_mapping
                ):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # if is_pp_missing_parameter(name, self):
                    #     continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    # if is_pp_missing_parameter(name, self):
                    #     continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        full_attention_layer_ids = [
            layer_id
            for layer_id in range(self.config.num_hidden_layers)
            if not self.config.is_kda_layer(layer_id)
        ]
        for layer_id in full_attention_layer_ids:
            self_attn = self.model.layers[layer_id].self_attn
            w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
            if hasattr(self_attn.kv_b_proj, "weight_scale"):
                self_attn.w_scale = self_attn.kv_b_proj.weight_scale


EntryClass = KimiK3ForConditionalGeneration
