# SPDX-License-Identifier: Apache-2.0
"""Expert weight store for MoE DRAM offloading.

Manages MoE expert weights in Host DRAM. During forward, only the
Top-K selected experts are loaded from Host DRAM to HBM on demand.

Two backends are supported:
  1. acc_offload (default when available): Uses MemFabric acc_offload
     AICore AIV kernel with MTE engine for batch sparse copy.
     Higher performance due to 32-core parallelism and reduced API overhead.
  2. PyTorch H2D (fallback): Uses tensor.to("npu", non_blocking=True).
     No external dependency, works everywhere.

Weight loading paths:
  - Prefill (with prefetch): prefetch_layer_to_buffer() async-loads ALL
    experts for the first N layers on h2d_stream. wait_prefill_prefetch()
    synchronizes via per-layer NPU event before compute.
  - Prefill (no prefetch, N=0): _load_experts_on_demand() loads Top-K
    experts into shared HBM buffers synchronously per layer.
  - Decode: build_active_weight_tensors() builds compact [num_active, ...]
    tensors by loading only active experts from DRAM -- no HBM caching,
    every decode step reads from Host DRAM in real time.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

logger = logging.getLogger(__name__)


def _get_hbm_usage_gb() -> Tuple[float, float]:
    """Get current HBM allocated/reserved memory in GB.

    Returns:
        (allocated_gb, reserved_gb)
        - allocated: memory currently held by tensors
        - reserved: total memory reserved by the caching allocator
                    (closer to what system tools report)
    """
    if not torch.npu.is_available():
        return 0.0, 0.0
    allocated = torch.npu.memory_allocated() / 1024**3
    reserved = torch.npu.memory_reserved() / 1024**3
    return allocated, reserved


class ExpertWeightStore:
    """Manages MoE expert weights across Host DRAM and HBM.

    Weights are stored in Host DRAM after process_weights_after_loading().
    During forward, only Top-K selected experts are loaded from DRAM to HBM.

    Attributes:
        dram_store: {(layer_id, expert_id): {weight_name: cpu_tensor}}
        h2d_stream: Dedicated NPU stream for H2D transfers
        use_acc_offload: Whether to use acc_offload sparse_copy
    """

    def __init__(
        self,
        dram_pool_size_gb: float = 1300.0,
        use_acc_offload: bool = True,
        use_pool_for_storage: bool = True,
        shared_buffer_max_gb: float = 0,
    ):
        self.dram_store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        # Decode mode flag: True during decode, False during prefill.
        # Used to select the weight loading path:
        #   - Prefill: _load_experts_on_demand / prefetch_layer_to_buffer
        #   - Decode: build_active_weight_tensors (compact, real-time)
        self._is_decode_mode = False

        # Dedicated stream for H2D transfers (separate from compute stream)
        self._h2d_stream = None
        self._initialized = False

        # acc_offload backend
        self.use_acc_offload = use_acc_offload
        self._offload = None
        self._offload_initialized = False
        self._dram_pool_size_bytes = int(dram_pool_size_gb * 1024**3)

        # Storage mode: when False (staging mode), weights are stored in
        # pinned memory instead of the acc_offload pool. The pool is
        # initialized with a small size (1 GB) only to enable the
        # sparse_copy API for H2D transfers.
        self._use_pool_for_storage = use_pool_for_storage
        if not use_pool_for_storage:
            self._dram_pool_size_bytes = 1 * 1024**3  # 1 GB staging

        # Track registered layers for warmup
        self._registered_layers: set = set()

        # Shared HBM weight buffer: all MoE layers reuse the same buffer.
        # Before each layer's forward, Top-K experts are loaded into it.
        # This avoids allocating separate HBM tensors for all 80 layers.
        # Key: weight_name, Value: HBM tensor of shape [num_experts, ...]
        self._shared_hbm_buffers: Dict[str, torch.Tensor] = {}
        self._shared_buffer_shapes: Dict[str, tuple] = {}
        # Budget (in bytes) for shared HBM buffers. 0 disables shared buffers
        # entirely -- all layers use per-forward allocation.
        self._shared_buffer_max_bytes = int(shared_buffer_max_gb * 1024**3)

        # Statistics
        self._stats = {"dram_load": 0, "total_requests": 0}

        # Hybrid storage: layers in _h2d_layer_ids use PyTorch H2D
        # (torch.empty) instead of acc_offload pool. Configured via
        # --moe-dram-acc-offload-layers: only the first N offloaded
        # layers use the pool; the rest use H2D.
        self._h2d_layer_ids: set = set()  # layer_ids that use PyTorch H2D

    def set_acc_offload_layers(
        self, acc_offload_layers: int, all_offloaded_layer_ids: list
    ):
        """Configure which layers use acc_offload pool vs PyTorch H2D.

        Args:
            acc_offload_layers: Number of offloaded layers (starting from
                the first) to use acc_offload pool. 0 = all use pool.
            all_offloaded_layer_ids: Sorted list of all layer_ids that
                will be offloaded (non-skip). Layers after the first
                `acc_offload_layers` will use PyTorch H2D.
        """
        if acc_offload_layers > 0 and all_offloaded_layer_ids:
            sorted_ids = sorted(all_offloaded_layer_ids)
            # First N layers use pool, rest use H2D
            self._h2d_layer_ids = set(sorted_ids[acc_offload_layers:])
            pool_ids = sorted_ids[:acc_offload_layers]
            logger.info(
                f"[ExpertWeightStore] acc_offload layers: "
                f"{len(pool_ids)} ({pool_ids[0]}..{pool_ids[-1]}), "
                f"H2D layers: {len(self._h2d_layer_ids)} "
                f"({sorted(self._h2d_layer_ids)})"
            )

    def _ensure_initialize(self):
        if not self._initialized:
            if torch.npu.is_available():
                self._h2d_stream = torch.npu.Stream()

                # Try to initialize acc_offload
                if self.use_acc_offload:
                    self._init_acc_offload()

            self._initialized = True

    def _init_acc_offload(self):
        """Initialize MemFabric acc_offload DRAM pool.

        When multiple ranks initialize simultaneously, they compete for
        huge pages allocation (HalMemCreate). Huge pages require
        contiguous physical memory, so even if free DRAM is sufficient,
        concurrent 140GB allocations can fail due to fragmentation /
        kernel lock contention.

        Serializing initialization via a barrier ensures each rank's
        HalMemCreate completes before the next rank starts, avoiding
        concurrent huge page allocation failures.
        """
        try:
            from memfabric_hybrid import offload
            import torch.distributed as dist

            # Serialize acc_offload initialization across ranks to avoid
            # concurrent huge pages allocation failures. Each rank waits
            # for the previous rank to finish before starting its own
            # HalMemCreate call.
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                for i in range(world_size):
                    if i == rank:
                        self._do_acc_offload_init(offload)
                    dist.barrier()
            else:
                self._do_acc_offload_init(offload)

        except ImportError:
            logger.warning(
                "[ExpertWeightStore] memfabric_hybrid not available, "
                "falling back to PyTorch H2D"
            )
            self.use_acc_offload = False

    def _do_acc_offload_init(self, offload):
        """Actual acc_offload initialization (called by _init_acc_offload)."""
        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        config.size = self._dram_pool_size_bytes
        ret = offload.initialize(config)
        if ret == 0:
            self._offload = offload
            self._offload_initialized = True
            logger.info(
                f"[ExpertWeightStore] acc_offload initialized: "
                f"device={config.device_id}, "
                f"dram_pool={self._dram_pool_size_bytes / 1024**3:.1f} GB"
            )
        else:
            logger.warning(
                f"[ExpertWeightStore] acc_offload init failed (ret={ret}), "
                f"falling back to PyTorch H2D"
            )
            self.use_acc_offload = False

    def _check_shared_buffer_budget(self, name: str, nbytes: int) -> bool:
        """Check if allocating a shared buffer of nbytes would fit the budget.

        0 = shared buffer disabled; all layers use per-forward allocation
        (lowest HBM footprint, higher allocation overhead per forward).
        """
        if self._shared_buffer_max_bytes <= 0:
            return False
        current = sum(t.nbytes for t in self._shared_hbm_buffers.values())
        if current + nbytes > self._shared_buffer_max_bytes:
            logger.warning(
                f"[ExpertWeightStore] Shared buffer '{name}' "
                f"({nbytes / 1024**2:.1f} MB) would exceed budget "
                f"({self._shared_buffer_max_bytes / 1024**3:.1f} GB, "
                f"current={current / 1024**2:.1f} MB). "
                f"Skipping shared buffer; will use per-forward allocation."
            )
            return False
        return True

    def get_shared_hbm_buffer(
        self, name: str, shape: tuple, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Get or create a shared HBM buffer for a weight name.

        All MoE layers share the same buffer (same shape/dtype).
        Before each layer's forward, Top-K experts are loaded into it.
        This avoids allocating 80 separate HBM tensors (~160G total).
        Instead, only one buffer (~2G) is allocated and reused.

        Returns None if the buffer would exceed the HBM budget.
        """
        if name in self._shared_hbm_buffers:
            return self._shared_hbm_buffers[name]

        estimated_nbytes = (
            int(torch.tensor(list(shape)).prod().item()) * dtype.itemsize
            if shape else 0
        )
        if not self._check_shared_buffer_budget(name, estimated_nbytes):
            return None

        target_device = "npu" if torch.npu.is_available() else "cpu"
        self._shared_hbm_buffers[name] = torch.empty(
            shape, dtype=dtype, device=target_device
        )
        self._shared_buffer_shapes[name] = shape
        alloc_now, reserved_now = _get_hbm_usage_gb()
        logger.info(
            f"[ExpertWeightStore] Allocated shared HBM buffer '{name}': "
            f"shape={shape}, dtype={dtype}, "
            f"size={self._shared_hbm_buffers[name].nbytes / 1024**2:.1f} MB. "
            f"HBM now: alloc={alloc_now:.2f} GB, reserved={reserved_now:.2f} GB"
        )
        return self._shared_hbm_buffers[name]

    def register_expert(
        self,
        layer_id: int,
        expert_id: int,
        weights: Dict[str, torch.Tensor],
    ):
        """Register expert weights from HBM to Host DRAM.

        Called after process_weights_after_loading(). Copies the processed
        (NZ-format, packed) weights from HBM to Host DRAM.

        Args:
            layer_id: Layer index
            expert_id: Expert index within the layer
            weights: Dict of {weight_name: hbm_tensor} e.g.
                     {"w13_weight": ..., "w2_weight": ...,
                      "w13_weight_scale": ..., "w2_weight_scale": ...}
        """
        self._ensure_initialize()
        key = (layer_id, expert_id)

        cpu_weights = {}
        total_bytes = 0
        for name, tensor in weights.items():
            # NPU internal format (e.g., FRACTAL_NZ) cannot be copied via
            # copy_() or .cpu() -- NPU raises "do not support internal format".
            # npu_format_cast to ND may only change metadata without
            # reformatting storage, so .contiguous() forces a real ND copy.
            if tensor.device.type != "cpu":
                # FRACTAL_NZ format cannot be copied via .copy_() or .cpu().
                # Cast to ND first, then .contiguous() forces a real format
                # conversion (not just metadata change). If this fails, raise
                # immediately -- a silent fallback to .contiguous() alone does
                # NOT guarantee NZ->ND and would cause "do not support internal
                # format" errors later in copy_().
                tensor = torch_npu.npu_format_cast(
                    tensor, NPUACLFormat.ACL_FORMAT_ND
                ).contiguous()
                tensor = tensor.cpu()

            use_pool = (
                self._use_pool_for_storage
                and self.use_acc_offload
                and self._offload_initialized
                and layer_id not in self._h2d_layer_ids
            )
            if use_pool:
                # Allocate from acc_offload DRAM pool.
                dram_tensor = self._offload.empty(
                    tensor.shape, dtype=tensor.dtype
                )
            else:
                # H2D tail layer or pool unavailable: PyTorch torch.empty.
                dram_tensor = torch.empty(
                    tensor.shape, dtype=tensor.dtype, pin_memory=False
                )
            dram_tensor.copy_(tensor)
            cpu_weights[name] = dram_tensor
            total_bytes += dram_tensor.nbytes

        self.dram_store[key] = cpu_weights
        self._registered_layers.add(layer_id)

        if expert_id % 64 == 0:
            logger.info(
                f"[ExpertWeightStore] D2H layer_id={layer_id} expert_id={expert_id}: "
                f"{len(cpu_weights)} tensors, {total_bytes / 1024**2:.1f} MB copied to DRAM"
            )

    def _batch_h2d_copy(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        sync: bool = True,
        layer_id: Optional[int] = None,
    ) -> None:
        """Batch H2D copy via acc_offload sparse_copy with PyTorch fallback.

        Centralizes all H2D transfers so that sparse_copy constraints are
        enforced in one place:
          - size tensor MUST be 0-D scalar (matches reference usage)
          - num_pairs MUST be even: if odd, split the last pair into two
            halves (src_ptr + half, dst_ptr + half, len/2) to make it even
          - sparse_copy runs on default stream (no stream context)

        Args:
            pairs: List of (src_cpu_tensor, dst_hbm_tensor) pairs.
                   src/dst must have the same nbytes.
            sync: Whether to synchronize after copy. Set False for async
                  prefetch (caller records an event instead).
            layer_id: If in _h2d_layer_ids, skip sparse_copy and use
                      copy_() directly (H2D tail layers stored via
                      torch.empty, not in acc_offload pool).
        """
        num_pairs = len(pairs)
        if num_pairs == 0:
            return

        # H2D tail layers: src tensors are torch.empty (not in pool).
        # Skip sparse_copy entirely — it would fail and waste time.
        use_sparse = (
            self.use_acc_offload
            and self._offload_initialized
            and (layer_id is None or layer_id not in self._h2d_layer_ids)
        )

        if use_sparse:
            # Build (src_ptr, dst_ptr, nbytes) triples from pairs.
            # If num_pairs is odd, split the last pair into two halves
            # to make it even (sparse_copy requires even count).
            src_ptrs = []
            dst_ptrs = []
            len_ptrs = []

            for src, dst in pairs:
                src_ptrs.append(src.data_ptr())
                dst_ptrs.append(dst.data_ptr())
                len_ptrs.append(src.nbytes)

            if num_pairs % 2 != 0:
                # Split last pair into two halves
                last_src = src_ptrs[-1]
                last_dst = dst_ptrs[-1]
                last_len = len_ptrs[-1]
                half = last_len // 2
                # Replace last pair with two half-size pairs
                src_ptrs[-1] = last_src
                dst_ptrs[-1] = last_dst
                len_ptrs[-1] = half
                src_ptrs.append(last_src + half)
                dst_ptrs.append(last_dst + half)
                len_ptrs.append(half)
                num_pairs += 1

            src_ptr_t = torch.tensor(src_ptrs, dtype=torch.int64, device="npu")
            dst_ptr_t = torch.tensor(dst_ptrs, dtype=torch.int64, device="npu")
            len_t = torch.tensor(len_ptrs, dtype=torch.int32, device="npu")
            # 0-D scalar, matches reference usage in local_dram_offload.py
            size_t = torch.tensor(num_pairs, dtype=torch.int32, device="npu")

            device = torch.device(f"npu:{torch.npu.current_device()}")
            # sparse_copy on default stream (no stream context), matching
            # the reference usage. Running on h2d_stream causes the kernel
            # to execute on a different stream than its args tensors.
            ret = self._offload.sparse_copy(
                src_ptr_t, dst_ptr_t, len_t, size_t, device
            )

            if ret == 0:
                if sync:
                    torch.npu.synchronize()
                return

            logger.warning(
                f"[ExpertWeightStore] sparse_copy ret={ret}, "
                f"using copy_() fallback"
            )

        # Fallback: PyTorch H2D copy_ (runs on h2d_stream).
        with torch.npu.stream(self._h2d_stream):
            for src, dst in pairs:
                dst.copy_(src, non_blocking=True)
        if sync:
            self._h2d_stream.synchronize()

    def batch_load_to_shared_buffer(
        self,
        layer_id: int,
        expert_ids: List[int],
        shared_buffers: Dict[str, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load expert weights directly into shared HBM buffers.

        This avoids the extra HBM->HBM copy that a separate per-expert
        allocation would do. Weights are written directly into
        shared_buffers[expert_id].

        Args:
            layer_id: Layer index
            expert_ids: List of expert IDs to load
            shared_buffers: {weight_name: HBM tensor of shape [num_experts, ...]}

        Returns:
            {expert_id: {weight_name: view into shared_buffer}} for stats
        """
        self._ensure_initialize()
        results = {}
        missing = []

        for eid in expert_ids:
            key = (layer_id, eid)
            self._stats["total_requests"] += 1

            if key not in self.dram_store:
                continue

            self._stats["dram_load"] += 1
            missing.append(key)

        if not missing:
            return results

        # Build (src_cpu, dst_hbm) pairs pointing directly into the shared
        # buffers. _batch_h2d_copy handles sparse_copy + fallback + sync.
        pairs = []
        for key in missing:
            eid = key[1]
            dram_weights = self.dram_store[key]
            expert_views = {}

            for name, dram_tensor in dram_weights.items():
                if name not in shared_buffers:
                    continue
                dst_tensor = shared_buffers[name][eid]
                expert_views[name] = dst_tensor
                pairs.append((dram_tensor, dst_tensor))

            results[eid] = expert_views

        self._batch_h2d_copy(pairs, sync=True, layer_id=layer_id)

        return results

    def build_active_weight_tensors(
        self,
        layer_id: int,
        active_expert_ids: List[int],
        weight_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Build compact [num_active, ...] weight tensors for active experts.

        Loads only the active (token-bearing) experts from Host DRAM to HBM
        in real time. No HBM caching -- every decode step reads from DRAM.

        Args:
            layer_id: Layer index
            active_expert_ids: Sorted list of expert IDs with tokens
            weight_names: List of weight parameter names

        Returns:
            {weight_name: tensor of shape [num_active, ...]}
        """
        self._ensure_initialize()

        num_active = len(active_expert_ids)
        sample_key = (layer_id, active_expert_ids[0])

        # Allocate compact [num_active, ...] HBM buffers
        result = {}
        for name in weight_names:
            sample_tensor = self.dram_store[sample_key][name]
            full_shape = (num_active,) + sample_tensor.shape
            result[name] = torch.empty(
                full_shape, dtype=sample_tensor.dtype, device="npu"
            )

        # Build (src_cpu, dst_hbm) pairs for batch sparse_copy.
        pairs = []
        for i, eid in enumerate(active_expert_ids):
            self._stats["total_requests"] += 1
            self._stats["dram_load"] += 1
            dram_weights = self.dram_store[(layer_id, eid)]
            for name in weight_names:
                pairs.append((dram_weights[name], result[name][i]))

        self._batch_h2d_copy(pairs, sync=True, layer_id=layer_id)

        return result

    # ------------------------------------------------------------------
    # Prefill full-layer prefetch + cache mode management
    # ------------------------------------------------------------------ #

    def set_cache_mode(self, is_prefill: bool):
        """Toggle between prefill and decode mode.

        Sets _is_decode_mode which controls the weight loading path:
          - Prefill (is_prefill=True): _load_experts_on_demand /
            prefetch_layer_to_buffer loads into [num_local_experts, ...] buffers
          - Decode (is_prefill=False): build_active_weight_tensors builds
            compact [num_active, ...] tensors directly from DRAM
        """
        self._is_decode_mode = not is_prefill

    def sync_prefetch(self):
        """Block until all pending h2d_stream operations complete."""
        self._ensure_initialize()
        if self._h2d_stream is not None:
            self._h2d_stream.synchronize()

    def prefetch_layer_to_buffer(
        self, layer_id: int, num_experts: int
    ) -> Tuple[Dict[str, torch.Tensor], Optional["torch.npu.Event"]]:
        """Prefetch ALL experts for a layer into per-layer HBM buffers.

        Allocates [num_experts, ...] tensors and loads from DRAM on
        h2d_stream (async, no sync). Caller must wait on the returned
        event (or call sync_prefetch() if the event is None) before using
        the buffers, and free_layer_buffers() after compute to release HBM.

        Returns:
            (buffers, event): buffers is {weight_name: hbm_tensor of shape
            [num_experts, ...]}; event is an NPU event recorded on the
            h2d_stream after all H2D copies for this layer, or None when
            h2d_stream is unavailable (CPU-only). Use event.wait() on the
            compute stream to synchronize ONLY this layer's prefetch --
            avoids global h2d_stream.synchronize() which would stall all
            in-flight prefetches and destroy H2D/compute overlap.
        """
        self._ensure_initialize()

        sample_key = (layer_id, 0)
        weight_names = list(self.dram_store[sample_key].keys())

        buffers = {}
        for name in weight_names:
            sample_tensor = self.dram_store[sample_key][name]
            full_shape = (num_experts,) + sample_tensor.shape
            buffers[name] = torch.empty(
                full_shape, dtype=sample_tensor.dtype, device="npu"
            )

        expert_ids = list(range(num_experts))

        # Build (src_cpu, dst_hbm) pairs for batch sparse_copy.
        pairs = []
        for eid in expert_ids:
            key = (layer_id, eid)
            dram_weights = self.dram_store[key]
            for name, dram_tensor in dram_weights.items():
                if name in buffers:
                    pairs.append((dram_tensor, buffers[name][eid]))

        event = None
        if self._h2d_stream is not None:
            event = torch.npu.Event()
            # Async batch H2D: sparse_copy runs on h2d_stream without sync.
            # Caller waits on event (recorded below) before using buffers.
            self._batch_h2d_copy(pairs, sync=False, layer_id=layer_id)
            with torch.npu.stream(self._h2d_stream):
                event.record()
        else:
            # No h2d_stream (CPU-only): synchronous copy, no event needed.
            for src, dst in pairs:
                dst.copy_(src)

        return buffers, event

    def free_layer_buffers(self, buffers: Dict[str, torch.Tensor]):
        """Free per-layer HBM buffers allocated by prefetch_layer_to_buffer.

        Only clears Python references; the caching allocator reclaims and
        reuses the memory automatically. No gc.collect()/empty_cache() here
        -- empty_cache() triggers a device-wide sync on NPU, which waits for
        pending h2d_stream prefetch operations, destroying compute/prefetch
        overlap and causing multi-second stalls per layer.
        """
        if not buffers:
            return
        freed_mb = sum(t.nbytes for t in buffers.values()) / 1024**2
        buffers.clear()

    def uninitialize(self):
        """Cleanup acc_offload resources."""
        if self._offload_initialized:
            try:
                self._offload.uninitialize()
            except Exception:
                pass
            self._offload_initialized = False

    def get_stats(self) -> dict:
        total = max(self._stats["total_requests"], 1)
        return {
            "dram_load_count": self._stats["dram_load"],
            "total_requests": self._stats["total_requests"],
            "dram_total_experts": len(self.dram_store),
            "backend": "acc_offload" if self.use_acc_offload else "pytorch_h2d",
        }

    def get_dram_usage_gb(self) -> float:
        """Get total DRAM usage in GB."""
        total = 0
        for weights in self.dram_store.values():
            total += sum(t.nbytes for t in weights.values())
        return total / 1024**3

    def release_hbm_weights(self):
        """Release all HBM shared buffers.

        Called after offload registration to free HBM used during the
        registration process. Shared buffers should be empty at this point
        (not yet used), so this is mostly gc + empty_cache.
        """
        shared_buffer_count = len(self._shared_hbm_buffers)
        shared_buffer_bytes = sum(
            t.nbytes for t in self._shared_hbm_buffers.values()
        )

        self._shared_hbm_buffers.clear()

        logger.debug(
            f"[ExpertWeightStore] release_hbm_weights: "
            f"cleared {shared_buffer_count} shared buffers "
            f"({shared_buffer_bytes / 1024**2:.1f} MB)"
        )

        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
