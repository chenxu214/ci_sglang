# SPDX-License-Identifier: Apache-2.0
"""Expert weight store for MoE DRAM offloading.

Manages MoE expert weights in Host DRAM with an LRU cache in HBM.
During forward, only the Top-K selected experts are loaded from
Host DRAM to HBM on demand.

Two backends are supported:
  1. acc_offload (default when available): Uses MemFabric acc_offload
     AICore AIV kernel with MTE engine for batch sparse copy.
     Higher performance due to 32-core parallelism and reduced API overhead.
  2. PyTorch H2D (fallback): Uses tensor.to("npu", non_blocking=True).
     No external dependency, works everywhere.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.utils.common import get_int_env_var
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
    An LRU cache in HBM holds recently-used experts. During forward,
    only Top-K selected experts are loaded from DRAM to HBM.

    LRU implementation:
        self._per_layer_caches is a dict of OrderedDicts, one per layer_id.
        Each layer's cache is an OrderedDict[expert_id, weights]. On cache
        hit, move_to_end() moves the entry to the tail (most recently used).
        On eviction, popitem(last=False) removes the head (least recently
        used). The slot-count limit (20 for decode) is applied per-layer
        so that cross-layer eviction does not occur during decode.

    Attributes:
        dram_store: {(layer_id, expert_id): {weight_name: cpu_tensor}}
        _per_layer_caches: {layer_id: OrderedDict[expert_id, {name: hbm_tensor}]}
        h2d_stream: Dedicated NPU stream for H2D transfers
        use_acc_offload: Whether to use acc_offload sparse_copy
    """

    def __init__(
        self,
        dram_pool_size_gb: float = 1300.0,
        use_acc_offload: bool = True,
    ):
        self.dram_store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        # Per-layer LRU caches: {layer_id: OrderedDict[expert_id, weights]}
        # Each layer has its own slot-count limit (20 for decode, unlimited
        # for prefill). This prevents cross-layer eviction during decode.
        self._per_layer_caches: Dict[int, OrderedDict] = {}
        self.hbm_cache_used_bytes = 0

        # Slot-count limit for decode LRU (env-tunable). When > 0, evict by
        # entry count in addition to byte-size. Set to 0 (unlimited) during
        # prefill via set_cache_mode().
        self._decode_cache_slots = get_int_env_var(
            "SGLANG_KIMI_DECODE_CACHE_SLOTS", 20
        )
        self.hbm_cache_max_slots = self._decode_cache_slots

        # Dedicated stream for H2D transfers (separate from compute stream)
        self._h2d_stream = None
        self._initialized = False

        # acc_offload backend
        self.use_acc_offload = use_acc_offload
        self._offload = None
        self._offload_initialized = False
        self._dram_pool_size_bytes = int(dram_pool_size_gb * 1024**3)

        # Track registered layers for warmup
        self._registered_layers: set = set()

        # Shared HBM weight buffer: all MoE layers reuse the same buffer.
        # Before each layer's forward, Top-K experts are loaded into it.
        # This avoids allocating separate HBM tensors for all 80 layers.
        # Key: weight_name, Value: HBM tensor of shape [num_experts, ...]
        self._shared_hbm_buffers: Dict[str, torch.Tensor] = {}
        self._shared_buffer_shapes: Dict[str, tuple] = {}

        # Statistics
        self._stats = {"hbm_hit": 0, "dram_load": 0, "total_requests": 0}

    def _ensure_initialized(self):
        if not self._initialized:
            if torch.npu.is_available():
                self._h2d_stream = torch.npu.Stream()

                # Try to initialize acc_offload
                if self.use_acc_offload:
                    self._init_acc_offload()

            self._initialized = True

    def _get_layer_cache(self, layer_id: int) -> OrderedDict:
        """Get (or create) the per-layer LRU OrderedDict."""
        if layer_id not in self._per_layer_caches:
            self._per_layer_caches[layer_id] = OrderedDict()
        return self._per_layer_caches[layer_id]

    def _init_acc_offload(self):
        """Initialize MemFabric acc_offload DRAM pool."""
        try:
            from memfabric_hybrid import offload

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
        except ImportError:
            logger.warning(
                "[ExpertWeightStore] memfabric_hybrid not available, "
                "falling back to PyTorch H2D"
            )
            self.use_acc_offload = False

    def get_shared_hbm_buffer(
        self, name: str, shape: tuple, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or create a shared HBM buffer for a weight name.

        All MoE layers share the same buffer (same shape/dtype).
        Before each layer's forward, Top-K experts are loaded into it.
        This avoids allocating 80 separate HBM tensors (~160G total).
        Instead, only one buffer (~2G) is allocated and reused.
        """
        if name not in self._shared_hbm_buffers:
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
        self._ensure_initialized()
        key = (layer_id, expert_id)

        cpu_weights = {}
        total_bytes = 0
        for name, tensor in weights.items():
            # NPU internal format (e.g., FRACTAL_NZ) cannot be copied via
            # copy_() or .cpu() — NPU raises "do not support internal format".
            # npu_format_cast to ND may only change metadata without
            # reformatting storage, so .contiguous() forces a real ND copy.
            if tensor.device.type != "cpu":
                # FRACTAL_NZ format cannot be copied via .copy_() or .cpu().
                # Cast to ND first, then .contiguous() forces a real format
                # conversion (not just metadata change). If this fails, raise
                # immediately — a silent fallback to .contiguous() alone does
                # NOT guarantee NZ→ND and would cause "do not support internal
                # format" errors later in copy_().
                tensor = torch_npu.npu_format_cast(
                    tensor, NPUACLFormat.ACL_FORMAT_ND
                ).contiguous()
                tensor = tensor.cpu()

            if self.use_acc_offload and self._offload_initialized:
                # Allocate from acc_offload DRAM pool
                dram_tensor = self._offload.empty(
                    tensor.shape, dtype=tensor.dtype
                )
            else:
                # Fallback: PyTorch pinned memory
                dram_tensor = torch.empty(
                    tensor.shape, dtype=tensor.dtype, pin_memory=True
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

    def get_expert_weights(
        self, layer_id: int, expert_id: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get expert weights from HBM cache, loading from DRAM if needed.

        Returns None if expert is not registered.
        """
        self._ensure_initialized()
        key = (layer_id, expert_id)
        self._stats["total_requests"] += 1

        if key not in self.dram_store:
            return None

        # Check per-layer HBM LRU cache
        lc = self._get_layer_cache(layer_id)
        if expert_id in lc:
            lc.move_to_end(expert_id)  # LRU: mark as recently used
            self._stats["hbm_hit"] += 1
            return lc[expert_id]

        # Load from DRAM to HBM
        self._stats["dram_load"] += 1
        return self._load_to_hbm(key)

    def batch_get_expert_weights(
        self, layer_id: int, expert_ids: List[int]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load expert weights to HBM.

        When acc_offload is enabled, uses sparse_copy (AICore AIV kernel
        with MTE engine, 32-core parallel) for batch transfer.
        Otherwise, uses PyTorch H2D on a dedicated stream.
        """
        self._ensure_initialized()
        results = {}
        missing = []
        hit_ids = []

        lc = self._get_layer_cache(layer_id)
        for eid in expert_ids:
            key = (layer_id, eid)
            self._stats["total_requests"] += 1

            if key not in self.dram_store:
                continue

            if eid in lc:
                lc.move_to_end(eid)  # LRU: mark as recently used
                self._stats["hbm_hit"] += 1
                hit_ids.append(eid)
                results[eid] = lc[eid]
            else:
                self._stats["dram_load"] += 1
                missing.append(key)

        if missing:
            miss_ids = [k[1] for k in missing]
            logger.info(
                f"[ExpertWeightStore batch_get] layer_id={layer_id}: "
                f"requested={expert_ids}, hbm_hit={hit_ids} ({len(hit_ids)}), "
                f"dram_load={miss_ids} ({len(miss_ids)}), "
                f"backend={'acc_offload' if (self.use_acc_offload and self._offload_initialized) else 'pytorch_h2d'}"
            )

        if not missing:
            return results

        if self.use_acc_offload and self._offload_initialized:
            # Use acc_offload sparse_copy (AICore MTE, 32-core parallel)
            results.update(self._sparse_copy_batch(layer_id, missing))
        else:
            # Fallback: PyTorch H2D on dedicated stream
            results.update(self._pytorch_h2d_batch(missing))

        return results

    def batch_load_to_shared_buffer(
        self,
        layer_id: int,
        expert_ids: List[int],
        shared_buffers: Dict[str, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load expert weights directly into shared HBM buffers.

        This avoids the extra HBM→HBM copy that batch_get_expert_weights +
        _load_experts_on_demand would do. Weights are written directly
        into shared_buffers[expert_id].

        Args:
            layer_id: Layer index
            expert_ids: List of expert IDs to load
            shared_buffers: {weight_name: HBM tensor of shape [num_experts, ...]}

        Returns:
            {expert_id: {weight_name: view into shared_buffer}} for cache stats
        """
        self._ensure_initialized()
        results = {}
        missing = []

        for eid in expert_ids:
            key = (layer_id, eid)
            self._stats["total_requests"] += 1

            if key not in self.dram_store:
                continue

            # Always load from DRAM for shared buffer path (no LRU check)
            self._stats["dram_load"] += 1
            missing.append(key)

        if not missing:
            return results

        # Build (src_ptr, dst_ptr, len) triples pointing directly into
        # the shared buffers. This avoids allocating per-expert HBM tensors
        # and the subsequent copy_ into shared buffers.
        src_ptrs = []
        dst_ptrs = []
        len_ptrs = []

        for key in missing:
            eid = key[1]
            dram_weights = self.dram_store[key]
            expert_views = {}

            for name, dram_tensor in dram_weights.items():
                if name not in shared_buffers:
                    continue
                # Destination: expert's slot in the shared buffer
                dst_tensor = shared_buffers[name][eid]
                expert_views[name] = dst_tensor

                src_ptrs.append(dram_tensor.data_ptr())
                dst_ptrs.append(dst_tensor.data_ptr())
                len_ptrs.append(dram_tensor.nbytes)

            results[eid] = expert_views

        num_pairs = len(src_ptrs)
        if num_pairs == 0:
            return results

        if self.use_acc_offload and self._offload_initialized:
            src_tensor = torch.tensor(src_ptrs, dtype=torch.int64, device="npu")
            dst_tensor = torch.tensor(dst_ptrs, dtype=torch.int64, device="npu")
            len_tensor = torch.tensor(len_ptrs, dtype=torch.int32, device="npu")
            size_tensor = torch.tensor(num_pairs, dtype=torch.int32, device="npu")

            device = torch.device(f"npu:{torch.npu.current_device()}")
            with torch.npu.stream(self._h2d_stream):
                ret = self._offload.sparse_copy(
                    src_tensor, dst_tensor, len_tensor, size_tensor, device
                )
            self._h2d_stream.synchronize()

            if ret != 0:
                logger.error(
                    f"[ExpertWeightStore] sparse_copy failed (ret={ret}), "
                    f"falling back to PyTorch H2D"
                )
                # Fallback: PyTorch H2D into shared buffers
                with torch.npu.stream(self._h2d_stream):
                    for key in missing:
                        eid = key[1]
                        dram_weights = self.dram_store[key]
                        for name, dram_tensor in dram_weights.items():
                            if name in shared_buffers:
                                shared_buffers[name][eid].copy_(
                                    dram_tensor, non_blocking=True
                                )
                self._h2d_stream.synchronize()
        else:
            # PyTorch H2D directly into shared buffers
            with torch.npu.stream(self._h2d_stream):
                for key in missing:
                    eid = key[1]
                    dram_weights = self.dram_store[key]
                    for name, dram_tensor in dram_weights.items():
                        if name in shared_buffers:
                            shared_buffers[name][eid].copy_(
                                dram_tensor, non_blocking=True
                            )
            self._h2d_stream.synchronize()

        return results

    def build_active_weight_tensors(
        self,
        layer_id: int,
        active_expert_ids: List[int],
        weight_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Build compact [num_active, ...] weight tensors for active experts.

        Called after dispatch (decode only) when we know which experts
        received tokens. Checks LRU cache first (HBM→HBM copy on hit),
        loads from DRAM on miss. Replaces the shared [224, ...] buffer
        with a smaller [num_active, ...] tensor.

        Args:
            layer_id: Layer index
            active_expert_ids: Sorted list of expert IDs with tokens
            weight_names: List of weight parameter names

        Returns:
            {weight_name: tensor of shape [num_active, ...]}
        """
        self._ensure_initialized()

        num_active = len(active_expert_ids)
        lc = self._get_layer_cache(layer_id)

        # Snapshot cache hits BEFORE loading misses.
        # _put_hbm_cache (called inside _sparse_copy_batch / _pytorch_h2d_batch)
        # may evict entries when LRU is full, including hits snapshotted here.
        # The snapshot holds references so evicted tensors survive until
        # torch.stack copies the data into the compact tensor.
        cache_hits = {}
        missing_keys = []
        for eid in active_expert_ids:
            self._stats["total_requests"] += 1
            if eid in lc:
                self._stats["hbm_hit"] += 1
                cache_hits[eid] = lc[eid]
            else:
                self._stats["dram_load"] += 1
                missing_keys.append((layer_id, eid))

        # Load misses from DRAM (return value holds references regardless
        # of LRU eviction)
        loaded = {}
        if missing_keys:
            if self.use_acc_offload and self._offload_initialized:
                loaded = self._sparse_copy_batch(layer_id, missing_keys)
            else:
                loaded = self._pytorch_h2d_batch(missing_keys)

        # Build compact tensors from snapshots (not LRU cache, which may
        # have evicted entries during loading)
        result = {}
        for name in weight_names:
            tensors = []
            for eid in active_expert_ids:
                if eid in cache_hits:
                    tensors.append(cache_hits[eid][name])
                else:
                    tensors.append(loaded[eid][name])
            result[name] = torch.stack(tensors, dim=0)

        return result

    # ------------------------------------------------------------------
    # acc_offload sparse_copy backend
    # ------------------------------------------------------------------

    def _sparse_copy_batch(
        self, layer_id: int, missing_keys: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load experts using acc_offload sparse_copy.

        Uses the AICore AIV kernel (OffloadSparseCopyOps) which runs
        on 32 AIV cores with MTE engine for parallel GM→GM copy.
        Data path: DRAM(via GVA) → MTE → UB → MTE → HBM.

        Note: The kernel splits pairs into two halves (K and V).
        All pairs are still copied, just split across two loops.
        """
        src_ptrs = []
        dst_ptrs = []
        len_ptrs = []
        expert_results = {}
        total_copy_bytes = 0

        alloc_before, reserved_before = _get_hbm_usage_gb()

        # Collect all (src, dst, len) pairs for all missing experts
        for key in missing_keys:
            eid = key[1]
            dram_weights = self.dram_store[key]
            hbm_weights = {}

            for name, dram_tensor in dram_weights.items():
                # Allocate HBM destination
                hbm_tensor = torch.empty(
                    dram_tensor.shape,
                    dtype=dram_tensor.dtype,
                    device="npu",
                )
                hbm_weights[name] = hbm_tensor

                # Collect pointers
                src_ptrs.append(dram_tensor.data_ptr())
                dst_ptrs.append(hbm_tensor.data_ptr())
                len_ptrs.append(dram_tensor.nbytes)
                total_copy_bytes += dram_tensor.nbytes

            expert_results[eid] = hbm_weights
            self._put_hbm_cache(key, hbm_weights)

        if not src_ptrs:
            return expert_results

        # Build NPU tensors for sparse_copy arguments
        num_pairs = len(src_ptrs)
        src_tensor = torch.tensor(src_ptrs, dtype=torch.int64, device="npu")
        dst_tensor = torch.tensor(dst_ptrs, dtype=torch.int64, device="npu")
        len_tensor = torch.tensor(len_ptrs, dtype=torch.int32, device="npu")
        size_tensor = torch.tensor(num_pairs, dtype=torch.int32, device="npu")

        # Execute sparse_copy on h2d_stream
        device = torch.npu.current_device()
        with torch.npu.stream(self._h2d_stream):
            ret = self._offload.sparse_copy(
                src_tensor, dst_tensor, len_tensor, size_tensor, device
            )
            if ret != 0:
                logger.error(
                    f"[ExpertWeightStore] sparse_copy failed (ret={ret}), "
                    f"falling back to PyTorch H2D for this batch"
                )
                # Fallback: use PyTorch H2D for this batch
                return self._pytorch_h2d_batch(missing_keys)

        # Wait for sparse_copy to complete
        self._h2d_stream.synchronize()

        alloc_after, reserved_after = _get_hbm_usage_gb()
        logger.info(
            f"[ExpertWeightStore sparse_copy] layer_id={layer_id}: "
            f"loaded {len(missing_keys)} experts, {num_pairs} tensors, "
            f"{total_copy_bytes / 1024**2:.1f} MB DRAM→HBM. "
            f"HBM alloc {alloc_before:.2f}→{alloc_after:.2f} GB "
            f"(+{alloc_after - alloc_before:.2f} GB)"
        )

        return expert_results

    # ------------------------------------------------------------------
    # PyTorch H2D backend (fallback)
    # ------------------------------------------------------------------

    def _pytorch_h2d_batch(
        self, missing_keys: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load experts using PyTorch H2D (fallback)."""
        results = {}
        total_copy_bytes = 0
        layer_id = missing_keys[0][0] if missing_keys else -1

        alloc_before, _ = _get_hbm_usage_gb()

        with torch.npu.stream(self._h2d_stream):
            for key in missing_keys:
                eid = key[1]
                hbm_weights = self._load_to_hbm_unchecked(key)
                results[eid] = hbm_weights
                total_copy_bytes += sum(
                    t.nbytes for t in hbm_weights.values()
                )
                # Put into HBM cache so hbm_cache_used_bytes is updated
                # and subsequent lookups get cache hits.
                self._put_hbm_cache(key, hbm_weights)

        self._h2d_stream.synchronize()

        alloc_after, _ = _get_hbm_usage_gb()
        logger.info(
            f"[ExpertWeightStore pytorch_h2d] layer_id={layer_id}: "
            f"loaded {len(missing_keys)} experts, "
            f"{total_copy_bytes / 1024**2:.1f} MB DRAM→HBM. "
            f"HBM alloc {alloc_before:.2f}→{alloc_after:.2f} GB "
            f"(+{alloc_after - alloc_before:.2f} GB)"
        )
        return results

    def _load_to_hbm(
        self, key: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Load a single expert from DRAM to HBM (with cache eviction)."""
        hbm_weights = self._load_to_hbm_unchecked(key)
        self._put_hbm_cache(key, hbm_weights)
        return hbm_weights

    def _load_to_hbm_unchecked(
        self, key: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Copy expert weights from DRAM to HBM without cache management.

        Uses PyTorch H2D (.to("npu")). For acc_offload path,
        use _sparse_copy_batch instead.
        """
        dram_weights = self.dram_store[key]
        hbm_weights = {}
        for name, cpu_tensor in dram_weights.items():
            hbm_tensor = cpu_tensor.to("npu", non_blocking=True)
            hbm_weights[name] = hbm_tensor
        return hbm_weights

    def _put_hbm_cache(
        self, key: Tuple[int, int], weights: Dict[str, torch.Tensor]
    ):
        """Put expert weights into per-layer HBM LRU cache, evicting if needed.

        Eviction is per-layer: each layer's cache has its own slot-count limit
        (20 for decode, unlimited for prefill). This prevents cross-layer
        eviction during decode so that layer i's cached experts survive
        while layers i+1..i+91 are processed.

        Byte-size eviction is global (across all layers) as a safety net.
        """
        layer_id, expert_id = key
        lc = self._get_layer_cache(layer_id)
        weight_size = sum(t.nbytes for t in weights.values())

        # Handle re-insertion (e.g., _sparse_copy_batch fallback to
        # _pytorch_h2d_batch): remove old entry first to prevent
        # false slot-count eviction and hbm_cache_used_bytes inflation.
        if expert_id in lc:
            old_weights = lc.pop(expert_id)
            old_size = sum(t.nbytes for t in old_weights.values())
            self.hbm_cache_used_bytes -= old_size
            del old_weights

        # Per-layer slot-count eviction (decode LRU: 20 experts max per layer)
        if self.hbm_cache_max_slots > 0:
            while (
                len(lc) >= self.hbm_cache_max_slots
                and len(lc) > 0
            ):
                evicted_eid, evicted_weights = lc.popitem(last=False)
                evicted_size = sum(t.nbytes for t in evicted_weights.values())
                self.hbm_cache_used_bytes -= evicted_size
                del evicted_weights

        lc[expert_id] = weights
        self.hbm_cache_used_bytes += weight_size

    def async_prefetch(self, layer_id: int, expert_ids: List[int]):
        """Asynchronously prefetch experts for the next layer.

        Submits H2D copies on the h2d_stream without blocking.
        The next layer's forward will find these in HBM cache.
        """
        self._ensure_initialized()
        if self._h2d_stream is None:
            return

        # Collect missing experts that need prefetching
        missing = []
        lc = self._get_layer_cache(layer_id)
        for eid in expert_ids:
            key = (layer_id, eid)
            if key not in self.dram_store or eid in lc:
                continue
            missing.append(key)

        if not missing:
            return

        if self.use_acc_offload and self._offload_initialized:
            # Async sparse_copy (no synchronize, will complete before next use)
            with torch.npu.stream(self._h2d_stream):
                self._sparse_copy_batch(layer_id, missing)
            # Note: no synchronize() here, prefetch is async
        else:
            # PyTorch H2D async prefetch
            with torch.npu.stream(self._h2d_stream):
                for key in missing:
                    hbm_weights = self._load_to_hbm_unchecked(key)
                    self._put_hbm_cache(key, hbm_weights)

    # ------------------------------------------------------------------
    # Prefill full-layer prefetch + cache mode management
    # ------------------------------------------------------------------ #

    def set_cache_mode(self, is_prefill: bool):
        """Toggle between prefill (unlimited slots) and decode (20-slot LRU)."""
        if is_prefill:
            self.hbm_cache_max_slots = 0  # unlimited: prefill loads all experts
        else:
            self.hbm_cache_max_slots = self._decode_cache_slots

    def prefetch_full_layer(self, layer_id: int, num_experts: int):
        """Async prefetch ALL experts for a layer on the h2d_stream.

        Used during prefill: while layer L computes, layer L+N's full expert
        set is loaded into HBM cache. Call sync_prefetch() before using.
        """
        self.async_prefetch(layer_id, list(range(num_experts)))

    def sync_prefetch(self):
        """Block until all pending h2d_stream operations complete."""
        self._ensure_initialized()
        if self._h2d_stream is not None:
            self._h2d_stream.synchronize()

    def prefetch_layer_to_buffer(
        self, layer_id: int, num_experts: int
    ) -> Dict[str, torch.Tensor]:
        """Prefetch ALL experts for a layer into per-layer HBM buffers.

        Allocates [num_experts, ...] tensors and loads from DRAM on
        h2d_stream (async, no sync). Does NOT use LRU cache. Caller must
        call sync_prefetch() before using the buffers, and free_layer_buffers()
        after compute to release HBM.

        Returns:
            {weight_name: hbm_tensor of shape [num_experts, ...]}
        """
        self._ensure_initialized()

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

        with torch.npu.stream(self._h2d_stream):
            for eid in expert_ids:
                key = (layer_id, eid)
                dram_weights = self.dram_store[key]
                for name, dram_tensor in dram_weights.items():
                    if name in buffers:
                        buffers[name][eid].copy_(
                            dram_tensor, non_blocking=True
                        )

        return buffers

    def free_layer_buffers(self, buffers: Dict[str, torch.Tensor]):
        """Free per-layer HBM buffers allocated by prefetch_layer_to_buffer."""
        if not buffers:
            return
        freed_mb = sum(t.nbytes for t in buffers.values()) / 1024**2
        buffers.clear()
        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
        logger.info(
            f"[ExpertWeightStore] free_layer_buffers: freed {freed_mb:.1f} MB"
        )

    def release_layer_hbm_cache(self, layer_id: int):
        """Remove all HBM cache entries for a given layer.

        Called after prefill compute for a layer to cap HBM usage at ~(N+1)
        concurrent layers' worth of cached experts.
        """
        if layer_id not in self._per_layer_caches:
            return
        lc = self._per_layer_caches.pop(layer_id)
        freed_bytes = sum(
            sum(t.nbytes for t in w.values()) for w in lc.values()
        )
        self.hbm_cache_used_bytes -= freed_bytes
        evicted = len(lc)
        del lc
        if evicted > 0:
            logger.info(
                f"[ExpertWeightStore release_layer] layer_id={layer_id}: "
                f"released {evicted} experts, "
                f"{freed_bytes / 1024**2:.1f} MB freed"
            )

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
        cache_count = sum(len(lc) for lc in self._per_layer_caches.values())
        return {
            "hbm_hit_rate": self._stats["hbm_hit"] / total,
            "dram_load_count": self._stats["dram_load"],
            "hbm_cache_count": cache_count,
            "hbm_cache_used_gb": self.hbm_cache_used_bytes / 1024**3,
            "dram_total_experts": len(self.dram_store),
            "backend": "acc_offload" if self.use_acc_offload else "pytorch_h2d",
        }

    def get_dram_usage_gb(self) -> float:
        """Get total DRAM usage in GB."""
        total = 0
        for weights in self.dram_store.values():
            total += sum(t.nbytes for t in weights.values())
        return total / 1024**3
