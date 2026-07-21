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

logger = logging.getLogger(__name__)


class ExpertWeightStore:
    """Manages MoE expert weights across Host DRAM and HBM.

    Weights are stored in Host DRAM after process_weights_after_loading().
    An LRU cache in HBM holds recently-used experts. During forward,
    only Top-K selected experts are loaded from DRAM to HBM.

    LRU implementation:
        self.hbm_cache is an OrderedDict. On cache hit, move_to_end()
        moves the entry to the tail (most recently used). On eviction,
        popitem(last=False) removes the head (least recently used).

    Attributes:
        dram_store: {(layer_id, expert_id): {weight_name: cpu_tensor}}
        hbm_cache: LRU cache {(layer_id, expert_id): {weight_name: hbm_tensor}}
        h2d_stream: Dedicated NPU stream for H2D transfers
        use_acc_offload: Whether to use acc_offload sparse_copy
    """

    def __init__(
        self,
        hbm_cache_layers: int = 0,
        dram_pool_size_gb: float = 1300.0,
        use_acc_offload: bool = True,
    ):
        self.dram_store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.hbm_cache: OrderedDict[Tuple[int, int], Dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self.hbm_cache_size_bytes = 0  # Auto-calculated during warmup
        self.hbm_cache_used_bytes = 0
        self.hbm_cache_layers = hbm_cache_layers

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
            logger.info(
                f"[ExpertWeightStore] Allocated shared HBM buffer '{name}': "
                f"shape={shape}, dtype={dtype}, "
                f"size={self._shared_hbm_buffers[name].nbytes / 1024**2:.1f} MB"
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
        for name, tensor in weights.items():
            if self.use_acc_offload and self._offload_initialized:
                # Allocate from acc_offload DRAM pool
                dram_tensor = self._offload.empty(
                    tensor.shape, dtype=tensor.dtype
                )
                dram_tensor.copy_(tensor.cpu())
            else:
                # Fallback: PyTorch pinned memory
                dram_tensor = torch.empty(
                    tensor.shape, dtype=tensor.dtype, pin_memory=True
                )
                dram_tensor.copy_(tensor.cpu())
            cpu_weights[name] = dram_tensor

        self.dram_store[key] = cpu_weights
        self._registered_layers.add(layer_id)

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

        # Check HBM LRU cache
        if key in self.hbm_cache:
            self.hbm_cache.move_to_end(key)  # LRU: mark as recently used
            self._stats["hbm_hit"] += 1
            return self.hbm_cache[key]

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

        for eid in expert_ids:
            key = (layer_id, eid)
            self._stats["total_requests"] += 1

            if key not in self.dram_store:
                continue

            if key in self.hbm_cache:
                self.hbm_cache.move_to_end(key)  # LRU: mark as recently used
                self._stats["hbm_hit"] += 1
                results[eid] = self.hbm_cache[key]
            else:
                self._stats["dram_load"] += 1
                missing.append(key)

        if not missing:
            return results

        if self.use_acc_offload and self._offload_initialized:
            # Use acc_offload sparse_copy (AICore MTE, 32-core parallel)
            results.update(self._sparse_copy_batch(layer_id, missing))
        else:
            # Fallback: PyTorch H2D on dedicated stream
            results.update(self._pytorch_h2d_batch(missing))

        return results

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

        return expert_results

    # ------------------------------------------------------------------
    # PyTorch H2D backend (fallback)
    # ------------------------------------------------------------------

    def _pytorch_h2d_batch(
        self, missing_keys: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load experts using PyTorch H2D (fallback)."""
        results = {}

        with torch.npu.stream(self._h2d_stream):
            for key in missing_keys:
                eid = key[1]
                hbm_weights = self._load_to_hbm_unchecked(key)
                results[eid] = hbm_weights
                # Put into HBM cache so hbm_cache_used_bytes is updated
                # and subsequent lookups get cache hits.
                self._put_hbm_cache(key, hbm_weights)

        self._h2d_stream.synchronize()
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
        """Put expert weights into HBM LRU cache, evicting if needed.

        LRU eviction: When cache is full, removes the least recently used
        entry (head of OrderedDict) until there's space for the new entry.
        """
        weight_size = sum(t.nbytes for t in weights.values())

        # Evict LRU entries until we have space
        while (
            self.hbm_cache_used_bytes + weight_size > self.hbm_cache_size_bytes
            and len(self.hbm_cache) > 0
        ):
            evicted_key, evicted_weights = self.hbm_cache.popitem(last=False)
            self.hbm_cache_used_bytes -= sum(
                t.nbytes for t in evicted_weights.values()
            )
            del evicted_weights

        self.hbm_cache[key] = weights
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
        for eid in expert_ids:
            key = (layer_id, eid)
            if key not in self.dram_store or key in self.hbm_cache:
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

    def warmup_hbm_cache(self, num_layers: int = 0):
        """Pre-populate HBM cache with experts from the first N layers.

        Called after all expert weights are registered to DRAM. Loads
        the first `num_layers` layers' experts into HBM cache so that
        the initial forward passes have high cache hit rates.

        Also auto-calculates the HBM cache size based on the number of
        layers (with 20% headroom for on-demand loading of other layers).

        Args:
            num_layers: Number of layers to pre-load into HBM.
                        0 means use self.hbm_cache_layers.
        """
        self._ensure_initialized()

        if num_layers > 0:
            self.hbm_cache_layers = num_layers

        if self.hbm_cache_layers <= 0:
            return

        # Get sorted list of registered layers
        sorted_layers = sorted(self._registered_layers)
        layers_to_load = sorted_layers[: self.hbm_cache_layers]

        if not layers_to_load:
            return

        # Auto-calculate HBM cache size from layer count
        total_size = 0
        for layer_id in layers_to_load:
            for key, weights in self.dram_store.items():
                if key[0] == layer_id:
                    total_size += sum(t.nbytes for t in weights.values())
        # Add 20% headroom for on-demand loading of other layers' experts
        self.hbm_cache_size_bytes = int(total_size * 1.2)
        logger.info(
            f"[ExpertWeightStore] HBM cache sized: "
            f"{self.hbm_cache_size_bytes / 1024**3:.1f} GB "
            f"for {len(layers_to_load)} layers"
        )

        # Batch load all experts from the first N layers into HBM
        loaded_count = 0
        for layer_id in layers_to_load:
            # Collect all expert IDs for this layer
            expert_ids = [
                key[1]
                for key in self.dram_store.keys()
                if key[0] == layer_id
            ]
            if not expert_ids:
                continue

            self.batch_get_expert_weights(layer_id, expert_ids)
            loaded_count += len(expert_ids)

        logger.info(
            f"[ExpertWeightStore] Warmup complete: loaded {loaded_count} experts "
            f"from {len(layers_to_load)} layers into HBM cache "
            f"({self.hbm_cache_used_bytes / 1024**3:.1f} GB used)"
        )

    def release_hbm_weights(self):
        """Release all HBM cached weights and shared buffers (free HBM memory)."""
        self.hbm_cache.clear()
        self.hbm_cache_used_bytes = 0
        self._shared_hbm_buffers.clear()
        self._shared_buffer_shapes.clear()

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
            "hbm_hit_rate": self._stats["hbm_hit"] / total,
            "dram_load_count": self._stats["dram_load"],
            "hbm_cache_count": len(self.hbm_cache),
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
