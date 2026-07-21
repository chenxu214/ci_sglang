# SPDX-License-Identifier: Apache-2.0
"""Expert weight store for MoE DRAM offloading.

Manages MoE expert weights in Host DRAM with an LRU cache in HBM.
During forward, only the Top-K selected experts are loaded from
Host DRAM to HBM on demand.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _get_hbm_usage_gb() -> Tuple[float, float]:
    """Get current HBM (allocated, reserved) memory in GB."""
    if not torch.npu.is_available():
        return 0.0, 0.0
    return (
        torch.npu.memory_allocated() / 1024**3,
        torch.npu.memory_reserved() / 1024**3,
    )


class ExpertWeightStore:
    """Manages MoE expert weights across Host DRAM and HBM.

    Weights are stored in Host DRAM after process_weights_after_loading().
    An LRU cache in HBM holds recently-used experts. During forward,
    only Top-K selected experts are loaded from DRAM to HBM.
    """

    def __init__(
        self,
        dram_pool_size_gb: float = 1300.0,
        use_acc_offload: bool = True,
    ):
        self.dram_store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.hbm_cache: OrderedDict[Tuple[int, int], Dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self.hbm_cache_size_bytes = 0  # 0 = unlimited (no eviction)
        self.hbm_cache_used_bytes = 0

        self._h2d_stream = None
        self._initialized = False

        self.use_acc_offload = use_acc_offload
        self._offload = None
        self._offload_initialized = False
        self._dram_pool_size_bytes = int(dram_pool_size_gb * 1024**3)

        self._registered_layers: set = set()

        # Shared HBM weight buffer: all MoE layers reuse the same buffer.
        # Before each layer's forward, Top-K experts are loaded into it.
        self._shared_hbm_buffers: Dict[str, torch.Tensor] = {}
        self._shared_buffer_shapes: Dict[str, tuple] = {}

        self._stats = {"hbm_hit": 0, "dram_load": 0, "total_requests": 0}

    def _ensure_initialized(self):
        if not self._initialized:
            if torch.npu.is_available():
                self._device = f"npu:{torch.npu.current_device()}"
                self._h2d_stream = torch.npu.Stream()
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
            self._offload_device_id = config.device_id
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
        """Get or create a shared HBM buffer (reused across all MoE layers)."""
        if name not in self._shared_hbm_buffers:
            target_device = (
                f"npu:{torch.npu.current_device()}"
                if torch.npu.is_available()
                else "cpu"
            )
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
        """
        self._ensure_initialized()
        key = (layer_id, expert_id)

        cpu_weights = {}
        total_bytes = 0
        for name, tensor in weights.items():
            if self.use_acc_offload and self._offload_initialized:
                dram_tensor = self._offload.empty(
                    tensor.shape, dtype=tensor.dtype
                )
            else:
                dram_tensor = torch.empty(
                    tensor.shape, dtype=tensor.dtype, pin_memory=True
                )
            # Direct cross-device copy (HBM→DRAM or CPU→DRAM) avoids
            # the temporary CPU copy that tensor.cpu() would create.
            dram_tensor.copy_(tensor)
            cpu_weights[name] = dram_tensor
            total_bytes += dram_tensor.nbytes

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

        if key in self.hbm_cache:
            self.hbm_cache.move_to_end(key)
            self._stats["hbm_hit"] += 1
            return self.hbm_cache[key]

        self._stats["dram_load"] += 1
        return self._load_to_hbm(key)

    def batch_get_expert_weights(
        self, layer_id: int, expert_ids: List[int]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load expert weights to HBM.

        Uses acc_offload sparse_copy when available,
        otherwise PyTorch H2D on a dedicated stream.
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
                self.hbm_cache.move_to_end(key)
                self._stats["hbm_hit"] += 1
                results[eid] = self.hbm_cache[key]
            else:
                self._stats["dram_load"] += 1
                missing.append(key)

        if not missing:
            return results

        if self.use_acc_offload and self._offload_initialized:
            results.update(self._sparse_copy_batch(layer_id, missing))
        else:
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

        # Debug: log sparse_copy parameters (first call only)
        if not getattr(self, "_logged_sparse_copy_params", False):
            self._logged_sparse_copy_params = True
            logger.warning(
                f"[ExpertWeightStore] sparse_copy H2D first call: "
                f"num_pairs={num_pairs} (must be even for K/V pair design), "
                f"current_device={torch.npu.current_device()}, "
                f"offload_device={getattr(self, '_offload_device_id', '?')}, "
                f"h2d_stream={self._h2d_stream}"
            )
            for i in range(min(4, num_pairs)):
                logger.warning(
                    f"  pair[{i}]: src_ptr={hex(src_ptrs[i])}, "
                    f"dst_ptr={hex(dst_ptrs[i])}, len={len_ptrs[i]} "
                    f"({len_ptrs[i] / 1024**2:.1f} MB)"
                )
            if num_pairs % 2 != 0:
                logger.error(
                    f"[ExpertWeightStore] num_pairs={num_pairs} is ODD! "
                    f"sparse_copy will drop last pair (size_={num_pairs // 2})"
                )

        if self.use_acc_offload and self._offload_initialized:
            src_tensor = torch.tensor(src_ptrs, dtype=torch.int64, device=self._device)
            dst_tensor = torch.tensor(dst_ptrs, dtype=torch.int64, device=self._device)
            len_tensor = torch.tensor(len_ptrs, dtype=torch.int32, device=self._device)
            # Use 1-dim tensor (not 0-dim scalar) to ensure data_ptr()
            # returns a valid address. NPU's 0-dim tensor data_ptr()
            # may be unreliable in some torch_npu versions.
            size_tensor = torch.tensor([num_pairs], dtype=torch.int32, device=self._device)

            # device from HBM tensor, matching reference example
            device = shared_buffers[
                list(shared_buffers.keys())[0]
            ].device
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

    # ------------------------------------------------------------------
    # acc_offload sparse_copy backend
    # ------------------------------------------------------------------

    def _sparse_copy_batch(
        self, layer_id: int, missing_keys: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load experts using acc_offload sparse_copy."""
        src_ptrs = []
        dst_ptrs = []
        len_ptrs = []
        expert_results = {}

        # Collect all (src, dst, len) pairs for all missing experts.
        # HBM destinations are allocated upfront; transfer happens in a
        # single batched sparse_copy call below.
        for key in missing_keys:
            eid = key[1]
            dram_weights = self.dram_store[key]
            hbm_weights = {}

            for name, dram_tensor in dram_weights.items():
                hbm_tensor = torch.empty(
                    dram_tensor.shape,
                    dtype=dram_tensor.dtype,
                    device=self._device,
                )
                hbm_weights[name] = hbm_tensor

                src_ptrs.append(dram_tensor.data_ptr())
                dst_ptrs.append(hbm_tensor.data_ptr())
                len_ptrs.append(dram_tensor.nbytes)

            expert_results[eid] = hbm_weights
            self._put_hbm_cache(key, hbm_weights)

        if not src_ptrs:
            return expert_results

        num_pairs = len(src_ptrs)
        src_tensor = torch.tensor(src_ptrs, dtype=torch.int64, device=self._device)
        dst_tensor = torch.tensor(dst_ptrs, dtype=torch.int64, device=self._device)
        len_tensor = torch.tensor(len_ptrs, dtype=torch.int32, device=self._device)
        size_tensor = torch.tensor([num_pairs], dtype=torch.int32, device=self._device)

        # sparse_copy wrapper expects a torch.device (uses .index attribute)
        device = torch.device(f"npu:{torch.npu.current_device()}")
        with torch.npu.stream(self._h2d_stream):
            ret = self._offload.sparse_copy(
                src_tensor, dst_tensor, len_tensor, size_tensor, device
            )
            if ret != 0:
                logger.error(
                    f"[ExpertWeightStore] sparse_copy failed (ret={ret}), "
                    f"falling back to PyTorch H2D"
                )
                return self._pytorch_h2d_batch(missing_keys)

        self._h2d_stream.synchronize()

        logger.debug(
            f"[ExpertWeightStore] sparse_copy layer={layer_id}: "
            f"{len(missing_keys)} experts, {num_pairs} tensors"
        )

        return expert_results

    # ------------------------------------------------------------------
    # PyTorch H2D backend (fallback)
    # ------------------------------------------------------------------

    def _pytorch_h2d_batch(
        self, missing_keys: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load experts using PyTorch H2D (fallback).

        All H2D copies are submitted on the h2d_stream before synchronizing,
        allowing the DMA engine to pipeline multiple transfers.
        """
        results = {}
        layer_id = missing_keys[0][0] if missing_keys else -1

        with torch.npu.stream(self._h2d_stream):
            for key in missing_keys:
                eid = key[1]
                dram_weights = self.dram_store[key]
                hbm_weights = {}
                for name, cpu_tensor in dram_weights.items():
                    # non_blocking=True enables async H2D on the stream.
                    # Pinned memory (allocated in register_expert) enables
                    # true async DMA without an internal synchronous copy.
                    hbm_tensor = cpu_tensor.to("npu", non_blocking=True)
                    hbm_weights[name] = hbm_tensor
                results[eid] = hbm_weights
                self._put_hbm_cache(key, hbm_weights)

        self._h2d_stream.synchronize()

        logger.debug(
            f"[ExpertWeightStore] pytorch_h2d layer={layer_id}: "
            f"{len(missing_keys)} experts"
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
        """Copy expert weights from DRAM to HBM without cache management."""
        dram_weights = self.dram_store[key]
        hbm_weights = {}
        for name, cpu_tensor in dram_weights.items():
            hbm_tensor = cpu_tensor.to("npu", non_blocking=True)
            hbm_weights[name] = hbm_tensor
        return hbm_weights

    def _put_hbm_cache(
        self, key: Tuple[int, int], weights: Dict[str, torch.Tensor]
    ):
        """Put expert weights into HBM LRU cache, evicting if needed."""
        weight_size = sum(t.nbytes for t in weights.values())

        # hbm_cache_size_bytes == 0 means unlimited (no eviction).
        while (
            self.hbm_cache_size_bytes > 0
            and self.hbm_cache_used_bytes + weight_size > self.hbm_cache_size_bytes
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
        """Asynchronously prefetch experts for the next layer on h2d_stream."""
        self._ensure_initialized()
        if self._h2d_stream is None:
            return

        missing = []
        for eid in expert_ids:
            key = (layer_id, eid)
            if key not in self.dram_store or key in self.hbm_cache:
                continue
            missing.append(key)

        if not missing:
            return

        if self.use_acc_offload and self._offload_initialized:
            with torch.npu.stream(self._h2d_stream):
                self._sparse_copy_batch(layer_id, missing)
        else:
            with torch.npu.stream(self._h2d_stream):
                for key in missing:
                    hbm_weights = self._load_to_hbm_unchecked(key)
                    self._put_hbm_cache(key, hbm_weights)

    def release_hbm_weights(self):
        """Release all HBM cached weights and shared buffers (free HBM memory)."""
        cache_count = len(self.hbm_cache)
        cache_gb = self.hbm_cache_used_bytes / 1024**3
        shared_buffer_count = len(self._shared_hbm_buffers)
        shared_buffer_bytes = sum(t.nbytes for t in self._shared_hbm_buffers.values())

        alloc_before, _ = _get_hbm_usage_gb()

        self.hbm_cache.clear()
        self.hbm_cache_used_bytes = 0
        self._shared_hbm_buffers.clear()
        self._shared_buffer_shapes.clear()

        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()

        alloc_after, _ = _get_hbm_usage_gb()

        logger.info(
            f"[ExpertWeightStore] release_hbm_weights: "
            f"cleared {cache_count} experts ({cache_gb:.2f} GB) + "
            f"{shared_buffer_count} buffers ({shared_buffer_bytes / 1024**2:.1f} MB). "
            f"HBM {alloc_before:.2f}→{alloc_after:.2f} GB "
            f"(freed {alloc_before - alloc_after:.2f} GB)"
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
