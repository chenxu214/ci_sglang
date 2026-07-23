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
    if not torch.npu.is_available():
        return 0.0, 0.0
    return (
        torch.npu.memory_allocated() / 1024**3,
        torch.npu.memory_reserved() / 1024**3,
    )


class ExpertWeightStore:
    """Manages MoE expert weights across Host DRAM and HBM."""

    def __init__(
        self,
        dram_pool_size_gb: float = 1300.0,
        use_acc_offload: bool = True,
        shared_buffer_max_gb: float = 0,
    ):
        self.dram_store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.hbm_cache: OrderedDict[Tuple[int, int], Dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self.hbm_cache_size_bytes = 0
        self.hbm_cache_used_bytes = 0

        self._h2d_stream = None
        self._initialized = False

        self.use_acc_offload = use_acc_offload
        self._offload = None
        self._offload_initialized = False
        self._dram_pool_size_bytes = int(dram_pool_size_gb * 1024**3)

        self._registered_layers: set = set()
        self._nz_weight_names: set = set()

        self._shared_hbm_buffers: Dict[str, torch.Tensor] = {}
        self._shared_buffer_shapes: Dict[str, tuple] = {}
        self._shared_buffer_max_bytes = int(shared_buffer_max_gb * 1024**3)

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
                    f"[ExpertWeightStore] acc_offload enabled: "
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

    def _check_shared_buffer_budget(self, name: str, nbytes: int) -> bool:
        # 0 = shared buffer disabled; all layers use per-forward allocation
        # (lowest HBM footprint, higher allocation overhead per forward).
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

    def _allocate_hbm_buffer(
        self, shape: tuple, dtype: torch.dtype, nz_format: bool = False
    ) -> torch.Tensor:
        """Allocate an HBM buffer. For NZ format, creates a transposed
        FRACTAL_NZ buffer matching CANN kernel expectations."""
        target_device = (
            f"npu:{torch.npu.current_device()}"
            if torch.npu.is_available()
            else "cpu"
        )
        if nz_format:
            import torch_npu
            original_shape = list(shape)
            if len(original_shape) >= 3:
                original_shape[1], original_shape[2] = (
                    original_shape[2],
                    original_shape[1],
                )
            buf = torch.empty(
                tuple(original_shape), dtype=dtype, device=target_device
            )
            buf = torch_npu.npu_format_cast(buf, 29)
            buf = buf.transpose(1, 2)
        else:
            buf = torch.empty(shape, dtype=dtype, device=target_device)
        return buf

    def get_shared_hbm_buffer(
        self, name: str, shape: tuple, dtype: torch.dtype,
        nz_format: bool = False,
    ) -> Optional[torch.Tensor]:
        """Get or create a cached shared HBM buffer.

        Returns None if the buffer would exceed the HBM budget.
        """
        if name in self._shared_hbm_buffers:
            return self._shared_hbm_buffers[name]

        estimated_nbytes = 1
        for s in shape:
            estimated_nbytes *= s
        estimated_nbytes *= torch.tensor([], dtype=dtype).element_size()
        if nz_format:
            estimated_nbytes = int(estimated_nbytes * 1.2)

        if not self._check_shared_buffer_budget(name, estimated_nbytes):
            return None

        buf = self._allocate_hbm_buffer(shape, dtype, nz_format=nz_format)
        self._shared_hbm_buffers[name] = buf
        self._shared_buffer_shapes[name] = shape
        total_mb = sum(
            t.nbytes for t in self._shared_hbm_buffers.values()
        ) / 1024**2
        logger.info(
            f"[ExpertWeightStore] Allocated shared HBM buffer '{name}': "
            f"shape={shape}, dtype={dtype}, nz={nz_format}, "
            f"size={buf.nbytes / 1024**2:.1f} MB, "
            f"total_shared={total_mb:.1f} MB"
        )
        return buf

    def allocate_temp_hbm_buffer(
        self, shape: tuple, dtype: torch.dtype, nz_format: bool = False
    ) -> torch.Tensor:
        """Allocate a temporary (non-cached) HBM buffer for per-forward use.
        Not subject to the shared buffer budget."""
        return self._allocate_hbm_buffer(shape, dtype, nz_format=nz_format)

    def register_expert(
        self,
        layer_id: int,
        expert_id: int,
        weights: Dict[str, torch.Tensor],
        nz_weight_names: Optional[set] = None,
    ):
        """Register expert weights from HBM to Host DRAM.

        NZ-format weights use sparse_copy (raw memcpy) to bypass
        PyTorch format conversion. Contiguous tensors use copy_().
        """
        self._ensure_initialized()
        key = (layer_id, expert_id)
        nz_weight_names = nz_weight_names or set()

        cpu_weights = {}
        src_ptrs = []
        dst_ptrs = []
        len_ptrs = []

        for name, tensor in weights.items():
            is_nz = name in nz_weight_names
            if is_nz:
                self._nz_weight_names.add(name)

            if self.use_acc_offload and self._offload_initialized:
                dram_tensor = self._offload.empty(
                    tensor.shape, dtype=tensor.dtype
                )
                if is_nz:
                    src_ptrs.append(tensor.data_ptr())
                    dst_ptrs.append(dram_tensor.data_ptr())
                    len_ptrs.append(tensor.nbytes)
                else:
                    dram_tensor.copy_(tensor)
            else:
                # Weights are ND format (NZ cast skipped for DRAM offload),
                # so copy_() works directly.
                dram_tensor = torch.empty(
                    tensor.shape, dtype=tensor.dtype, pin_memory=True
                )
                dram_tensor.copy_(tensor)

            cpu_weights[name] = dram_tensor

        if src_ptrs:
            num_pairs = len(src_ptrs)
            if num_pairs % 2 != 0:
                src_ptrs.append(src_ptrs[-1])
                dst_ptrs.append(dst_ptrs[-1])
                len_ptrs.append(len_ptrs[-1])
                num_pairs = len(src_ptrs)

            src_tensor = torch.tensor(src_ptrs, dtype=torch.int64).npu()
            dst_tensor = torch.tensor(dst_ptrs, dtype=torch.int64).npu()
            len_tensor = torch.tensor(len_ptrs, dtype=torch.int32).npu()
            size_tensor = torch.tensor(num_pairs, dtype=torch.int32).npu()

            device = torch.device(f"npu:{torch.npu.current_device()}")
            with torch.npu.stream(self._h2d_stream):
                ret = self._offload.sparse_copy(
                    src_tensor, dst_tensor, len_tensor, size_tensor, device
                )
            self._h2d_stream.synchronize()
            if ret != 0:
                raise RuntimeError(
                    f"[ExpertWeightStore] D2H sparse_copy failed (ret={ret})"
                )

        self.dram_store[key] = cpu_weights
        self._registered_layers.add(layer_id)

    def get_expert_weights(
        self, layer_id: int, expert_id: int
    ) -> Optional[Dict[str, torch.Tensor]]:
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
        """Batch load expert weights directly into shared HBM buffers."""
        self._ensure_initialized()
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
                dst_tensor = shared_buffers[name][eid]
                expert_views[name] = dst_tensor

                src_ptrs.append(dram_tensor.data_ptr())
                dst_ptrs.append(dst_tensor.data_ptr())
                len_ptrs.append(dram_tensor.nbytes)

            results[eid] = expert_views

        num_pairs = len(src_ptrs)
        if num_pairs == 0:
            return results

        if num_pairs % 2 != 0:
            src_ptrs.append(src_ptrs[-1])
            dst_ptrs.append(dst_ptrs[-1])
            len_ptrs.append(len_ptrs[-1])
            num_pairs = len(src_ptrs)

        if self.use_acc_offload and self._offload_initialized:
            src_tensor = torch.tensor(src_ptrs, dtype=torch.int64).npu()
            dst_tensor = torch.tensor(dst_ptrs, dtype=torch.int64).npu()
            len_tensor = torch.tensor(len_ptrs, dtype=torch.int32).npu()
            size_tensor = torch.tensor(num_pairs, dtype=torch.int32).npu()

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

    def _sparse_copy_batch(
        self, layer_id: int, missing_keys: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        src_ptrs = []
        dst_ptrs = []
        len_ptrs = []
        expert_results = {}

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
        if num_pairs % 2 != 0:
            src_ptrs.append(src_ptrs[-1])
            dst_ptrs.append(dst_ptrs[-1])
            len_ptrs.append(len_ptrs[-1])
            num_pairs = len(src_ptrs)

        src_tensor = torch.tensor(src_ptrs, dtype=torch.int64).npu()
        dst_tensor = torch.tensor(dst_ptrs, dtype=torch.int64).npu()
        len_tensor = torch.tensor(len_ptrs, dtype=torch.int32).npu()
        size_tensor = torch.tensor(num_pairs, dtype=torch.int32).npu()

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

    def _pytorch_h2d_batch(
        self, missing_keys: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        results = {}
        layer_id = missing_keys[0][0] if missing_keys else -1

        with torch.npu.stream(self._h2d_stream):
            for key in missing_keys:
                eid = key[1]
                dram_weights = self.dram_store[key]
                hbm_weights = {}
                for name, cpu_tensor in dram_weights.items():
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
        hbm_weights = self._load_to_hbm_unchecked(key)
        self._put_hbm_cache(key, hbm_weights)
        return hbm_weights

    def _load_to_hbm_unchecked(
        self, key: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        dram_weights = self.dram_store[key]
        hbm_weights = {}
        for name, cpu_tensor in dram_weights.items():
            hbm_tensor = cpu_tensor.to("npu", non_blocking=True)
            hbm_weights[name] = hbm_tensor
        return hbm_weights

    def _put_hbm_cache(
        self, key: Tuple[int, int], weights: Dict[str, torch.Tensor]
    ):
        weight_size = sum(t.nbytes for t in weights.values())
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
        """Release all HBM cached weights and shared buffers."""
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
        total = 0
        for weights in self.dram_store.values():
            total += sum(t.nbytes for t in weights.values())
        return total / 1024**3
