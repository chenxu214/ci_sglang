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
import torch_npu

from sglang.srt.utils.common import get_int_env_var

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
        hbm_cache_layers: int = 0,
        dram_pool_size_gb: float = 1300.0,
        use_acc_offload: bool = True,
    ):
        self.dram_store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        # Per-layer LRU caches: {layer_id: OrderedDict[expert_id, weights]}
        # Each layer has its own slot-count limit (20 for decode, unlimited
        # for prefill). This prevents cross-layer eviction during decode.
        self._per_layer_caches: Dict[int, OrderedDict] = {}
        self.hbm_cache_size_bytes = 0  # Auto-calculated during warmup
        self.hbm_cache_used_bytes = 0
        self.hbm_cache_layers = hbm_cache_layers

        # Slot-count limit for decode LRU (env-tunable). When > 0, evict by
        # entry count in addition to byte-size. Set to 0 (unlimited) during
        # prefill via set_cache_mode().
        self._decode_cache_slots = get_int_env_var(
            "SGLANG_KIMI_DECODE_CACHE_SLOTS", 20
        )
        self.hbm_cache_max_slots = self._decode_cache_slots

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
                self._h2d_stream = torch.npu.Stream()
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
        """Get or create a shared HBM buffer (reused across all MoE layers)."""
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
        """
        self._ensure_initialized()
        key = (layer_id, expert_id)

        cpu_weights = {}
        total_bytes = 0
        for name, tensor in weights.items():
            # NZ format (FRACTAL_NZ, format=29) cannot be copied via copy_()
            # or .cpu() — NPU raises "do not support internal format".
            # Convert to ND (format=0) before D2H transfer.
            try:
                if tensor.npu_format == 29:  # FRACTAL_NZ
                    tensor = torch_npu.npu_format_cast(tensor, 0)
            except AttributeError:
                pass  # Not an NPU tensor or no npu_format attr

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

        lc = self._get_layer_cache(layer_id)
        if expert_id in lc:
            lc.move_to_end(expert_id)
            self._stats["hbm_hit"] += 1
            return lc[expert_id]

        self._stats["dram_load"] += 1
        return self._load_to_hbm(key)

    def batch_get_expert_weights(
        self, layer_id: int, expert_ids: List[int]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load expert weights to HBM.

        Uses acc_offload sparse_copy (32-core MTE parallel) when available,
        otherwise PyTorch H2D on a dedicated stream.
        """
        self._ensure_initialized()
        results = {}
        missing = []

        lc = self._get_layer_cache(layer_id)
        for eid in expert_ids:
            key = (layer_id, eid)
            self._stats["total_requests"] += 1

            if key not in self.dram_store:
                continue

            if eid in lc:
                lc.move_to_end(eid)
                self._stats["hbm_hit"] += 1
                results[eid] = lc[eid]
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

    # ------------------------------------------------------------------
    # acc_offload sparse_copy backend
    # ------------------------------------------------------------------

    def _sparse_copy_batch(
        self, layer_id: int, missing_keys: List[Tuple[int, int]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load experts using acc_offload sparse_copy (32-core MTE parallel)."""
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
                    device="npu",
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
        src_tensor = torch.tensor(src_ptrs, dtype=torch.int64, device="npu")
        dst_tensor = torch.tensor(dst_ptrs, dtype=torch.int64, device="npu")
        len_tensor = torch.tensor(len_ptrs, dtype=torch.int32, device="npu")
        size_tensor = torch.tensor(num_pairs, dtype=torch.int32, device="npu")

        device = torch.npu.current_device()
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

        # Global byte-size eviction (warmup-based; 0 = skip)
        if self.hbm_cache_size_bytes > 0:
            while (
                self.hbm_cache_used_bytes + weight_size > self.hbm_cache_size_bytes
                and len(lc) > 0
            ):
                evicted_eid, evicted_weights = lc.popitem(last=False)
                evicted_size = sum(t.nbytes for t in evicted_weights.values())
                self.hbm_cache_used_bytes -= evicted_size
                del evicted_weights

        lc[expert_id] = weights
        self.hbm_cache_used_bytes += weight_size

    def async_prefetch(self, layer_id: int, expert_ids: List[int]):
        """Asynchronously prefetch experts for the next layer on h2d_stream."""
        self._ensure_initialized()
        if self._h2d_stream is None:
            return

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
            with torch.npu.stream(self._h2d_stream):
                self._sparse_copy_batch(layer_id, missing)
        else:
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

    def warmup_hbm_cache(self, num_layers: int = 0):
        """Pre-populate HBM cache with experts from the first N layers.

        Also auto-calculates the HBM cache size based on the number of
        layers (with 20% headroom for on-demand loading of other layers).
        """
        self._ensure_initialized()

        if num_layers > 0:
            self.hbm_cache_layers = num_layers

        if self.hbm_cache_layers <= 0:
            return

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
        self.hbm_cache_size_bytes = int(total_size * 1.2)

        alloc_before, _ = _get_hbm_usage_gb()
        logger.info(
            f"[ExpertWeightStore] HBM cache sized: "
            f"{self.hbm_cache_size_bytes / 1024**3:.1f} GB "
            f"for {len(layers_to_load)} layers. "
            f"HBM before warmup: alloc={alloc_before:.2f} GB"
        )

        loaded_count = 0
        for layer_id in layers_to_load:
            expert_ids = [
                key[1]
                for key in self.dram_store.keys()
                if key[0] == layer_id
            ]
            if not expert_ids:
                continue

            self.batch_get_expert_weights(layer_id, expert_ids)
            loaded_count += len(expert_ids)

        alloc_after, _ = _get_hbm_usage_gb()
        logger.info(
            f"[ExpertWeightStore] Warmup complete: {loaded_count} experts "
            f"({self.hbm_cache_used_bytes / 1024**3:.1f} GB). "
            f"HBM {alloc_before:.2f}→{alloc_after:.2f} GB"
        )

    def release_hbm_weights(self):
        """Release all HBM cached weights and shared buffers (free HBM memory)."""
        cache_count = sum(len(lc) for lc in self._per_layer_caches.values())
        cache_gb = self.hbm_cache_used_bytes / 1024**3
        shared_buffer_count = len(self._shared_hbm_buffers)
        shared_buffer_bytes = sum(t.nbytes for t in self._shared_hbm_buffers.values())

        alloc_before, _ = _get_hbm_usage_gb()

        self._per_layer_caches.clear()
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
