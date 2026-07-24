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
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.utils.common import get_int_env_var
from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

logger = logging.getLogger(__name__)


class ExpertWeightStore:
    """Manages MoE expert weights across Host DRAM and HBM.

    Weights are stored in Host DRAM after process_weights_after_loading().
    During forward, only Top-K selected experts are loaded from DRAM to HBM
    via one of three paths:
      - Prefill: prefetch_layer_to_buffer() loads ALL experts into standalone
        per-layer HBM buffers.
      - Decode (on-demand): batch_load_to_shared_buffer() loads selected
        experts into a per-forward temporary HBM buffer.
      - Decode (compact): build_active_weight_tensors() loads only active
        experts into pre-allocated decode buffers with slot-map reuse.

    Attributes:
        dram_store: {(layer_id, expert_id): {weight_name: cpu_tensor}}
        h2d_stream: Dedicated NPU stream for H2D transfers
        use_acc_offload: Whether to use acc_offload sparse_copy
    """

    def __init__(
        self,
        dram_pool_size_gb: float = 1300.0,
        use_acc_offload: bool = True,
    ):
        self.dram_store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        # Pre-allocated decode buffers: {layer_id: {name: [max_slots, ...]}}
        # Reused across decode steps to avoid per-step torch.stack allocation.
        self._decode_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        # Slot map for decode: {layer_id: OrderedDict[expert_id, slot_index]}
        # Maps expert IDs to buffer slots for cache hits during decode.
        self._decode_slot_maps: Dict[int, OrderedDict] = {}

        # Slot-count limit for decode (env-tunable). When > 0, indicates
        # decode mode (used by w4a8/w4a16 to select compact weight path).
        # Set to 0 during prefill via set_cache_mode().
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

    def batch_load_to_shared_buffer(
        self,
        layer_id: int,
        expert_ids: List[int],
        shared_buffers: Dict[str, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load expert weights directly into caller-provided HBM buffers.

        Weights are written directly into shared_buffers[expert_id],
        avoiding per-expert HBM tensor allocation.

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

            # Always load from DRAM
            self._stats["dram_load"] += 1
            missing.append(key)

        if not missing:
            return results

        # Build (src_ptr, dst_ptr, len) triples pointing directly into
        # the caller-provided buffers.
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
                # Destination: expert's slot in the buffer
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
                # Fallback: PyTorch H2D into buffers
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
            # PyTorch H2D directly into buffers
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

        Uses pre-allocated _decode_buffers as the sole storage. A slot map
        (expert_id → slot_index) tracks which expert is in which buffer slot
        for cache hits.

        Flow:
          1. Assign slots: hits reuse old slot, misses get free/evicted slot
          2. Load misses from DRAM directly into buffer slots (no per-expert
             tensor allocation)
          3. Compact: swap data to sequential positions 0..num_active-1
          4. Update slot map: all active experts → their compacted positions

        Args:
            layer_id: Layer index
            active_expert_ids: Sorted list of expert IDs with tokens
            weight_names: List of weight parameter names

        Returns:
            {weight_name: tensor of shape [num_active, ...]}
        """
        self._ensure_initialized()

        num_active = len(active_expert_ids)
        max_slots = self._decode_cache_slots

        # Get or create decode buffers
        if layer_id not in self._decode_buffers:
            sample_key = (layer_id, 0)
            buffers = {}
            for name in weight_names:
                sample_tensor = self.dram_store[sample_key][name]
                full_shape = (max_slots,) + sample_tensor.shape
                buffers[name] = torch.empty(
                    full_shape, dtype=sample_tensor.dtype, device="npu"
                )
            self._decode_buffers[layer_id] = buffers
        buffers = self._decode_buffers[layer_id]

        # Get or create slot map (expert_id → slot_index)
        if layer_id not in self._decode_slot_maps:
            self._decode_slot_maps[layer_id] = OrderedDict()
        slot_map = self._decode_slot_maps[layer_id]

        # Assign slots: hits reuse old slot, misses get free/evicted slot
        assigned = {}  # {expert_id: slot_index}
        for eid in active_expert_ids:
            self._stats["total_requests"] += 1
            if eid in slot_map:
                self._stats["hbm_hit"] += 1
                assigned[eid] = slot_map.pop(eid)
            else:
                self._stats["dram_load"] += 1

        # Find free slots for misses (not in current slot_map or assigned)
        used = set(slot_map.values()) | set(assigned.values())
        free_slots = [s for s in range(max_slots) if s not in used]

        for eid in active_expert_ids:
            if eid not in assigned:
                if free_slots:
                    assigned[eid] = free_slots.pop(0)
                elif slot_map:
                    evicted_eid, evicted_slot = slot_map.popitem(last=False)
                    assigned[eid] = evicted_slot
                else:
                    raise RuntimeError(
                        f"No free decode buffer slots for layer {layer_id}, "
                        f"num_active={num_active}, max_slots={max_slots}"
                    )

        # Load misses from DRAM directly into buffer slots (no per-expert
        # tensor allocation)
        for eid in active_expert_ids:
            if eid not in slot_map:  # miss
                slot = assigned[eid]
                key = (layer_id, eid)
                dram_weights = self.dram_store[key]
                for name, dram_tensor in dram_weights.items():
                    if name in buffers:
                        buffers[name][slot].copy_(
                            dram_tensor, non_blocking=True
                        )

        # Compact: swap data to sequential positions 0..num_active-1.
        # Uses swap (3-way via temp slot) to avoid data loss.
        # slot_to_eid tracks which expert is at each slot for O(1) lookup.
        temp_slot = max_slots - 1
        slot_to_eid = {s: e for e, s in assigned.items()}

        for i in range(num_active):
            eid = active_expert_ids[i]
            src_slot = assigned[eid]
            if src_slot == i:
                continue
            # Swap buf[i] and buf[src_slot] via temp
            for name in weight_names:
                buf = buffers[name]
                buf[temp_slot].copy_(buf[i])
                buf[i].copy_(buf[src_slot])
                buf[src_slot].copy_(buf[temp_slot])
            # Update mappings
            other_eid = slot_to_eid.get(i)
            assigned[eid] = i
            slot_to_eid[i] = eid
            if other_eid is not None:
                assigned[other_eid] = src_slot
                slot_to_eid[src_slot] = other_eid

        # Update slot map: all active experts at sequential positions
        slot_map.clear()
        for i, eid in enumerate(active_expert_ids):
            slot_map[eid] = i

        # Return views into pre-allocated buffer
        result = {}
        for name in weight_names:
            result[name] = buffers[name][:num_active]

        return result

    # ------------------------------------------------------------------
    # Prefill full-layer prefetch + cache mode management
    # ------------------------------------------------------------------ #

    def set_cache_mode(self, is_prefill: bool):
        """Toggle between prefill and decode mode.

        Sets hbm_cache_max_slots to 0 (prefill, unlimited) or
        _decode_cache_slots (decode, 20 by default). The w4a8/w4a16
        schemes check this flag to select the compact weight path.
        """
        if is_prefill:
            self.hbm_cache_max_slots = 0  # prefill mode
            # Free decode buffers and slot maps (not needed during prefill).
            self._decode_buffers.clear()
            self._decode_slot_maps.clear()
        else:
            self.hbm_cache_max_slots = self._decode_cache_slots

    def release_decode_buffers(self):
        """Release decode HBM buffers to free HBM.

        Called before prefill to free decode buffers. Layer weight
        references must also be set to None (via _release_dram_offload_weights)
        to fully release the tensor memory.
        """
        self._decode_buffers.clear()
        self._decode_slot_maps.clear()
        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()

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
        h2d_stream (async, no sync). Caller must call sync_prefetch()
        before using the buffers, and free_layer_buffers() after compute
        to release HBM.

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
        """Free per-layer HBM buffers allocated by prefetch_layer_to_buffer.

        Only clears Python references; the caching allocator reclaims and
        reuses the memory automatically. No gc.collect()/empty_cache() here
        — empty_cache() triggers a device-wide sync on NPU, which waits for
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
            "hbm_hit_rate": self._stats["hbm_hit"] / total,
            "dram_load_count": self._stats["dram_load"],
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
        """Release all HBM cached weights.

        Called after offload registration to free HBM used during the
        registration process. Decode buffers should be empty at this
        point (not yet used), so this is mostly gc + empty_cache.
        """
        self._decode_buffers.clear()
        self._decode_slot_maps.clear()

        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
