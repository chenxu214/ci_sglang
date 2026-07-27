# SPDX-License-Identifier: Apache-2.0
"""Expert weight store for MoE DRAM offloading.

Manages MoE expert weights in Host DRAM. During forward, only the
Top-K selected experts are loaded from Host DRAM to HBM on demand.

Two backends are supported:
  1. acc_offload (default when available): Uses MemFabric acc_offload
     group_pack_copy kernel with MTE engine for batch H2D copy.
     Higher performance due to 32-core parallelism and reduced API overhead.
  2. PyTorch H2D (fallback): Uses tensor.copy_(non_blocking=True).
     No external dependency, works everywhere.

Weight loading paths (all use group_pack_copy kernel):
  - Prefill (with prefetch): prefetch_layer_to_buffer() async-loads ALL
    experts for the first N layers on h2d_stream. Uses a fake all-ones
    group_list (real group_list not available pre-dispatch); CANN later
    uses the real group_list from DeepEP dispatch. wait_prefill_prefetch()
    synchronizes via per-layer NPU event before compute.
  - Prefill (no prefetch, N=0): _load_experts_on_demand() loads ALL
    local experts into shared HBM buffers synchronously per layer.
    Same fake all-ones group_list approach as prefetch.
  - Decode: group_pack_copy_active_weights() uses the real post-dispatch
    group_list to load and compact active expert weights on-device,
    outputting a packed group_list for CANN. No D2H sync required.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

logger = logging.getLogger(__name__)


def _drop_kernel_page_cache() -> None:
    """Drop kernel page cache to free contiguous physical memory.

    Huge page allocation (HalMemCreate) requires physically contiguous 2MB
    regions. When the kernel page cache is large (e.g. from safetensors
    mmap during weight loading), fragmentation can cause allocation failures
    even when MemAvailable looks sufficient.

    This function:
      1. Calls sync() to flush dirty pages to disk.
      2. Writes "3" to /proc/sys/vm/drop_caches to free pagecache + slabs.
         (requires root; silently skips if no permission)
      3. Calls malloc_trim(0) to release glibc arenas back to OS.
      4. Sleeps briefly to allow kernel reclaim to settle.
      5. Logs before/after MemAvailable for observability.
    """
    import os
    import time
    import ctypes

    def _read_mem_available_kb() -> int:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    before_kb = _read_mem_available_kb()

    # 1. sync() — flush dirty pages to disk before dropping cache.
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.sync()
    except Exception:
        pass

    # 2. drop_caches — write 3 to free pagecache + dentries + inodes.
    #    Requires root (CAP_SYS_ADMIN). Silently skip if not permitted.
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
    except (PermissionError, OSError):
        # Non-root user — cannot drop kernel cache. Best effort only.
        pass
    except Exception:
        pass

    # 3. malloc_trim(0) — release glibc malloc arenas back to OS.
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

    # 4. Brief sleep to allow kernel reclaim to settle before huge page alloc.
    time.sleep(5)

    after_kb = _read_mem_available_kb()
    delta_gb = (after_kb - before_kb) / 1024 / 1024
    logger.info(
        f"[ExpertWeightStore] Dropped kernel page cache: "
        f"MemAvailable {before_kb / 1024 / 1024:.1f} GB -> "
        f"{after_kb / 1024 / 1024:.1f} GB (delta={delta_gb:+.1f} GB)"
    )


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
    ):
        self.dram_store: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        # Decode mode flag: True during decode, False during prefill.
        # Used to select the weight loading path:
        #   - Prefill: _load_experts_on_demand / prefetch_layer_to_buffer
        #   - Decode: group_pack_copy_active_weights (on-device compaction)
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
        import torch.distributed as dist
        init_failed = False
        init_error: Optional[Exception] = None

        try:
            from memfabric_hybrid import offload
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                for i in range(world_size):
                    if i == rank:
                        try:
                            self._do_acc_offload_init(offload)
                        except Exception as e:
                            # Record failure but continue to participate
                            # in barriers so other ranks don't deadlock.
                            init_failed = True
                            init_error = e
                    dist.barrier()
            else:
                self._do_acc_offload_init(offload)

        except ImportError as e:
            init_failed = True
            init_error = e
        except Exception as e:
            init_failed = True
            init_error = e

        if init_failed:
            logger.warning(
                f"[ExpertWeightStore] acc_offload init failed "
                f"({type(init_error).__name__}: {init_error}). "
                f"Falling back to PyTorch H2D."
            )
            self.use_acc_offload = False

    def _do_acc_offload_init(self, offload):
        """Actual acc_offload initialization (called by _init_acc_offload).

        Strategy: try direct init first (fast path). If it fails, drop
        kernel page cache to free contiguous physical memory for huge
        page allocation, then retry. Up to 3 attempts total.

        Page cache from safetensors mmap can fragment physical memory,
        causing HalMemCreate to fail even when MemAvailable looks
        sufficient. Dropping cache is slow (~5s), so we only do it on
        retry, not the first attempt.
        """
        import time

        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        config.size = self._dram_pool_size_bytes

        max_attempts = 3
        for attempt in range(max_attempts):
            if attempt > 0:
                # Only drop cache on retry — first attempt tries direct init.
                logger.info(
                    f"[ExpertWeightStore] acc_offload init attempt {attempt + 1}/{max_attempts}, "
                    f"dropping page cache before retry..."
                )
                _drop_kernel_page_cache()
                time.sleep(2)
            else:
                logger.info(
                    f"[ExpertWeightStore] acc_offload init attempt {attempt + 1}/{max_attempts} "
                    f"(direct, no cache drop)"
                )

            ret = offload.initialize(config)
            if ret == 0:
                self._offload = offload
                self._offload_initialized = True
                logger.info(
                    f"[ExpertWeightStore] acc_offload initialized: "
                    f"device={config.device_id}, "
                    f"dram_pool={self._dram_pool_size_bytes / 1024**3:.1f} GB "
                    f"(attempt={attempt + 1})"
                )
                return
            logger.warning(
                f"[ExpertWeightStore] acc_offload init attempt {attempt + 1} "
                f"failed (ret={ret})"
            )

        # All attempts failed — fall back to PyTorch H2D.
        logger.warning(
            "[ExpertWeightStore] acc_offload init failed after retries, "
            "falling back to PyTorch H2D"
        )
        self.use_acc_offload = False

    def register_layer_batch(
        self,
        layer_id: int,
        weights_dict: Dict[str, torch.Tensor],
    ):
        """Batch-register all experts of a layer to DRAM in one pass.

        Processes each weight name once with a single large allocation
        and copy, then slices per-expert views into dram_store.

        Args:
            layer_id: Layer index
            weights_dict: Dict of {weight_name: full_tensor[num_experts, ...]}
        """
        self._ensure_initialize()
        num_experts = None
        total_bytes = 0
        temp_cpu_tensors = []

        use_pool = (
            self._use_pool_for_storage
            and self.use_acc_offload
            and self._offload_initialized
            and layer_id not in self._h2d_layer_ids
        )

        try:
            for name, full_tensor in weights_dict.items():
                if num_experts is None:
                    num_experts = full_tensor.shape[0]

                # Handle NPU→CPU conversion (NZ→ND + .cpu()) in one shot
                # for the entire [num_experts, ...] tensor.
                if full_tensor.device.type != "cpu":
                    nd_tensor = torch_npu.npu_format_cast(
                        full_tensor, NPUACLFormat.ACL_FORMAT_ND
                    ).contiguous()
                    cpu_tensor = nd_tensor.cpu()
                    del nd_tensor
                    temp_cpu_tensors.append(cpu_tensor)
                else:
                    cpu_tensor = full_tensor

                # Single large allocation + single copy for all experts
                if use_pool:
                    dram_tensor = self._offload.empty(
                        cpu_tensor.shape, dtype=cpu_tensor.dtype
                    )
                else:
                    dram_tensor = torch.empty(
                        cpu_tensor.shape, dtype=cpu_tensor.dtype, pin_memory=False
                    )
                dram_tensor.copy_(cpu_tensor)
                total_bytes += dram_tensor.nbytes

                # Slice per-expert views into dram_store
                for expert_id in range(num_experts):
                    key = (layer_id, expert_id)
                    if key not in self.dram_store:
                        self.dram_store[key] = {}
                    self.dram_store[key][name] = dram_tensor[expert_id]
        finally:
            del temp_cpu_tensors

        self._registered_layers.add(layer_id)

        logger.info(
            f"[ExpertWeightStore] D2H batch layer_id={layer_id}: "
            f"{num_experts} experts, {len(weights_dict)} weights, "
            f"{total_bytes / 1024**2:.1f} MB copied to DRAM"
        )

    def _release_cpu_cache(self):
        """Release CPU memory back to the OS after register_layer_batch() calls.

        PyTorch CPU tensors are allocated via glibc malloc (not PyTorch's
        CPU caching allocator unless PYTORCH_CPU_ALLOC_CONF is set).
        torch.cpu.empty_cache() only releases PyTorch's own caching
        allocator cache — it does NOT touch glibc malloc's arena.

        When a layer's CPU tensors (created by torch.empty(device="cpu")
        in loader.py Phase 3a, and by .transpose().contiguous() in
        process_weights_after_loading) are released via delattr +
        gc.collect(), glibc malloc holds the freed memory in its arena
        instead of returning it to the OS. This causes host DRAM usage
        to grow ~3.75 GB per layer (63 layers → ~236 GB) even after
        Python references are gone.

        Fix: call malloc_trim(0) to release glibc arenas back to the OS.
        Also call torch.cpu.empty_cache() for PyTorch CPU caching
        allocator. The MALLOC_TRIM_THRESHOLD_ environment variable can
        also help (set to 0 to make glibc return memory immediately).
        """
        import gc
        gc.collect()
        # Release PyTorch CPU caching allocator cache (no-op if disabled).
        try:
            torch.cpu.empty_cache()
        except (AttributeError, RuntimeError):
            pass
        # Release glibc malloc arenas back to the OS.
        # Critical: without this, host DRAM grows unbounded because glibc
        # holds freed memory in its arena (especially with multi-threaded
        # PyTorch which creates per-thread arenas).
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            # malloc_trim(0) releases free regions from all arenas.
            libc.malloc_trim(0)
        except Exception:
            pass
        # Debug: print arena stats (set SGLANG_DEBUG_MALLOC=1 to enable).
        try:
            if __import__("os").environ.get("SGLANG_DEBUG_MALLOC"):
                libc.malloc_stats()
        except Exception:
            pass

    def group_pack_copy_to_buffers(
        self,
        layer_id: int,
        weight_names: List[str],
        target_buffers: Dict[str, torch.Tensor],
    ) -> None:
        """Load ALL expert weights from DRAM into target HBM buffers via group_pack_copy.

        Prefill path: loads all local experts in order [0..N-1] without
        compaction. Uses a synthetic all-ones group_list so the kernel copies
        every expert. CANN later uses the real group_list from DeepEP dispatch
        (not the synthetic one), so packed_group_list is discarded.

        Args:
            layer_id: Layer index
            weight_names: List of weight parameter names
            target_buffers: {name: [num_local_experts, ...] HBM tensor} —
                            weights are written directly into these buffers

        Raises:
            RuntimeError: if group_pack_copy kernel returns a non-zero error.
        """
        self._ensure_initialize()

        num_local_experts = target_buffers[weight_names[0]].shape[0]
        target_device = target_buffers[weight_names[0]].device

        use_group_pack = (
            self.use_acc_offload
            and self._offload_initialized
            and (layer_id not in self._h2d_layer_ids)
        )

        if not use_group_pack:
            # H2D tail layers: weights are stored via torch.empty (not in
            # the acc_offload pool). Fall back to tensor.copy_() on the
            # current stream — serialized with preceding compute and
            # subsequent CANN ops on the same stream. non_blocking=True
            # enables async H2D when called within a stream context (e.g.
            # prefetch on _h2d_stream); for unpinned sources PyTorch
            # silently falls back to synchronous copy.
            for eid in range(num_local_experts):
                key = (layer_id, eid)
                if key not in self.dram_store:
                    continue
                dram_weights = self.dram_store[key]
                for name in weight_names:
                    target_buffers[name][eid].copy_(
                        dram_weights[name], non_blocking=True
                    )
            return

        # All-ones group_list: kernel copies ALL experts in order [0..N-1]
        # without compaction. packed_group_list is all-ones and discarded —
        # CANN uses the real group_list from DeepEP dispatch.
        group_list = torch.ones(
            num_local_experts, dtype=torch.int64, device=target_device
        )
        packed_group_list = torch.zeros(
            num_local_experts, dtype=torch.int64, device=target_device
        )
        device = torch.device(f"npu:{torch.npu.current_device()}")

        for name in weight_names:
            src_ptrs = []
            dst_ptrs = []
            len_ptrs = []
            for eid in range(num_local_experts):
                key = (layer_id, eid)
                if key not in self.dram_store:
                    src_ptrs.append(0)
                    dst_ptrs.append(target_buffers[name][eid].data_ptr())
                    len_ptrs.append(0)
                    continue
                dram_tensor = self.dram_store[key][name]
                src_ptrs.append(dram_tensor.data_ptr())
                dst_ptrs.append(target_buffers[name][eid].data_ptr())
                len_ptrs.append(dram_tensor.nbytes)

            # Guard against int32 overflow: kernel lens are uint32.
            max_len = max(len_ptrs) if len_ptrs else 0
            if max_len >= 2**31:
                msg = (
                    f"expert weight nbytes ({max_len}) exceeds int32 range, "
                    f"layer={layer_id} name={name}"
                )
                logger.error(msg)
                raise ValueError(msg)

            src_ptr_t = torch.tensor(src_ptrs, dtype=torch.int64, device=target_device)
            dst_ptr_t = torch.tensor(dst_ptrs, dtype=torch.int64, device=target_device)
            len_t = torch.tensor(len_ptrs, dtype=torch.int32, device=target_device)
            num_le_t = torch.tensor(num_local_experts, dtype=torch.int32, device=target_device)

            ret = self._offload.group_pack_copy(
                src_ptr_t, dst_ptr_t, len_t, num_le_t,
                group_list, packed_group_list, device,
            )
            if ret != 0:
                msg = (
                    f"[ExpertWeightStore] group_pack_copy failed ret={ret} "
                    f"layer={layer_id} name={name}"
                )
                logger.error(msg)
                raise RuntimeError(msg)

    def group_pack_copy_active_weights(
        self,
        layer_id: int,
        group_list: torch.Tensor,
        weight_names: List[str],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Build compact weight tensors via group_pack_copy kernel.

        Decode path: passes ALL local experts to group_pack_copy with the
        per-expert group_list; the NPU kernel copies only non-zero entries
        to the front of the output buffer and outputs packedGroupList
        (compacted group_list).

        Eliminates:
          - group_list.cpu() D2H sync
          - nonzero().squeeze(-1).tolist() host materialization
          - group_list_cpu[active_mask].to(device) H2D re-upload
          - sparse_copy odd-pair halving workaround

        Args:
            layer_id: Layer index
            group_list: [num_local_experts] int64 tensor on device,
                        per-expert token counts from DeepEP dispatch
            weight_names: List of weight parameter names

        Returns:
            (weights, packed_group_list):
              weights: {name: [num_local_experts, ...] tensor} — only
                       [0..M) slots are valid (M = non-zero group_list count)
              packed_group_list: [num_local_experts] int64 tensor —
                       first M entries are non-zero (compacted), rest are zero

        Raises:
            RuntimeError: if group_pack_copy kernel returns a non-zero error.
        """
        self._ensure_initialize()

        # Validate group_list properties to catch mismatches early.
        assert group_list.dim() == 1, (
            f"group_list must be 1-D, got shape {group_list.shape}"
        )
        assert group_list.dtype == torch.int64, (
            f"group_list must be int64, got {group_list.dtype}"
        )
        assert group_list.device.type == "npu", (
            f"group_list must be on NPU, got {group_list.device}"
        )

        num_local_experts = group_list.shape[0]
        target_device = group_list.device
        sample_key = (layer_id, 0)
        if sample_key not in self.dram_store:
            return {}, group_list

        # Pre-allocate [num_local_experts, ...] HBM buffers (reusable across
        # decode steps). The kernel writes compacted data to [0..M); the tail
        # [M..N) is stale but CANN skips it because packed_group_list[M..N)==0.
        # Validate shape on reuse to handle heterogeneous MoE layers safely.
        if not hasattr(self, "_shared_decode_buffers") or self._shared_decode_buffers is None:
            self._shared_decode_buffers = {}
        for name in weight_names:
            sample_tensor = self.dram_store[sample_key][name]
            full_shape = (num_local_experts,) + tuple(sample_tensor.shape)
            buf = self._shared_decode_buffers.get(name)
            if buf is None or tuple(buf.shape) != full_shape:
                buf = torch.empty(
                    full_shape, dtype=sample_tensor.dtype, device=target_device
                )
                self._shared_decode_buffers[name] = buf
        result = self._shared_decode_buffers

        use_group_pack = (
            self.use_acc_offload
            and self._offload_initialized
            and (layer_id not in self._h2d_layer_ids)
        )

        if not use_group_pack:
            # H2D tail layers: weights are stored via torch.empty (not in the
            # acc_offload pool), so group_pack_copy cannot be used. Fall back
            # to tensor.copy_() directly on the current (default) stream.
            # No cross-stream synchronization needed — copy_() is serialized
            # with preceding compute and subsequent CANN ops on the same
            # stream. Use original group_list (uncompacted); CANN skips zero
            # entries.
            for eid in range(num_local_experts):
                key = (layer_id, eid)
                if key not in self.dram_store:
                    continue
                dram_weights = self.dram_store[key]
                for name in weight_names:
                    result[name][eid].copy_(
                        dram_weights[name], non_blocking=True
                    )
            return result, group_list

        # Allocate packed_group_list output buffer (zero-filled so the tail
        # beyond M remains zero for CANN to skip). Small tensor (N int64
        # values, e.g. 2 KB for 256 experts), no need to cache across calls.
        packed_group_list = torch.zeros(
            num_local_experts, dtype=torch.int64, device=target_device
        )

        device = torch.device(f"npu:{torch.npu.current_device()}")

        # Call group_pack_copy once per weight name. Each call gets the same
        # group_list and packed_group_list (per-expert, not per-weight).
        for name in weight_names:
            src_ptrs = []
            dst_ptrs = []
            len_ptrs = []
            for eid in range(num_local_experts):
                key = (layer_id, eid)
                if key not in self.dram_store:
                    src_ptrs.append(0)
                    dst_ptrs.append(result[name][eid].data_ptr())
                    len_ptrs.append(0)
                    continue
                dram_tensor = self.dram_store[key][name]
                src_ptrs.append(dram_tensor.data_ptr())
                dst_ptrs.append(result[name][eid].data_ptr())
                len_ptrs.append(dram_tensor.nbytes)

            # Guard against int32 overflow: kernel lens are uint32, so any
            # single expert weight exceeding 2^31 bytes would wrap silently.
            max_len = max(len_ptrs) if len_ptrs else 0
            if max_len >= 2**31:
                msg = (
                    f"expert weight nbytes ({max_len}) exceeds int32 range, "
                    f"layer={layer_id} name={name}"
                )
                logger.error(msg)
                raise ValueError(msg)

            src_ptr_t = torch.tensor(src_ptrs, dtype=torch.int64, device=target_device)
            dst_ptr_t = torch.tensor(dst_ptrs, dtype=torch.int64, device=target_device)
            len_t = torch.tensor(len_ptrs, dtype=torch.int32, device=target_device)
            num_le_t = torch.tensor(num_local_experts, dtype=torch.int32, device=target_device)

            ret = self._offload.group_pack_copy(
                src_ptr_t, dst_ptr_t, len_t, num_le_t,
                group_list, packed_group_list, device,
            )
            if ret != 0:
                msg = (
                    f"[ExpertWeightStore] group_pack_copy failed ret={ret} "
                    f"layer={layer_id} name={name}"
                )
                logger.error(msg)
                raise RuntimeError(msg)

        # No explicit sync needed: group_pack_copy runs on the current (default)
        # NPU stream via c10_npu::getCurrentNPUStream, and subsequent CANN GMM
        # operations also run on the default stream. Stream ordering guarantees
        # the copy completes before GMM reads the buffer.
        return result, packed_group_list

    # ------------------------------------------------------------------
    # Prefill full-layer prefetch + cache mode management
    # ------------------------------------------------------------------ #

    def set_cache_mode(self, is_prefill: bool):
        """Toggle between prefill and decode mode.

        Sets _is_decode_mode which controls the weight loading path:
          - Prefill (is_prefill=True): _load_experts_on_demand /
            prefetch_layer_to_buffer uses group_pack_copy with a fake
            all-ones group_list to load ALL experts into
            [num_local_experts, ...] buffers
          - Decode (is_prefill=False): group_pack_copy_active_weights uses
            the real post-dispatch group_list to compact active expert
            weights on-device with no D2H sync
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
        h2d_stream (async, no sync). Uses group_pack_copy with an all-ones
        (fake) group_list — at prefetch time the real group_list from
        DeepEP dispatch is not yet available. After dispatch, CANN uses
        the real group_list to index into the loaded weights; the fake
        group_list is never passed to CANN.

        Caller must wait on the returned event (or call sync_prefetch()
        if the event is None) before using the buffers, and
        free_layer_buffers() after compute to release HBM.

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
        if sample_key not in self.dram_store:
            logger.warning(
                f"[ExpertWeightStore] prefetch_layer_to_buffer: "
                f"layer_id={layer_id} not in dram_store, skipping"
            )
            return {}, None
        weight_names = list(self.dram_store[sample_key].keys())

        buffers = {}
        for name in weight_names:
            sample_tensor = self.dram_store[sample_key][name]
            full_shape = (num_experts,) + sample_tensor.shape
            buffers[name] = torch.empty(
                full_shape, dtype=sample_tensor.dtype, device="npu"
            )

        event = None
        if self._h2d_stream is not None:
            event = torch.npu.Event()
            # Run group_pack_copy on _h2d_stream for async prefetch.
            # The stream context ensures group_pack_copy (which calls
            # c10_npu::getCurrentNPUStream internally) executes on
            # _h2d_stream, and the event captures its completion.
            with torch.npu.stream(self._h2d_stream):
                self.group_pack_copy_to_buffers(
                    layer_id=layer_id,
                    weight_names=weight_names,
                    target_buffers=buffers,
                )
                event.record()
        else:
            # No h2d_stream (CPU-only): synchronous copy, no event needed.
            self.group_pack_copy_to_buffers(
                layer_id=layer_id,
                weight_names=weight_names,
                target_buffers=buffers,
            )

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
        buffers.clear()

    def get_dram_usage_gb(self) -> float:
        """Get total DRAM usage in GB."""
        total = 0
        for weights in self.dram_store.values():
            total += sum(t.nbytes for t in weights.values())
        return total / 1024**3

    def release_hbm_weights(self):
        """Release HBM used during the registration process.

        Called after offload registration. Shared buffers are not used
        (per-forward allocation), so this is mostly gc + empty_cache.
        """
        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
