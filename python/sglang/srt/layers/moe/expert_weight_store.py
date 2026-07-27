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
        use_nz_storage: bool = True,
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

        # Statistics
        self._stats = {"dram_load": 0, "total_requests": 0}

        # Hybrid storage: layers in _h2d_layer_ids use PyTorch H2D
        # (torch.empty) instead of acc_offload pool. Configured via
        # --moe-dram-acc-offload-layers: only the first N offloaded
        # layers use the pool; the rest use H2D.
        self._h2d_layer_ids: set = set()  # layer_ids that use PyTorch H2D

        # NZ format storage: when True, w13_weight and w2_weight are stored
        # in NZ (FRACTAL_NZ) format in DRAM via sparse_copy, eliminating the
        # need for ND→NZ conversion at forward time. Only applies to
        # acc_offload pool layers (not H2D layers, which lack sparse_copy).
        # Scales are always stored in ND format (CANN operator expects ND
        # scales with transposed layout, not NZ).
        self._use_nz_storage = use_nz_storage
        self._nz_weight_names = {"w13_weight", "w2_weight"}
        # Per-layer per-weight original ND shape (needed for HBM allocation
        # since the DRAM buffer for NZ weights is a flat byte buffer).
        self._nz_weight_shapes: Dict[Tuple[int, str], torch.Size] = {}

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

    # ------------------------------------------------------------------ #
    # NZ format storage helpers
    # ------------------------------------------------------------------ #

    def use_nz_storage_for_layer(self, layer_id: int) -> bool:
        """Check if a layer uses NZ format storage for weights in DRAM.

        NZ storage requires:
          1. _use_nz_storage flag enabled
          2. acc_offload backend available (sparse_copy is the only API
             that can transfer NZ-format bytes; torch copy_() fails on
             internal format tensors)
          3. Layer is NOT in _h2d_layer_ids (H2D layers use torch.empty
             + copy_(), which cannot handle NZ format)
        """
        return (
            self._use_nz_storage
            and self.use_acc_offload
            and self._offload_initialized
            and layer_id not in self._h2d_layer_ids
        )

    def is_nz_weight(self, layer_id: int, weight_name: str) -> bool:
        """Check if a specific weight is stored in NZ format in DRAM."""
        if not self.use_nz_storage_for_layer(layer_id):
            return False
        return weight_name in self._nz_weight_names

    def _cast_to_nz(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast an ND uint8 tensor to NZ format (FRACTAL_NZ, format=29).

        Used for W4A8 MXFP4 weights where 2 fp4 items are packed in a
        uint8 byte. The customize_dtype/input_dtype parameters tell the
        NPU runtime how to interpret the packed sub-byte layout during
        the NZ block reorganization.
        """
        return torch_npu.npu_format_cast(
            tensor, 29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )

    def _sparse_copy_npu_to_dram(
        self,
        src_npu_tensor: torch.Tensor,
        dst_dram_tensor: torch.Tensor,
    ) -> None:
        """Copy NZ-format bytes from NPU tensor to DRAM buffer via sparse_copy.

        torch copy_() cannot handle NZ (internal format) tensors — it
        raises "do not support internal format". sparse_copy operates on
        raw bytes via an AIV kernel, preserving the NZ block layout
        exactly (verified by test_offload.py Scenario 1).

        sparse_copy requires even num_pairs. A single pair (1 src, 1 dst)
        is odd, so we split into 2 halves to satisfy the constraint.
        """
        nbytes = src_npu_tensor.element_size() * src_npu_tensor.numel()
        assert nbytes % 2 == 0, f"NZ storage size must be even, got {nbytes}"
        half = nbytes // 2

        src_ptrs = [src_npu_tensor.data_ptr(), src_npu_tensor.data_ptr() + half]
        dst_ptrs = [dst_dram_tensor.data_ptr(), dst_dram_tensor.data_ptr() + half]
        len_ptrs = [half, half]

        src_ptr_t = torch.tensor(src_ptrs, dtype=torch.int64, device="npu")
        dst_ptr_t = torch.tensor(dst_ptrs, dtype=torch.int64, device="npu")
        len_t = torch.tensor(len_ptrs, dtype=torch.int32, device="npu")
        size_t = torch.tensor(2, dtype=torch.int32, device="npu")

        device = torch.device(f"npu:{torch.npu.current_device()}")
        ret = self._offload.sparse_copy(src_ptr_t, dst_ptr_t, len_t, size_t, device)
        if ret != 0:
            raise RuntimeError(
                f"sparse_copy D2H (NZ→DRAM) failed: ret={ret}, "
                f"nbytes={nbytes}"
            )
        torch.npu.synchronize()

    def _allocate_nz_hbm_buffer(
        self,
        num_experts: int,
        layer_id: int,
        weight_name: str,
        dtype: torch.dtype,
        device: str,
    ) -> torch.Tensor:
        """Allocate an HBM buffer in NZ format for NZ-stored weights.

        Creates an ND tensor with the original per-expert shape, then
        casts to NZ format. The NZ bytes will be overwritten by
        sparse_copy from DRAM, so the initial content is irrelevant.

        For non-NZ weights, use torch.empty() directly (ND format).
        """
        orig_shape = self._nz_weight_shapes.get((layer_id, weight_name))
        if orig_shape is None:
            # Fallback: use DRAM buffer's shape (ND weights)
            sample_key = (layer_id, 0)
            orig_shape = self.dram_store[sample_key][weight_name].shape

        full_shape = (num_experts,) + tuple(orig_shape)
        nd_tensor = torch.empty(full_shape, dtype=dtype, device=device)
        nz_tensor = self._cast_to_nz(nd_tensor)
        # del nd_tensor is safe: nz_tensor shares or owns the storage.
        # Keeping a Python reference to nd_tensor is unnecessary.
        del nd_tensor
        return nz_tensor

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

        Before each HalMemCreate attempt, we drop kernel page cache to free
        up contiguous physical memory. Huge pages require physically
        contiguous 2MB regions, and excessive file cache (from safetensors
        mmap during weight loading) can cause allocation failure even when
        MemAvailable looks sufficient.

        We try up to 2 times, dropping cache before each attempt.
        """
        config = offload.OffloadConfig()
        config.device_id = torch.npu.current_device()
        config.size = self._dram_pool_size_bytes

        for attempt in range(2):
            # Drop page cache before each attempt to maximize contiguous
            # physical memory available for huge page allocation.
            logger.info(
                f"[ExpertWeightStore] acc_offload init attempt {attempt + 1}/2, "
                f"dropping page cache before HalMemCreate..."
            )
            _drop_kernel_page_cache()

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

    def register_expert(
        self,
        layer_id: int,
        expert_id: int,
        weights: Dict[str, torch.Tensor],
    ):
        """Register expert weights from HBM to Host DRAM.

        Called after process_weights_after_loading(). Copies the processed
        (NZ-format, packed) weights from HBM to Host DRAM.

        For NZ-storage layers (acc_offload pool), w13_weight and w2_weight
        are converted to NZ format on NPU and stored via sparse_copy,
        eliminating ND→NZ conversion at forward time. Scales and H2D-layer
        weights use the ND path (torch copy_()).

        Args:
            layer_id: Layer index
            expert_id: Expert index within the layer
            weights: Dict of {weight_name: hbm_tensor} e.g.
                     {"w13_weight": ..., "w2_weight": ...,
                      "w13_weight_scale": ..., "w2_weight_scale": ...}
        """
        self._ensure_initialize()
        key = (layer_id, expert_id)

        # Determine if this layer uses NZ storage for weights.
        use_nz = self.use_nz_storage_for_layer(layer_id)

        cpu_weights = {}
        total_bytes = 0
        # Track temporary CPU/NPU tensors so we can release them explicitly
        # after copying to the DRAM pool. Without this, PyTorch caching
        # allocators hold the memory and do not return it to the OS, causing
        # host DRAM / HBM usage to grow unbounded.
        #
        # Note: We do NOT call torch.cpu.empty_cache() here because it would
        # be invoked 896 times per layer (once per expert), causing significant
        # overhead. The caller (offload_expert_weights_to_dram) is responsible
        # for calling _release_cpu_cache() once after all experts are registered.
        temp_cpu_tensors = []
        temp_npu_tensors = []
        try:
            for name, tensor in weights.items():
                is_nz_weight = use_nz and name in self._nz_weight_names

                if is_nz_weight:
                    # NZ storage path: convert ND→NZ on NPU, sparse_copy to DRAM.
                    #
                    # tensor is [K_packed, N] (from process_weights_after_loading
                    # which does .transpose(1, 2).contiguous()). We need to:
                    #   1. Move to NPU
                    #   2. Transpose to [N, K_packed] (the layout NZ conversion
                    #      expects, matching the non-offload path in
                    #      process_weights_after_loading)
                    #   3. Cast to NZ format
                    #   4. sparse_copy NZ bytes to DRAM buffer
                    #
                    # At forward time, the NZ bytes are loaded into [E, N, K] NZ
                    # HBM tensor, then .transpose(1,2) gives [E, K, N] NZ for GMM.
                    if tensor.device.type != "cpu":
                        # Already on NPU (e.g., non-offload path) — shouldn't
                        # happen for DRAM offload, but handle gracefully.
                        npu_nd = tensor
                    else:
                        npu_nd = tensor.npu()
                        temp_npu_tensors.append(npu_nd)

                    # Transpose [K, N] → [N, K] and make contiguous
                    npu_nd_t = npu_nd.transpose(0, 1).contiguous()
                    temp_npu_tensors.append(npu_nd_t)

                    # Store original shape for HBM allocation at forward time
                    self._nz_weight_shapes[(layer_id, name)] = npu_nd_t.shape

                    # Cast to NZ format
                    nz_tensor = self._cast_to_nz(npu_nd_t)
                    temp_npu_tensors.append(nz_tensor)

                    storage_size = (
                        nz_tensor.element_size() * nz_tensor.numel()
                    )

                    # Allocate DRAM buffer (flat byte buffer)
                    dram_tensor = self._offload.empty(
                        [storage_size], dtype=torch.uint8
                    )

                    # sparse_copy NZ bytes from NPU to DRAM
                    self._sparse_copy_npu_to_dram(nz_tensor, dram_tensor)

                    cpu_weights[name] = dram_tensor
                    total_bytes += storage_size

                    # Free NPU tensors to release HBM
                    del nz_tensor, npu_nd_t, npu_nd
                    temp_npu_tensors.clear()
                else:
                    # ND storage path (scales, H2D layers, or NZ disabled):
                    # use torch copy_() to copy ND tensor to DRAM.
                    #
                    # NPU internal format (e.g., FRACTAL_NZ) cannot be copied
                    # via copy_() or .cpu() -- NPU raises "do not support
                    # internal format". npu_format_cast to ND may only change
                    # metadata without reformatting storage, so .contiguous()
                    # forces a real ND copy.
                    if tensor.device.type != "cpu":
                        nd_tensor = torch_npu.npu_format_cast(
                            tensor, NPUACLFormat.ACL_FORMAT_ND
                        ).contiguous()
                        cpu_tensor = nd_tensor.cpu()
                        del nd_tensor
                        temp_cpu_tensors.append(cpu_tensor)
                    else:
                        cpu_tensor = tensor

                    use_pool = (
                        self._use_pool_for_storage
                        and self.use_acc_offload
                        and self._offload_initialized
                        and layer_id not in self._h2d_layer_ids
                    )
                    if use_pool:
                        # Allocate from acc_offload DRAM pool.
                        dram_tensor = self._offload.empty(
                            cpu_tensor.shape, dtype=cpu_tensor.dtype
                        )
                    else:
                        # H2D tail layer or pool unavailable: PyTorch torch.empty.
                        dram_tensor = torch.empty(
                            cpu_tensor.shape, dtype=cpu_tensor.dtype, pin_memory=False
                        )
                    dram_tensor.copy_(cpu_tensor)
                    cpu_weights[name] = dram_tensor
                    total_bytes += dram_tensor.nbytes
        finally:
            # Release temporary tensor Python references immediately.
            # PyTorch caching allocators may still hold the underlying
            # memory; caller must invoke _release_cpu_cache() after the full
            # layer registration loop to return it to the OS.
            del temp_cpu_tensors
            del temp_npu_tensors

        self.dram_store[key] = cpu_weights
        self._registered_layers.add(layer_id)

        if expert_id % 64 == 0:
            logger.info(
                f"[ExpertWeightStore] D2H layer_id={layer_id} expert_id={expert_id}: "
                f"{len(cpu_weights)} tensors, {total_bytes / 1024**2:.1f} MB copied to DRAM"
            )

    def _release_cpu_cache(self):
        """Release CPU memory back to the OS after register_expert() calls.

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

    def _release_layer_cpu_tensors(self, layer_id: int):
        """Force-release all CPU tensors associated with a layer.

        Called after offload_expert_weights_to_dram() to release:
          1. Parameter references (via delattr in offload_expert_weights_to_dram)
          2. layer_ws references (via del in loader.py)
          3. glibc malloc arenas (via malloc_trim)
          4. PyTorch CPU caching allocator (via empty_cache)

        This is a more aggressive release than _release_cpu_cache alone,
        intended to be called once per layer after all references are gone.
        """
        self._release_cpu_cache()

    def _batch_h2d_copy(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        sync: bool = True,
        layer_id: Optional[int] = None,
        wait_for_compute: bool = False,
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
            wait_for_compute: If True, make _h2d_stream wait for the
                  current (compute) stream before starting H2D copy.
                  Required when writing to _shared_hbm_buffers, which
                  the previous forward's compute may still be reading.
                  Set True for batch_load_to_hbm (shared buffer reuse),
                  False for prefetch_layer_to_buffer (separate buffers).
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
        #
        # Race condition fix: when writing to _shared_hbm_buffers (reused
        # across forwards), the previous forward's compute on the default
        # stream may still be reading from the buffer. Without a stream
        # dependency, the H2D copy on _h2d_stream can overlap with that
        # compute, corrupting the data and causing precision degradation.
        # acc_offload doesn't have this issue because sparse_copy runs on
        # the default stream and torch.npu.synchronize() syncs all streams.
        #
        # Fix: record an event on the compute stream and make _h2d_stream
        # wait for it before starting the copy. This serializes H2D with
        # the previous compute, eliminating the race. Performance impact
        # is minimal because batch_load_to_hbm is already synchronous.
        if wait_for_compute and self._h2d_stream is not None:
            compute_event = torch.npu.Event()
            compute_event.record()  # Record on current (compute) stream
            with torch.npu.stream(self._h2d_stream):
                compute_event.wait()  # _h2d_stream waits for compute
                for src, dst in pairs:
                    dst.copy_(src, non_blocking=True)
        else:
            with torch.npu.stream(self._h2d_stream):
                for src, dst in pairs:
                    dst.copy_(src, non_blocking=True)
        if sync:
            self._h2d_stream.synchronize()

    def batch_load_to_hbm(
        self,
        layer_id: int,
        expert_ids: List[int],
        shared_buffers: Dict[str, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load expert weights from DRAM to HBM buffers.

        Writes weights directly into the provided HBM buffers indexed by
        expert_id, avoiding an extra HBM→HBM copy.

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

        self._batch_h2d_copy(pairs, sync=True, layer_id=layer_id,
                             wait_for_compute=True)

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

        For NZ-storage layers, w13_weight and w2_weight HBM buffers are
        allocated in NZ format and filled via sparse_copy from DRAM. This
        eliminates the ND→NZ conversion at forward time (in
        w4a8_mxfp4_gmm_npu), reducing HBM peak and compute latency.

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

        # Allocate compact [num_active, ...] HBM buffers.
        # For NZ weights: allocate ND tensor, cast to NZ format.
        # For ND weights (scales): allocate ND tensor directly.
        result = {}
        for name in weight_names:
            sample_tensor = self.dram_store[sample_key][name]
            if self.is_nz_weight(layer_id, name):
                # NZ weight: use _allocate_nz_hbm_buffer which reads
                # the original shape from _nz_weight_shapes (since the
                # DRAM buffer is a flat [storage_size] byte buffer).
                result[name] = self._allocate_nz_hbm_buffer(
                    num_experts=num_active,
                    layer_id=layer_id,
                    weight_name=name,
                    dtype=sample_tensor.dtype,
                    device="npu",
                )
            else:
                # ND weight (scale): use DRAM buffer's shape directly.
                full_shape = (num_active,) + sample_tensor.shape
                result[name] = torch.empty(
                    full_shape, dtype=sample_tensor.dtype, device="npu"
                )

        # Build (src_dram, dst_hbm) pairs for batch sparse_copy.
        # For NZ weights, dst is a view into the NZ tensor (result[name][i]).
        # sparse_copy operates on raw bytes, so it correctly copies NZ bytes
        # from the flat DRAM buffer to the NZ-format HBM tensor view.
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

        # Allocate [num_experts, ...] HBM buffers.
        # For NZ weights: allocate ND tensor, cast to NZ format.
        # For ND weights (scales): allocate ND tensor directly.
        buffers = {}
        for name in weight_names:
            sample_tensor = self.dram_store[sample_key][name]
            if self.is_nz_weight(layer_id, name):
                buffers[name] = self._allocate_nz_hbm_buffer(
                    num_experts=num_experts,
                    layer_id=layer_id,
                    weight_name=name,
                    dtype=sample_tensor.dtype,
                    device="npu",
                )
            else:
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
        """Release HBM used during the registration process.

        Called after offload registration. Shared buffers are not used
        (per-forward allocation), so this is mostly gc + empty_cache.
        """
        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
