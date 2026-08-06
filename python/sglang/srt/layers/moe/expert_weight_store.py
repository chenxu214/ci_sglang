# SPDX-License-Identifier: Apache-2.0
"""Expert weight store for MoE DRAM offloading.

Manages MoE expert weights in Host DRAM. During forward, only the
Top-K selected experts are loaded from Host DRAM to HBM on demand.

Two backends are supported:
  1. acc_offload (default when available): Uses MemFabric acc_offload
     group_pack_copy kernel with MTE engine for batch H2D copy.
     Higher performance due to 32-core parallelism and reduced API overhead.
  2. PyTorch H2D (fallback): Uses sgl_kernel_npu transfer_weight kernel
     (aclrtMemcpyAsync) for D2H/H2D. No acc_offload dependency, works
     everywhere. Supports NPU graph capture.

Weight loading paths:
  - Prefill: group_pack_copy_to_buffers() loads ALL local experts into
    shared HBM buffers synchronously per layer. Uses a fake all-ones
    group_list (real group_list not available pre-dispatch); CANN later
    uses the real group_list from DeepEP dispatch.
  - Decode: group_pack_copy_active_weights() uses the real post-dispatch
    group_list to load and compact active expert weights on-device,
    outputting a packed group_list for CANN. No D2H sync required.

NZ format storage:
  Weight tensors (w13_weight, w2_weight) are stored in NZ format in DRAM.
  Both acc_offload (sparse_copy) and PyTorch H2D (transfer_weight) paths
  preserve NZ byte layout, eliminating forward-time ND→NZ conversion.
  Scale tensors remain ND format (CANN's antiquant_scale expects ND).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

logger = logging.getLogger(__name__)

# transfer_weight kernel from sgl_kernel_npu: uses aclrtMemcpyAsync for
# efficient H2D/D2H byte copy. Layout-agnostic (preserves NZ bytes),
# async (supports graph capture), and faster than torch.copy_.
try:
    from sgl_kernel_npu.kvcacheio import transfer_weight, TransferDirection
    _TRANSFER_WEIGHT_AVAILABLE = True
except ImportError:
    _TRANSFER_WEIGHT_AVAILABLE = False
    transfer_weight = None
    TransferDirection = None


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
        use_acc_offload: Whether to use acc_offload group_pack_copy
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
        #   - Prefill: group_pack_copy_to_buffers (synchronous, all experts)
        #   - Decode: group_pack_copy_active_weights (on-device compaction)
        self._is_decode_mode = False

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

        # pin_memory control via environment variable.
        # MOE_DRAM_PIN_MEMORY=1 (default): use pinned host memory for DRAM
        #   tensors, enabling async H2D via DMA engine (~30% faster).
        # MOE_DRAM_PIN_MEMORY=0: use regular host memory (for limited DRAM).
        import os
        pin_env = os.environ.get("MOE_DRAM_PIN_MEMORY", "1").lower()
        self._pin_memory = pin_env not in ("0", "false", "no", "off")

        # Track registered layers for warmup
        self._registered_layers: set = set()

        # Graph-safe caches for decode path (group_pack_copy_active_weights).
        # These tensors are fixed across decode steps (DRAM pool addresses,
        # shared buffer addresses, and weight sizes don't change), so they
        # are computed once before graph capture and reused during replay —
        # avoiding H2D transfers from torch.tensor(python_list, device="npu")
        # which crash NPU graph capture.
        # Key: (layer_id, name) → (src_ptr_t, dst_ptr_t, len_t)
        self._decode_ptr_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._num_le_t: Optional[torch.Tensor] = None
        self._num_le_cache: Optional[int] = None
        self._packed_group_list_buf: Optional[torch.Tensor] = None
        # Padded input group_list buffer for acc_offload path: when DeepEP
        # returns a shorter group_list (size < weight dim 0), pad with zeros
        # so group_pack_copy kernel can safely iterate [0, num_le).
        self._padded_group_list_buf: Optional[torch.Tensor] = None

        # Hybrid storage: layers in _h2d_layer_ids use PyTorch H2D
        # (torch.empty) instead of acc_offload pool. Configured via
        # --moe-dram-acc-offload-layers: only the first N offloaded
        # layers use the pool; the rest use H2D.
        self._h2d_layer_ids: set = set()  # layer_ids that use PyTorch H2D

        # Store original weight logical shapes for NZ storage path.
        # NZ-format DRAM buffers are flat byte arrays whose .shape doesn't
        # reflect the original [num_experts, N, K] layout needed for HBM
        # buffer allocation. Key: (layer_id, name) → original full_shape.
        self._weight_shapes: Dict[Tuple[int, str], Tuple[int, ...]] = {}

        # Full-layer DRAM tensor references for batch H2D copy.
        # Key: (layer_id, name) → full-layer DRAM tensor (all experts).
        self._dram_layer_tensors: Dict[Tuple[int, str], torch.Tensor] = {}

        # Global shared HBM buffers (all offloaded layers reuse these).
        # Prefill layers are processed sequentially, so only one layer's
        # weights need to reside in HBM at a time. Allocating once at init
        # eliminates per-layer npu_format_cast during prefill, which was
        # the primary cause of HBM OOM when 58 layers' buffers accumulated.
        #
        # _global_buffer_template stores the first-registered layer's
        # shape/dtype/NZ metadata so subsequent layers can verify compatibility.
        # _shared_global_buffers: {name: [num_local_experts, ...] NZ tensor}
        self._shared_global_buffers: Optional[Dict[str, torch.Tensor]] = None
        self._global_buffer_template: Optional[Dict[str, Tuple[Tuple[int, ...], torch.dtype, bool]]] = None

        # Cache for prefill ptr_t tensors (reused across layers).
        # group_pack_copy_to_buffers builds src_ptr_t/dst_ptr_t/len_t each
        # call; caching them avoids H2D tensor allocations inside graph capture.
        # Key: (layer_id, name) → (src_ptr_t, dst_ptr_t, len_t, num_le_t)
        self._prefill_ptr_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._prefill_num_le_t: Optional[torch.Tensor] = None
        self._prefill_num_le_cache: Optional[int] = None
        self._prefill_group_list_buf: Optional[torch.Tensor] = None
        self._prefill_packed_group_list_buf: Optional[torch.Tensor] = None

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
        else:
            # No acc_offload layers — use PyTorch H2D for all offloaded layers.
            self.use_acc_offload = False
            logger.info(
                "[ExpertWeightStore] acc_offload disabled (layers=0), "
                "using PyTorch H2D for all offloaded layers"
            )

    def _ensure_initialize(self):
        if not self._initialized:
            if torch.npu.is_available():
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

        logger.info(
            f"[ExpertWeightStore] acc_offload init attempt 1/1 "
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
                f"(attempt=1)"
            )
            return

        # Init failed — fall back to PyTorch H2D immediately.
        logger.warning(
            f"[ExpertWeightStore] acc_offload init failed (ret={ret}), "
            "falling back to PyTorch H2D"
        )
        self.use_acc_offload = False

    def _is_nz_storage(self, layer_id: int) -> bool:
        """Check if NZ format direct storage is active for this layer.

        NZ storage is always active for weight tensors, regardless of
        whether acc_offload is used. This eliminates forward-time ND→NZ
        conversion:
        - acc_offload path: sparse_copy D2H, group_pack_copy H2D
        - PyTorch H2D path: transfer_weight kernel D2H and H2D
        """
        return True

    def _is_pool_storage(self, layer_id: int) -> bool:
        """Check if acc_offload pool is used for DRAM storage."""
        return (
            self._use_pool_for_storage
            and self.use_acc_offload
            and self._offload_initialized
            and layer_id not in self._h2d_layer_ids
        )

    def _ensure_shared_global_buffers(
        self,
        layer_id: int,
        num_local_experts: int,
        weight_names: List[str],
        sample_key: Tuple[int, int],
        target_device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Allocate (once) and return global shared NZ HBM buffers.

        All offloaded layers share one set of HBM buffers because prefill
        processes layers sequentially — only one layer's weights need to
        be resident at a time. This eliminates 58 × ~60 MB = 3.5 GB of
        per-layer _shared_hbm_buffers accumulation that caused HBM OOM.

        The first registered layer determines the buffer shape/dtype; all
        subsequent layers must match (same model architecture guarantee).

        Weight tensors (uint8) are allocated in NZ format + transpose(1,2)
        to match the HBM-resident path exactly. Scale tensors remain ND.

        Graph-safe: allocation happens once before graph capture (warmup),
        so buffer addresses are stable across replay.
        """
        if self._shared_global_buffers is not None:
            return self._shared_global_buffers

        self._shared_global_buffers = {}
        self._global_buffer_template = {}
        use_nz = self._is_nz_storage(layer_id)

        for name in weight_names:
            sample_tensor = self.dram_store[sample_key][name]
            dtype = sample_tensor.dtype
            shape_key = (layer_id, name)
            if shape_key in self._weight_shapes:
                full_shape = self._weight_shapes[shape_key]
            else:
                full_shape = (num_local_experts,) + tuple(sample_tensor.shape)

            buf = torch.empty(full_shape, dtype=dtype, device=target_device)
            is_nz_weight = (
                use_nz
                and dtype == torch.uint8
                and "scale" not in name
                and shape_key in self._weight_shapes
            )
            if is_nz_weight:
                buf = torch_npu.npu_format_cast(
                    buf, 29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                buf = buf.transpose(1, 2)

            self._shared_global_buffers[name] = buf
            self._global_buffer_template[name] = (
                tuple(full_shape), dtype, is_nz_weight
            )

        total_bytes = sum(
            t.element_size() * t.numel()
            for t in self._shared_global_buffers.values()
        )
        logger.info(
            f"[ExpertWeightStore] Pre-allocated global shared HBM buffers: "
            f"{total_bytes / 1024**3:.3f} GB for {len(weight_names)} tensors "
            f"(NZ-weights={use_nz}). All offloaded layers reuse these."
        )
        # Invalidate decode ptr cache: dst addresses changed (from legacy
        # _shared_decode_buffers per-layer buffers to the new global buffers).
        # Without this, the first decode forward after a warmup prefill would
        # use stale dst pointers pointing to freed/never-allocated memory.
        self._decode_ptr_cache.clear()
        return self._shared_global_buffers

    def _sparse_copy_d2h(self, src_npu_tensor: torch.Tensor, dst_dram_tensor: torch.Tensor):
        """Copy NZ-format NPU tensor to DRAM pool via sparse_copy.

        sparse_copy is the only API that can transfer NZ-format data
        (torch copy_()/.cpu() fail with "do not support internal format").
        The kernel operates on raw bytes, preserving the NZ block layout.

        Requires even num_pairs; split into 2 halves for odd-sized tensors.
        """
        storage_size = src_npu_tensor.element_size() * src_npu_tensor.numel()
        half = storage_size // 2
        src_ptrs = [src_npu_tensor.data_ptr(), src_npu_tensor.data_ptr() + half]
        dst_ptrs = [dst_dram_tensor.data_ptr(), dst_dram_tensor.data_ptr() + half]
        len_ptrs = [half, half]

        target_device = torch.device(f"npu:{torch.npu.current_device()}")
        src_ptr_t = torch.tensor(src_ptrs, dtype=torch.int64, device=target_device)
        dst_ptr_t = torch.tensor(dst_ptrs, dtype=torch.int64, device=target_device)
        len_t = torch.tensor(len_ptrs, dtype=torch.int32, device=target_device)
        size_t = torch.tensor(2, dtype=torch.int32, device=target_device)

        ret = self._offload.sparse_copy(src_ptr_t, dst_ptr_t, len_t, size_t, target_device)
        if ret != 0:
            raise RuntimeError(f"sparse_copy D2H failed ret={ret}")
        torch.npu.synchronize()

    def register_layer_batch(
        self,
        layer_id: int,
        weights_dict: Dict[str, torch.Tensor],
    ):
        """Batch-register all experts of a layer to DRAM in one pass.

        For acc_offload pool layers: weight tensors (uint8) are stored in
        NZ format via sparse_copy (D2H), eliminating forward-time ND→NZ
        conversion. Scale tensors are stored in ND format via copy_.

        For H2D tail layers: all tensors stored in ND format via copy_.

        Args:
            layer_id: Layer index
            weights_dict: Dict of {weight_name: full_tensor[num_experts, ...]}
        """
        self._ensure_initialize()
        num_experts = None
        total_bytes = 0
        temp_cpu_tensors = []

        use_pool = self._is_pool_storage(layer_id)

        import time
        t_layer_start = time.time()

        try:
            for name, full_tensor in weights_dict.items():
                if num_experts is None:
                    num_experts = full_tensor.shape[0]

                # NZ storage only for weight tensors (w13_weight, w2_weight),
                # NOT for scales — CANN's antiquant_scale expects ND format.
                # Both are uint8, so distinguish by name.
                is_weight = full_tensor.dtype == torch.uint8 and "scale" not in name

                if use_pool and is_weight:
                    # acc_offload NZ storage path: sparse_copy D2H
                    # NZ cast on original [E, N, K_packed] shape (matching
                    # non-offload path), then transpose NZ result (free view).
                    # Avoids .transpose(1,2).contiguous() on CPU (~22s/4.7GB).
                    t0 = time.time()
                    npu_tensor = full_tensor.npu() if full_tensor.device.type == "cpu" else full_tensor
                    nz_tensor = torch_npu.npu_format_cast(
                        npu_tensor, 29,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=torch_npu.float4_e2m1fn_x2,
                    )
                    nz_tensor = nz_tensor.transpose(1, 2)
                    storage_size = nz_tensor.element_size() * nz_tensor.numel()
                    per_expert_nz_nbytes = nz_tensor[0].nbytes
                    t1 = time.time()

                    dram_flat = self._offload.empty(
                        [storage_size], dtype=torch.uint8
                    )
                    self._sparse_copy_d2h(nz_tensor, dram_flat)
                    dram_tensor = dram_flat.view(num_experts, per_expert_nz_nbytes)
                    del npu_tensor, nz_tensor, dram_flat
                    total_bytes += storage_size

                    # Store ORIGINAL shape [E, N, K_packed] (not transposed)
                    # so HBM buffer allocation casts NZ on the same shape as
                    # the DRAM data. npu_format_cast with input_dtype=
                    # float4_e2m1fn_x2 unpacks the LAST dim as packed fp4,
                    # so the last dim MUST be K_packed (packed), not N.
                    # Storing transposed shape would cause CANN to unpack N
                    # instead of K_packed, producing wrong NZ block metadata
                    # (weight K=2*N instead of K=2*K_packed=hidden).
                    self._weight_shapes[(layer_id, name)] = tuple(full_tensor.shape)
                    logger.info(
                        f"[D2H timing] layer={layer_id} name={name} "
                        f"nz_cast={t1-t0:.2f}s sparse_copy+d2h={time.time()-t1:.2f}s "
                        f"size={storage_size/1024**2:.1f}MB"
                    )
                elif is_weight and _TRANSFER_WEIGHT_AVAILABLE:
                    # PyTorch H2D NZ storage path: transfer_weight D2H
                    # Uses aclrtMemcpyAsync to copy NZ bytes from HBM to DRAM,
                    # preserving NZ layout without ND→NZ conversion at forward.
                    #
                    # NZ cast on original [E, N, K_packed] shape (matching the
                    # non-offload path in process_weights_after_loading), then
                    # transpose the NZ result (free view on NZ tensor). This
                    # avoids the expensive .transpose(1,2).contiguous() on CPU
                    # which costs ~22s per 4.7GB weight due to non-sequential
                    # memory access. The .npu() H2D copy is ~0.2s by comparison.
                    t0 = time.time()
                    npu_tensor = full_tensor.npu() if full_tensor.device.type == "cpu" else full_tensor
                    t1 = time.time()
                    nz_tensor = torch_npu.npu_format_cast(
                        npu_tensor, 29,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=torch_npu.float4_e2m1fn_x2,
                    )
                    nz_tensor = nz_tensor.transpose(1, 2)
                    t2 = time.time()
                    storage_size = nz_tensor.element_size() * nz_tensor.numel()
                    per_expert_nz_nbytes = nz_tensor[0].nbytes

                    dram_tensor = torch.empty(
                        [storage_size], dtype=torch.uint8,
                        pin_memory=self._pin_memory,
                    ).view(num_experts, per_expert_nz_nbytes)
                    t3 = time.time()
                    transfer_weight(dram_tensor, nz_tensor, direction=TransferDirection.D2H)
                    torch.npu.synchronize()  # ensure D2H complete before releasing NZ tensor
                    t4 = time.time()
                    del npu_tensor, nz_tensor
                    t5 = time.time()
                    total_bytes += storage_size

                    # Store ORIGINAL shape [E, N, K_packed] (not transposed)
                    # so HBM buffer allocation casts NZ on the same shape as
                    # the DRAM data. npu_format_cast with input_dtype=
                    # float4_e2m1fn_x2 unpacks the LAST dim as packed fp4,
                    # so the last dim MUST be K_packed (packed), not N.
                    # Storing transposed shape would cause CANN to unpack N
                    # instead of K_packed, producing wrong NZ block metadata
                    # (weight K=2*N instead of K=2*K_packed=hidden).
                    self._weight_shapes[(layer_id, name)] = tuple(full_tensor.shape)
                    logger.info(
                        f"[D2H timing] layer={layer_id} name={name} "
                        f"h2d={t1-t0:.2f}s nz_cast={t2-t1:.2f}s "
                        f"pin_alloc={t3-t2:.2f}s d2h={t4-t3:.2f}s "
                        f"del={t5-t4:.2f}s size={storage_size/1024**2:.1f}MB"
                    )
                else:
                    # ND storage path (scales and H2D tail weights):
                    # Undo transpose before storing so DRAM has pre-transpose
                    # shape. Forward path (w4a8_mxfp4_gmm_npu) re-applies
                    # transpose at forward time to match HBM-resident state.
                    pre_transpose = full_tensor.transpose(1, 2).contiguous()
                    if pre_transpose.device.type != "cpu":
                        cpu_tensor = pre_transpose.cpu()
                    else:
                        cpu_tensor = pre_transpose
                    temp_cpu_tensors.append(cpu_tensor)

                    if use_pool:
                        dram_tensor = self._offload.empty(
                            cpu_tensor.shape, dtype=cpu_tensor.dtype
                        )
                    else:
                        # Scale tensors (ND format) are also copied via
                        # transfer_weight at forward time. aclrtMemcpyAsync
                        # requires pinned host memory as src to avoid an
                        # internal StreamSynchronize, which is not supported
                        # during NPU graph capture (error 107027
                        # "stream is captured"). Use self._pin_memory so
                        # graph mode works when MOE_DRAM_PIN_MEMORY=1.
                        dram_tensor = torch.empty(
                            cpu_tensor.shape, dtype=cpu_tensor.dtype,
                            pin_memory=self._pin_memory,
                        )
                    dram_tensor.copy_(cpu_tensor)
                    total_bytes += dram_tensor.nbytes

                # Save full-layer DRAM tensor reference for batch H2D copy.
                self._dram_layer_tensors[(layer_id, name)] = dram_tensor

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
            f"{total_bytes / 1024**2:.1f} MB copied to DRAM "
            f"(pool={'on' if use_pool else 'off'}, "
            f"pin_memory={'on' if self._pin_memory else 'off'})"
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
            # PyTorch H2D path: use transfer_weight kernel for NZ-stored
            # weights (aclrtMemcpyAsync, layout-agnostic, graph-safe),
            # and copy_ for ND-stored scales.
            for name in weight_names:
                shape_key = (layer_id, name)
                is_nz_weight = (
                    shape_key in self._weight_shapes
                    and target_buffers[name].dtype == torch.uint8
                    and "scale" not in name
                )
                if is_nz_weight and _TRANSFER_WEIGHT_AVAILABLE:
                    # NZ weight: batch H2D via transfer_weight (1 call per weight)
                    full_dram = self._dram_layer_tensors.get((layer_id, name))
                    if full_dram is not None:
                        transfer_weight(
                            target_buffers[name], full_dram,
                            direction=TransferDirection.H2D,
                        )
                    else:
                        for eid in range(num_local_experts):
                            key = (layer_id, eid)
                            if key not in self.dram_store:
                                continue
                            transfer_weight(
                                target_buffers[name][eid],
                                self.dram_store[key][name],
                                direction=TransferDirection.H2D,
                            )
                else:
                    # Scale tensor (ND): use copy_
                    for eid in range(num_local_experts):
                        key = (layer_id, eid)
                        if key not in self.dram_store:
                            continue
                        target_buffers[name][eid].copy_(
                            self.dram_store[key][name], non_blocking=True
                        )
            return

        # Reusable buffers: torch.zeros/ones allocates new tensors each call
        # with non-deterministic addresses, causing graph replay corruption.
        # Use a single persistent buffer reinitialized with .fill_(/.zero_()
        # (capturable element-wise kernels).
        if (
            self._prefill_group_list_buf is None
            or self._prefill_group_list_buf.shape[0] != num_local_experts
        ):
            self._prefill_group_list_buf = torch.ones(
                num_local_experts, dtype=torch.int64, device=target_device
            )
            self._prefill_packed_group_list_buf = torch.zeros(
                num_local_experts, dtype=torch.int64, device=target_device
            )
        group_list = self._prefill_group_list_buf
        packed_group_list = self._prefill_packed_group_list_buf
        # packed_group_list.zero_() — not strictly needed since all ones
        # input produces all ones output, but kept for safety.
        packed_group_list.zero_()
        device = torch.device(f"npu:{torch.npu.current_device()}")

        for name in weight_names:
            # Use cached ptr tensors; addresses are fixed (DRAM pool +
            # global HBM buffer addresses don't change after init).
            cache_key = (layer_id, name)
            if cache_key in self._prefill_ptr_cache:
                src_ptr_t, dst_ptr_t, len_t, num_le_t = self._prefill_ptr_cache[cache_key]
            else:
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
                if (
                    self._prefill_num_le_t is None
                    or self._prefill_num_le_cache != num_local_experts
                ):
                    self._prefill_num_le_t = torch.tensor(
                        num_local_experts, dtype=torch.int32, device=target_device
                    )
                    self._prefill_num_le_cache = num_local_experts
                num_le_t = self._prefill_num_le_t
                self._prefill_ptr_cache[cache_key] = (src_ptr_t, dst_ptr_t, len_t, num_le_t)

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

        # Reuse global shared NZ HBM buffers (same as prefill path).
        # Global buffers are allocated once by _ensure_shared_global_buffers
        # with NZ format + transpose(1,2), matching the exact state required
        # by CANN GMM. Decode writes compacted active experts to [0..M);
        # tail [M..N) is stale but CANN skips it (packed_group_list[M..N)==0).
        #
        # Using the same buffers for prefill and decode eliminates duplicate
        # HBM allocation (prev ~120 MB double-buffering) and avoids a second
        # npu_format_cast call.
        result = self._ensure_shared_global_buffers(
            layer_id, num_local_experts, weight_names, sample_key, target_device
        )

        if logger.isEnabledFor(logging.WARNING):
            w13_buf = result.get("w13_weight", None)
            logger.warning(
                f"[group_pack_copy_active] layer_id={layer_id} "
                f"group_list.shape={group_list.shape} "
                f"num_local_experts={num_local_experts} "
                f"buf_w13_shape={w13_buf.shape if w13_buf is not None else 'N/A'} "
                f"use_acc_offload={self.use_acc_offload} "
                f"h2d_layer={layer_id in self._h2d_layer_ids} "
                f"capturing={torch.npu.is_current_stream_capturing()}"
            )

        use_group_pack = (
            self.use_acc_offload
            and self._offload_initialized
            and (layer_id not in self._h2d_layer_ids)
        )

        if not use_group_pack:
            # PyTorch H2D path: dual-mode based on graph capture state.
            #
            # Graph mode (capture): transfer_weight batch copy ALL experts.
            #   - Graph-safe (aclrtMemcpyAsync can be captured/replayed)
            #   - CANN skips zero-group_list entries (no extra cost)
            #   - H2D bandwidth: full layer (~347MB)
            #
            # Eager mode: CPU-side selective copy via group_list D2H sync.
            #   - Only copies selected experts (~8 of 164)
            #   - H2D bandwidth: ~17MB (20x reduction)
            #   - Generates packed_group_list for CANN
            #   - Not graph-safe (requires D2H sync)
            is_capturing = torch.npu.is_current_stream_capturing()

            # CANN aclnnGroupedMatmulWeightNz requires group_list size ==
            # weight dim 0. The shared HBM buffer (result[name]) dim 0 is
            # determined by _weight_shapes (NZ storage) or num_local_experts
            # (ND storage), which may differ from the input group_list size
            # (e.g., DeepEP may return a shorter list). Pad group_list to
            # expected_size with zeros so CANN skips the tail experts.
            expected_size = result[weight_names[0]].shape[0]

            if is_capturing:
                # === Graph mode: batch copy all experts ===
                for name in weight_names:
                    shape_key = (layer_id, name)
                    is_nz_weight = (
                        shape_key in self._weight_shapes
                        and result[name].dtype == torch.uint8
                        and "scale" not in name
                    )
                    if is_nz_weight and _TRANSFER_WEIGHT_AVAILABLE:
                        full_dram = self._dram_layer_tensors.get((layer_id, name))
                        if full_dram is not None:
                            transfer_weight(
                                result[name], full_dram,
                                direction=TransferDirection.H2D,
                            )
                        else:
                            for eid in range(num_local_experts):
                                key = (layer_id, eid)
                                if key not in self.dram_store:
                                    continue
                                transfer_weight(
                                    result[name][eid],
                                    self.dram_store[key][name],
                                    direction=TransferDirection.H2D,
                                )
                    elif _TRANSFER_WEIGHT_AVAILABLE:
                        # Scale tensors: use transfer_weight for graph-safe H2D.
                        # torch.copy_() is not capturable for CPU→NPU copies.
                        full_dram = self._dram_layer_tensors.get((layer_id, name))
                        if full_dram is not None:
                            transfer_weight(
                                result[name], full_dram,
                                direction=TransferDirection.H2D,
                            )
                        else:
                            for eid in range(num_local_experts):
                                key = (layer_id, eid)
                                if key not in self.dram_store:
                                    continue
                                transfer_weight(
                                    result[name][eid],
                                    self.dram_store[key][name],
                                    direction=TransferDirection.H2D,
                                )
                    else:
                        for eid in range(num_local_experts):
                            key = (layer_id, eid)
                            if key not in self.dram_store:
                                continue
                            result[name][eid].copy_(
                                self.dram_store[key][name], non_blocking=True
                            )
                # Pad group_list to expected_size for CANN compatibility.
                if group_list.shape[0] != expected_size:
                    if (
                        self._packed_group_list_buf is None
                        or self._packed_group_list_buf.shape[0] != expected_size
                    ):
                        self._packed_group_list_buf = torch.zeros(
                            expected_size, dtype=torch.int64, device=target_device
                        )
                    else:
                        self._packed_group_list_buf.zero_()
                    self._packed_group_list_buf[:group_list.shape[0]] = group_list
                    group_list = self._packed_group_list_buf
                return result, group_list
            else:
                # === Eager mode: CPU-side selective copy ===
                group_list_cpu = group_list.cpu()
                packed_indices = []
                packed_values = []
                for i in range(num_local_experts):
                    val = group_list_cpu[i].item()
                    if val != 0:
                        packed_indices.append(i)
                        packed_values.append(val)

                num_packed = len(packed_indices)
                for name in weight_names:
                    shape_key = (layer_id, name)
                    is_nz_weight = (
                        shape_key in self._weight_shapes
                        and result[name].dtype == torch.uint8
                        and "scale" not in name
                    )
                    for packed_idx in range(num_packed):
                        orig_idx = packed_indices[packed_idx]
                        src_tensor = self.dram_store[(layer_id, orig_idx)][name]
                        if is_nz_weight and _TRANSFER_WEIGHT_AVAILABLE:
                            transfer_weight(
                                result[name][packed_idx], src_tensor,
                                direction=TransferDirection.H2D,
                            )
                        else:
                            result[name][packed_idx].copy_(
                                src_tensor, non_blocking=True
                            )

                # Pad packed_group_list to expected_size (weight dim 0) so
                # CANN group_list size matches weight dim 0. Tail entries
                # are zero — CANN skips experts with 0 tokens.
                if (
                    self._packed_group_list_buf is None
                    or self._packed_group_list_buf.shape[0] != expected_size
                ):
                    self._packed_group_list_buf = torch.zeros(
                        expected_size, dtype=torch.int64, device=target_device
                    )
                else:
                    self._packed_group_list_buf.zero_()
                if num_packed > 0:
                    self._packed_group_list_buf[:num_packed] = torch.tensor(
                        packed_values, dtype=torch.int64, device=target_device
                    )
                packed_group_list = self._packed_group_list_buf
                return result, packed_group_list

        # Pre-allocated packed_group_list buffer (reused via .zero_()).
        # .zero_() is a capturable memset kernel; torch.zeros() allocates a
        # new tensor each call whose address is non-deterministic on graph
        # replay, causing silent data corruption.
        #
        # Buffer size must match weight dim 0 (expected_size), not
        # num_local_experts, because CANN requires group_list size ==
        # weight dim 0. When DeepEP returns a shorter group_list (e.g.,
        # only active experts), num_local_experts < expected_size.
        if (
            self._packed_group_list_buf is None
            or self._packed_group_list_buf.shape[0] != expected_size
        ):
            self._packed_group_list_buf = torch.zeros(
                expected_size, dtype=torch.int64, device=target_device
            )
        else:
            self._packed_group_list_buf.zero_()
        packed_group_list = self._packed_group_list_buf

        device = torch.device(f"npu:{torch.npu.current_device()}")

        # Pad input group_list to expected_size so group_pack_copy kernel can
        # safely iterate [0, num_le=expected_size) without reading OOB.
        # group_list and packed_group_list must be separate buffers (kernel
        # reads group_list while writing packed_group_list).
        if group_list.shape[0] != expected_size:
            if (
                self._padded_group_list_buf is None
                or self._padded_group_list_buf.shape[0] != expected_size
            ):
                self._padded_group_list_buf = torch.zeros(
                    expected_size, dtype=torch.int64, device=target_device
                )
            else:
                self._padded_group_list_buf.zero_()
            self._padded_group_list_buf[:group_list.shape[0]] = group_list
            group_list = self._padded_group_list_buf

        # Call group_pack_copy once per weight name. Each call gets the same
        # group_list and packed_group_list (per-expert, not per-weight).
        # Use expected_size (weight dim 0) so ptr tensors cover all experts
        # in the HBM buffer, matching CANN's group_list size requirement.
        for name in weight_names:
            src_ptr_t, dst_ptr_t, len_t, num_le_t = self._get_decode_ptr_tensors(
                layer_id, name, expected_size, result[name], target_device
            )

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

    def _get_decode_ptr_tensors(
        self,
        layer_id: int,
        name: str,
        num_local_experts: int,
        dst_buffer: torch.Tensor,
        target_device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get or build cached pointer tensors for group_pack_copy (decode).

        On first call: builds src_ptr_t, dst_ptr_t, len_t from dram_store
        addresses and caches them. On subsequent calls: returns cached
        tensors directly, avoiding H2D transfer from
        torch.tensor(python_list, device="npu") which crashes NPU graph
        capture.

        These values are fixed across decode steps because:
          - DRAM pool addresses (src) are allocated once in register_layer_batch
          - Shared decode buffer addresses (dst) are pre-allocated once
          - Weight nbytes (len) don't change
          - num_local_experts is constant

        Returns:
            (src_ptr_t, dst_ptr_t, len_t, num_le_t) — all on-device tensors
        """
        cache_key = (layer_id, name)
        cached = self._decode_ptr_cache.get(cache_key)
        if cached is not None:
            return cached[0], cached[1], cached[2], self._num_le_t

        src_ptrs = []
        dst_ptrs = []
        len_ptrs = []
        for eid in range(num_local_experts):
            key = (layer_id, eid)
            if key not in self.dram_store:
                src_ptrs.append(0)
                dst_ptrs.append(dst_buffer[eid].data_ptr())
                len_ptrs.append(0)
                continue
            dram_tensor = self.dram_store[key][name]
            src_ptrs.append(dram_tensor.data_ptr())
            dst_ptrs.append(dst_buffer[eid].data_ptr())
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

        self._decode_ptr_cache[cache_key] = (src_ptr_t, dst_ptr_t, len_t)

        if self._num_le_cache != num_local_experts:
            self._num_le_t = torch.tensor(
                num_local_experts, dtype=torch.int32, device=target_device
            )
            self._num_le_cache = num_local_experts

        return src_ptr_t, dst_ptr_t, len_t, self._num_le_t

    # ------------------------------------------------------------------
    # Cache mode management
    # ------------------------------------------------------------------ #

    def set_cache_mode(self, is_prefill: bool):
        """Toggle between prefill and decode mode.

        Sets _is_decode_mode which controls the weight loading path:
          - Prefill (is_prefill=True): group_pack_copy_to_buffers loads
            ALL experts with a fake all-ones group_list into
            [num_local_experts, ...] shared HBM buffers
          - Decode (is_prefill=False): group_pack_copy_active_weights uses
            the real post-dispatch group_list to compact active expert
            weights on-device with no D2H sync
        """
        self._is_decode_mode = not is_prefill

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
