# SPDX-License-Identifier: Apache-2.0
"""Expert weight store for MoE DRAM offloading.

Manages MoE expert weights in Host DRAM. During forward, only the
Top-K selected experts are loaded from Host DRAM to HBM on demand.

Two backends are supported:
  1. acc_offload (default when available): Uses MemFabric acc_offload
     group_pack_copy AICore AIV kernel for batch sparse copy.
     Higher performance due to 32-core parallelism and reduced API overhead.
     No CPU sync (group_list stays on NPU, kernel writes packedGroupList).
  2. PyTorch H2D (fallback): Uses tensor.to("npu", non_blocking=True).
     No external dependency, works everywhere.

Weight loading paths:
  - Prefill (with prefetch): prefetch_layer_to_buffer() async-loads ALL
    experts for the first N layers on h2d_stream. wait_prefill_prefetch()
    synchronizes via per-layer NPU event before compute.
  - Prefill (no prefetch, N=0): _load_experts_on_demand() loads Top-K
    experts into shared HBM buffers synchronously per layer.
  - Decode: build_active_weights_group_pack() uses pre-allocated fixed
    [MAX_ACTIVE, ...] HBM buffers + group_pack_copy kernel. Active experts
    are compacted into the first num_active slots (kernel skips
    group_list[i]==0 entries); packedGroupList is written by kernel.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

logger = logging.getLogger(__name__)

# Maximum number of active experts per layer per rank during decode.
# Pre-allocated fixed HBM buffers [MAX_ACTIVE, ...] are created once per
# layer in prepare_group_pack_buffers() to:
#   1. Eliminate per-forward torch.empty() allocation overhead.
#   2. Eliminate per-forward torch.tensor(dst_ptrs) H2D sync (fixed
#      dstPtrs tensor pre-computed once).
#   3. Enable cuda_graph capture (fixed addresses + fixed shapes).
#
# HBM cost: MAX_ACTIVE * expert_weight_size * num_offloaded_layers.
# For KimiK3 (16.7 MB/expert): MAX_ACTIVE=16 -> ~267 MB/layer -> ~21 GB
# for 80 layers; MAX_ACTIVE=8 -> ~10 GB for 80 layers.
#
# DeepEP normal decode: each token routes to top-k=8 experts across all
# ranks; per-rank num_active (non-zero entries in group_list) is bounded
# by num_local_experts but typically <= 16 in practice for decode batches.
# MAX_ACTIVE=16 covers typical decode; if exceeded, kernel silently
# writes beyond slot 16 (caller never observes — buffer is sized to
# MAX_ACTIVE).
MAX_ACTIVE = 16


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
        #   - Prefill: _load_experts_on_demand / prefetch_layer_to_buffer
        #   - Decode: build_active_weights_group_pack (compact via group_pack_copy)
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
        # group_pack_copy API for H2D transfers.
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

        # group_pack_copy buffers (per-layer, per-weight, pre-computed after
        # register). ALL fields are fixed (DRAM addresses don't change, and
        # HBM dst buffers are pre-allocated once):
        #   - src_ptrs: [num_experts] int64 on NPU (DRAM ptrs)
        #   - lens:     [num_experts] int32 on NPU (bytes per expert)
        #   - num_le:   [1] int32 scalar on NPU (== num_experts)
        #   - dst_ptrs: [MAX_ACTIVE] int64 on NPU (HBM ptrs into fixed buffer)
        #   - hbm_buf:  [MAX_ACTIVE, ...] HBM tensor (compact weight output)
        #   - packed_gl:[MAX_ACTIVE] int64 on NPU (kernel writes here)
        # Per-weight storage: each weight has its own srcPtrs/lenPtrs/numLe/
        # dstPtrs/hbm_buf because group_pack_copy kernel uses a single outIdx
        # across all entries — interleaving weights would break output layout.
        self._group_pack_per_weight: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
        self._group_pack_weight_names: Dict[int, List[str]] = {}
        # Pre-allocated fixed HBM buffers and dstPtrs (per-layer, per-weight).
        # Persistent across forward calls — eliminates per-forward allocation
        # and torch.tensor(dst_ptrs) H2D sync.
        self._group_pack_fixed_hbm: Dict[int, Dict[str, torch.Tensor]] = {}
        self._group_pack_fixed_dst_ptrs: Dict[int, Dict[str, torch.Tensor]] = {}
        self._group_pack_fixed_packed_gl: Dict[int, torch.Tensor] = {}

    def prepare_group_pack_buffers(
        self, layer_id: int, num_experts: int, weight_names: List[str]
    ):
        """Pre-compute ALL fixed tensors for group_pack_copy after register_expert.

        All fields are pre-allocated once here and reused across all forward
        calls — no per-forward torch.empty / torch.tensor allocation, no H2D
        sync. This enables cuda_graph capture (fixed addresses + shapes).

        Pre-allocated per-layer per-weight:
          - src_ptrs: [num_experts] int64 on NPU (DRAM ptrs, fixed)
          - lens:     [num_experts] int32 on NPU (bytes per expert, fixed)
          - num_le:   [1] int32 scalar on NPU (== num_experts, fixed)
          - dst_ptrs: [MAX_ACTIVE] int64 on NPU (HBM ptrs, fixed)
          - hbm_buf:  [MAX_ACTIVE, ...] HBM tensor (compact output, fixed)
        Pre-allocated per-layer (shared across weights — same group_list):
          - packed_gl: [MAX_ACTIVE] int64 on NPU (kernel writes here)

        Per-weight storage: each weight gets its own srcPtrs/lenPtrs/numLe/
        dstPtrs/hbm_buf because group_pack_copy kernel uses a single outIdx
        across all entries — interleaving weights would break output layout.

        Must be called after all experts are registered (register_expert).

        HBM cost per layer = sum over weights of MAX_ACTIVE * expert_size.
        For KimiK3: 4 weights × 16.7 MB total per expert → 16 × 16.7 = 267 MB.
        """
        if not (self.use_acc_offload and self._offload_initialized):
            return
        if layer_id in self._h2d_layer_ids:
            return  # H2D tail layer, skip

        device = f"npu:{torch.npu.current_device()}"
        per_weight = {}
        fixed_hbm = {}
        fixed_dst_ptrs = {}
        total_hbm_bytes = 0
        for name in weight_names:
            src_ptrs = []
            lens = []
            for eid in range(num_experts):
                dram_tensor = self.dram_store[(layer_id, eid)][name]
                src_ptrs.append(dram_tensor.data_ptr())
                lens.append(dram_tensor.nbytes)

            per_weight[name] = {
                "src_ptrs": torch.tensor(src_ptrs, dtype=torch.int64, device=device),
                "lens": torch.tensor(lens, dtype=torch.int32, device=device),
                "num_le": torch.tensor([num_experts], dtype=torch.int32, device=device),
            }

            # Pre-allocate fixed HBM buffer [MAX_ACTIVE, ...] and dstPtrs
            # tensor [MAX_ACTIVE] int64. Persistent across forward calls.
            sample_tensor = self.dram_store[(layer_id, 0)][name]
            buf_shape = (MAX_ACTIVE,) + tuple(sample_tensor.shape)
            hbm_buf = torch.empty(
                buf_shape, dtype=sample_tensor.dtype, device=device
            )
            dst_ptrs_t = torch.tensor(
                [hbm_buf[i].data_ptr() for i in range(MAX_ACTIVE)],
                dtype=torch.int64,
                device=device,
            )
            fixed_hbm[name] = hbm_buf
            fixed_dst_ptrs[name] = dst_ptrs_t
            total_hbm_bytes += hbm_buf.nbytes + dst_ptrs_t.nbytes

        # packedGroupList output is shared across weights (same group_list
        # input produces same output); single tensor per layer.
        packed_gl = torch.zeros(MAX_ACTIVE, dtype=torch.int64, device=device)

        self._group_pack_per_weight[layer_id] = per_weight
        self._group_pack_weight_names[layer_id] = weight_names
        self._group_pack_fixed_hbm[layer_id] = fixed_hbm
        self._group_pack_fixed_dst_ptrs[layer_id] = fixed_dst_ptrs
        self._group_pack_fixed_packed_gl[layer_id] = packed_gl
        logger.info(
            f"[ExpertWeightStore] group_pack_copy prepared for layer {layer_id}: "
            f"{num_experts} experts x {len(weight_names)} weights, "
            f"MAX_ACTIVE={MAX_ACTIVE}, fixed HBM={total_hbm_bytes / 1024**2:.1f} MB"
        )

    def build_active_weights_group_pack(
        self,
        layer_id: int,
        group_list: torch.Tensor,
        num_experts: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Build compact weights via group_pack_copy using pre-allocated
        fixed HBM buffers, then truncate to [num_active].

        Uses pre-computed srcPtrs/lenPtrs/numLe/dstPtrs (all fixed) and the
        pre-allocated [MAX_ACTIVE, ...] HBM buffer. The kernel compacts
        non-zero group_list entries into the first num_active slots and
        writes packedGroupList.

        After kernel completes, narrows weight + group_list to [num_active]
        because CANN npu_grouped_matmul with group_list_type=1 requires
        weight.shape[0] == group_list.shape[0] == actual active expert count
        (does NOT tolerate trailing zero entries — stale weight slots would
        be read and corrupt output).

        The narrow introduces ONE .item() sync (counting non-zero entries in
        packed_gl). This is cheaper than sparse_copy's .cpu() + .tolist() +
        Python-side filtering + dynamic torch.empty allocation. Trade-off:
        cuda_graph capture is NOT supported (graph requires fixed shapes).

        Args:
            layer_id: Layer index
            group_list: [num_experts] int64 tensor on NPU (from DeepEP)
            num_experts: Number of local experts (must match prepare's value)

        Returns:
            (compact_weights, packed_group_list):
              compact_weights: {name: [num_active, ...] tensor (view into
                fixed HBM buffer; only first num_active slots are valid)}
              packed_group_list: [num_active] int64 tensor (view into fixed
                packed_gl; all entries non-zero)
        """
        self._ensure_initialize()

        per_weight = self._group_pack_per_weight[layer_id]
        weight_names = self._group_pack_weight_names[layer_id]
        fixed_hbm = self._group_pack_fixed_hbm[layer_id]
        fixed_dst_ptrs = self._group_pack_fixed_dst_ptrs[layer_id]
        packed_gl = self._group_pack_fixed_packed_gl[layer_id]
        device = torch.device(f"npu:{torch.npu.current_device()}")

        # Zero-init packed_gl in-place (kernel only writes first num_active
        # slots; leftover from previous forward would corrupt routing).
        packed_gl.zero_()

        # Call group_pack_copy once per weight (sharing group_list).
        # All weights see the same group_list, so packed_gl output is identical
        # across calls — we let each call overwrite (last one wins, same data).
        for name in weight_names:
            w_info = per_weight[name]
            ret = self._offload.group_pack_copy(
                w_info["src_ptrs"],
                fixed_dst_ptrs[name],
                w_info["lens"],
                w_info["num_le"],
                group_list,
                packed_gl,
                device,
            )
            if ret != 0:
                raise RuntimeError(
                    f"[ExpertWeightStore] group_pack_copy failed "
                    f"(ret={ret}, layer_id={layer_id}, weight='{name}'). "
                    f"DRAM pool may be corrupted or src/dst ptrs invalid."
                )

        # CANN npu_grouped_matmul with group_list_type=1 requires
        # weight.shape[0] == group_list.shape[0] == num_active (no trailing
        # zeros). Narrow the fixed buffer views to [num_active].
        # Single .item() sync — cheaper than sparse_copy's .cpu()+.tolist().
        num_active = (packed_gl > 0).sum().item()
        if num_active == 0:
            raise RuntimeError(
                f"[ExpertWeightStore] group_pack_copy produced 0 active "
                f"experts (layer_id={layer_id}). Check group_list input."
            )
        if num_active > MAX_ACTIVE:
            raise RuntimeError(
                f"[ExpertWeightStore] num_active={num_active} > MAX_ACTIVE="
                f"{MAX_ACTIVE} (layer_id={layer_id}). Increase MAX_ACTIVE."
            )

        # Narrow to [num_active] — views into fixed buffer, no copy.
        result = {name: fixed_hbm[name][:num_active] for name in weight_names}
        packed_gl_narrowed = packed_gl[:num_active]

        self._stats["total_requests"] += 1
        self._stats["dram_load"] += 1

        return result, packed_gl_narrowed

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
        # Track temporary CPU tensors (created by .cpu()) so we can release
        # them explicitly after copying to the DRAM pool. Without this,
        # PyTorch CPU caching allocator holds the memory and does not return
        # it to the OS, causing host DRAM usage to grow unbounded.
        #
        # Note: We do NOT call torch.cpu.empty_cache() here because it would
        # be invoked 896 times per layer (once per expert), causing significant
        # overhead. The caller (offload_expert_weights_to_dram) is responsible
        # for calling _release_cpu_cache() once after all experts are registered.
        temp_cpu_tensors = []
        try:
            for name, tensor in weights.items():
                # NPU internal format (e.g., FRACTAL_NZ) cannot be copied via
                # copy_() or .cpu() -- NPU raises "do not support internal
                # format". npu_format_cast to ND may only change metadata
                # without reformatting storage, so .contiguous() forces a real
                # ND copy.
                if tensor.device.type != "cpu":
                    # FRACTAL_NZ format cannot be copied via .copy_() or .cpu().
                    # Cast to ND first, then .contiguous() forces a real format
                    # conversion (not just metadata change). If this fails,
                    # raise immediately -- a silent fallback to .contiguous()
                    # alone does NOT guarantee NZ->ND and would cause "do not
                    # support internal format" errors later in copy_().
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
            # Release temporary CPU tensor Python references immediately.
            # PyTorch CPU caching allocator may still hold the underlying
            # memory; caller must invoke _release_cpu_cache() after the full
            # layer registration loop to return it to the OS.
            del temp_cpu_tensors

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

    def _h2d_copy_pytorch(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        sync: bool = True,
    ) -> None:
        """Batch H2D copy via PyTorch copy_() on h2d_stream.

        Used by prefill paths (batch_load_to_hbm / prefetch_layer_to_buffer).
        Decode path uses group_pack_copy via build_active_weights_group_pack
        (no sync, fixed buffers).

        Args:
            pairs: List of (src_cpu_tensor, dst_hbm_tensor) pairs.
                   src/dst must have the same nbytes.
            sync: Whether to synchronize after copy. Set False for async
                  prefetch (caller records an event instead).
        """
        if not pairs:
            return
        if self._h2d_stream is not None:
            with torch.npu.stream(self._h2d_stream):
                for src, dst in pairs:
                    dst.copy_(src, non_blocking=True)
            if sync:
                self._h2d_stream.synchronize()
        else:
            # No h2d_stream (CPU-only): synchronous copy.
            for src, dst in pairs:
                dst.copy_(src)

    def batch_load_to_hbm(
        self,
        layer_id: int,
        expert_ids: List[int],
        shared_buffers: Dict[str, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Batch load expert weights from DRAM to HBM buffers (PyTorch copy_).

        Used by prefill path (_load_experts_on_demand). Writes directly
        into the provided HBM buffers indexed by expert_id, avoiding an
        extra HBM→HBM copy.

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

        # Build (src_cpu, dst_hbm) pairs pointing directly into shared
        # buffers. _h2d_copy_pytorch runs on h2d_stream + sync.
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

        self._h2d_copy_pytorch(pairs, sync=True)

        return results

    # ------------------------------------------------------------------
    # Prefill full-layer prefetch + cache mode management
    # ------------------------------------------------------------------ #

    def set_cache_mode(self, is_prefill: bool):
        """Toggle between prefill and decode mode.

        Sets _is_decode_mode which controls the weight loading path:
          - Prefill (is_prefill=True): _load_experts_on_demand /
            prefetch_layer_to_buffer loads into [num_local_experts, ...] buffers
          - Decode (is_prefill=False): build_active_weights_group_pack
            compacts active experts into fixed [MAX_ACTIVE, ...] HBM buffer
            via group_pack_copy kernel (no sync, no per-forward alloc).
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

        Allocates [num_experts, ...] tensors and loads from DRAM on h2d_stream
        via PyTorch copy_() (async, no sync). Caller must wait on the returned
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

        # Build (src_cpu, dst_hbm) pairs for batch PyTorch H2D copy.
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
            # Async batch H2D on h2d_stream without sync.
            # Caller waits on event (recorded below) before using buffers.
            self._h2d_copy_pytorch(pairs, sync=False)
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
