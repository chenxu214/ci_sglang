from __future__ import annotations

import logging
from typing import Optional, Callable, TYPE_CHECKING

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported
from sglang.srt.layers.quantization.utils import (
    prepare_static_weights_for_trtllm_fp4_moe,
    reorder_w1w3_to_w3w1,
    replace_parameter,
    swizzle_blockscale,
)
from sglang.srt.utils import next_power_of_2, set_weight_attrs

import torch_npu

from sglang.srt.layers.activation import SituAndMul
from sglang.srt.hardware_backend.npu.utils import situ_and_mul

logger = logging.getLogger(__name__)

__all__ = ["NPUCompressedTensorsW4A8mxfp4MoE"]

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


count = 1

def set_count():
    global count
    count += 1

def get_count():
    global count
    return count

def _npu_swiglu(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.npu.npu_swiglu(x)


class NPUCompressedTensorsW4A8mxfp4MoE(CompressedTensorsMoEScheme):

    def __init__(self):
        self.group_size = 32
        self.act_fn: Callable = _npu_swiglu

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # Skip NZ format cast when MoE DRAM offload is enabled.
        # NZ format is incompatible with CPU round-trip (clone/copy_/
        # npu_format_cast(→0) all fail on internal format). For offload,
        # weights are stored in ND format here:
        #   - ND storage (H2D layers): stored as ND in DRAM, forward
        #     converts to NZ via is_nd_format flag in w4a8_mxfp4_gmm_npu.
        #   - NZ storage (acc_offload pool layers): register_expert
        #     converts ND→NZ on NPU and stores NZ bytes via sparse_copy.
        #     Forward uses NZ directly (is_nz_stored flag), no conversion.
        _skip_nz_cast = getattr(layer, "moe_dram_offload", False)

        # If weights are on CPU (DRAM offload with _force_cpu_allocation),
        # move to NPU first — npu_format_cast requires NPU backend.
        if not _skip_nz_cast:
            if layer.w13_weight.data.device.type == "cpu":
                layer.w13_weight.data = layer.w13_weight.data.npu()
                layer.w2_weight.data = layer.w2_weight.data.npu()
            if layer.w13_weight_scale.data.device.type == "cpu":
                layer.w13_weight_scale.data = layer.w13_weight_scale.data.npu()
                layer.w2_weight_scale.data = layer.w2_weight_scale.data.npu()

            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
            )
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
            )
            layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
            layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        else:
            # ND format: just transpose for offload storage.
            # For NZ storage layers, register_expert will transpose back
            # to [N, K] and convert to NZ format on NPU before storing.
            # For ND storage layers, forward will convert to NZ via
            # is_nd_format flag in w4a8_mxfp4_gmm_npu.
            layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.moe_runner_config.activation == "situ":
            # self.act_fn = SituAndMul(
            #     beta=self.moe_runner_config.activation_situ_beta,
            #     linear_beta=self.moe_runner_config.activation_situ_linear_beta,
            # )
            self.act_fn = situ_and_mul

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        combine_input = npu_apply_w4a8_mxfp4_moe_deepep(layer, dispatch_output, act_fn=self.act_fn)
        if combine_input is not None:
            return combine_input

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        hidden_states = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(hidden_states.dtype)
        top_k = (
            self.moe_runner_config.top_k
            if self.moe_runner_config is not None
            else topk_ids.shape[1]
        )

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale

        # DRAM offload path: weights are ND (contiguous). Extract only the
        # selected experts before NZ conversion to avoid converting the
        # entire [num_experts, ...] tensor (which doubles HBM and causes
        # OOM). Use the offload flag instead of is_contiguous() because
        # NZ-format weights may also report contiguous=True on NPU.
        #
        # NZ storage path: weights are already in NZ format (stored in DRAM
        # as NZ bytes via sparse_copy). No expert extraction or NZ conversion
        # needed — the non-DeepEP path is rarely used with NZ storage (DeepEP
        # handles expert extraction via build_active_weight_tensors).
        _store = getattr(layer, "_expert_weight_store", None)
        is_nz_stored = (
            _store is not None
            and _store.use_nz_storage_for_layer(layer.layer_id)
        )
        is_nd_format = (
            getattr(layer, "_dram_offload_enabled", False)
            and not is_nz_stored
        )
        if is_nd_format:
            unique_ids, inverse_indices = torch.unique(
                topk_ids, return_inverse=True
            )
            w13 = w13[unique_ids]
            w2 = w2[unique_ids]
            w13_scale = w13_scale[unique_ids]
            w2_scale = w2_scale[unique_ids]
            topk_ids = inverse_indices.to(torch.int32).view_as(topk_ids)

        output = npu_fused_experts_w4a8_mxfp4(
            hidden_states,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            top_k,
            act_fn=self.act_fn,
            is_nd_format=is_nd_format,
            is_nz_stored=is_nz_stored,
        )
        return StandardCombineInput(hidden_states=output)


def _reshape_mxfp4_scale_for_npu(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 3:
        num_experts, n, k32 = scale.shape
        if k32 % 2 != 0:
            raise ValueError(
                "MXFP4 scale K dimension must be divisible by 2 for "
                "[E, K/64, N, 2] layout."
            )
        scale = scale.view(num_experts, n, k32 // 2, 2).transpose(1, 2)
    return scale


def npu_fused_experts_w4a8_mxfp4(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    act_fn: Callable = _npu_swiglu,
    is_nd_format: bool = False,
    is_nz_stored: bool = False,
):
    if torch.npu.is_current_stream_capturing():
        return npu_fused_experts_w4a8_mxfp4_decode(
            hidden_states=hidden_states,
            w13=w13,
            w13_weight_scale=w13_weight_scale,
            w2=w2,
            w2_weight_scale=w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            act_fn=act_fn,
            is_nd_format=is_nd_format,
            is_nz_stored=is_nz_stored,
        )

    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens,
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)

    rows = hidden_states.shape[0]
    row_ids = torch.arange(rows, device=hidden_states.device, dtype=torch.int64)
    valid_mask = row_ids < expert_tokens[-1]
    valid_mask_2d = valid_mask.unsqueeze(1)

    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w13,
        weight_scale=w13_weight_scale,
        group_list_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    hidden_states = act_fn(hidden_states, expert_tokens, 0)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w2,
        weight_scale=w2_weight_scale,
        group_list_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )

    hidden_states = hidden_states * valid_mask_2d.to(hidden_states.dtype)

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def npu_fused_experts_w4a8_mxfp4_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    act_fn: Callable = _npu_swiglu,
    is_nd_format: bool = False,
    is_nz_stored: bool = False,
):
    num_tokens = hidden_states.shape[:-1].numel()
    global_num_experts = w13.shape[0]
    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    group_list_type = 1

    hidden_states, expanded_row_idx, expert_tokens, _ = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=group_list_type,
            expert_tokens_num_flag=True,
            active_expert_range=[0, global_num_experts],
            quant_mode=-1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w13,
        weight_scale=w13_weight_scale,
        group_list_type=group_list_type,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    hidden_states = act_fn(hidden_states, expert_tokens, group_list_type)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w2,
        weight_scale=w2_weight_scale,
        group_list_type=group_list_type,
        group_list=expert_tokens,
        output_dtype=original_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )

    final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=hidden_states,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def npu_apply_w4a8_mxfp4_moe_deepep(
    layer: torch.nn.Module,
    dispatch_output: "DispatchOutput",
    act_fn: Callable = _npu_swiglu,
) -> Optional["CombineInput"]:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )
    from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

    if not dispatch_output.format.is_deepep():
        return None

    output_dtype = torch.bfloat16
    group_list_type = 1

    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
            dispatch_output
        )
        group_list = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        combine_cls = DeepEPNormalCombineInput
    else:
        hidden_states, hidden_states_scale, _, _, group_list, _ = dispatch_output
        group_list = group_list.to(torch.int64)
        combine_cls = DeepEPLLCombineInput

    # Early return when this rank received no tokens (group_list_sum == 0).
    # In DeepEP, some ranks may receive 0 tokens for certain layers. Running
    # the CANN kernel with 0 tokens still requires group_list size == weight
    # dim 0, but the weight may be stale (e.g., [num_active, ...] from a
    # previous decode). Skip the kernel entirely — there's nothing to compute.
    if hidden_states.shape[0] == 0 or group_list.sum().item() == 0:
        return combine_cls(
            hidden_states=hidden_states,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    # Decode DRAM offload: build compact [num_active, ...] weight tensors
    # after dispatch (we know which experts received tokens). Replaces the
    # [224, ...] shared buffer with a smaller [num_active, ...] tensor.
    # Prefill uses the shared buffer (pre-loaded by _load_experts_on_demand).
    if (
        getattr(layer, "_dram_offload_enabled", False)
        and layer._expert_weight_store is not None
        and layer._expert_weight_store._is_decode_mode
    ):
        _store = layer._expert_weight_store

        if _store._use_group_pack_copy:
            # group_pack_copy path: graph-capturable, no group_list.cpu()
            # sync. The AIV kernel filters non-zero experts and copies
            # their weights from DRAM to HBM (compacted to slots [0..M)).
            # packed_group_list[0..M) has active token counts; [M..N) is 0
            # (zeroed before the call), so GMM skips inactive experts.
            # Weights are [num_local_experts, ...] (full size, not compact)
            # — trade-off: more HBM but enables graph capture.
            hbm_buffers, packed_group_list = (
                _store.build_active_weights_group_pack(
                    layer.layer_id, group_list, layer.num_local_experts
                )
            )
            for name, tensor in hbm_buffers.items():
                setattr(layer, name, tensor)
            group_list = packed_group_list
        else:
            # Fallback: build_active_weight_tensors with group_list.cpu()
            # sync. Produces compact [num_active, ...] tensors (less HBM
            # but not graph-capturable due to CPU sync).
            group_list_cpu = group_list.cpu()
            active_mask = group_list_cpu > 0
            active_expert_ids = active_mask.nonzero().squeeze(-1).tolist()
            if not isinstance(active_expert_ids, list):
                active_expert_ids = [active_expert_ids]
            num_active = len(active_expert_ids)

            if num_active > 16:
                logger.debug(
                    f"Decode active experts ({num_active}) exceeds 16, "
                    f"using compact path. active_expert_ids={active_expert_ids}"
                )

            if num_active == 0:
                return combine_cls(
                    hidden_states=hidden_states,
                    topk_ids=dispatch_output.topk_ids,
                    topk_weights=dispatch_output.topk_weights,
                )

            sample_key = (layer.layer_id, active_expert_ids[0])
            weight_names = list(
                _store.dram_store[sample_key].keys()
            )
            compact_weights = _store.build_active_weight_tensors(
                layer.layer_id, active_expert_ids, weight_names
            )
            # Set compact weights as layer attributes. NO transpose here
            # for NZ-storage weights — w4a8_mxfp4_gmm_npu's is_nz_stored
            # branch handles the [E, N, K]→[E, K, N] transpose.
            for name, tensor in compact_weights.items():
                setattr(layer, name, tensor)

            group_list = group_list_cpu[active_mask].to(hidden_states.device)

    # Determine NZ storage: when True, weights loaded from DRAM are already
    # in NZ format (stored via sparse_copy), so no ND→NZ conversion is needed
    # at forward time. is_nd_format is only True for H2D layers (ND storage).
    _store = layer._expert_weight_store
    is_nz_stored = (
        _store is not None
        and _store.use_nz_storage_for_layer(layer.layer_id)
    )
    is_nd_format = (
        getattr(layer, "_dram_offload_enabled", False)
        and not is_nz_stored
    )
    hidden_states = npu_apply_without_routing_weights_w4a8_mxfp4(
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
        act_fn=act_fn,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    return combine_cls(
        hidden_states=hidden_states,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


def npu_apply_without_routing_weights_w4a8_mxfp4(
    layer,
    hidden_states,
    hidden_states_scale,
    group_list_type,
    group_list,
    output_dtype,
    act_fn: Callable = _npu_swiglu,
    is_nd_format: bool = False,
    is_nz_stored: bool = False,
):
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=hidden_states_scale,
        weight=layer.w13_weight,
        weight_scale=layer.w13_weight_scale,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    # Release w13 compact weights after GMM to reduce HBM peak.
    # Both ND and NZ storage paths allocate compact [num_active, ...]
    # tensors via build_active_weight_tensors (decode path). Without
    # releasing, w13 compact + w2 compact coexist in HBM, causing OOM
    # during prefill (num_active ≈ 112).
    # For NZ storage, no NZ conversion tensor is created (weights are
    # already NZ), but the compact NZ tensor still occupies HBM.
    if is_nd_format or is_nz_stored:
        layer.w13_weight = None
        layer.w13_weight_scale = None
    hidden_states = act_fn(hidden_states, group_list, group_list_type)
    hidden_states = w4a8_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=layer.w2_weight,
        weight_scale=layer.w2_weight_scale,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
        is_nd_format=is_nd_format,
        is_nz_stored=is_nz_stored,
    )
    # Release w2 compact weights after GMM to reduce HBM peak.
    if is_nd_format or is_nz_stored:
        layer.w2_weight = None
        layer.w2_weight_scale = None
    return hidden_states


def w4a8_mxfp4_gmm_npu(
    input: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_list_type: int,
    group_list: torch.Tensor,
    output_dtype=torch.bfloat16,
    is_nd_format: bool = False,
    is_nz_stored: bool = False,
) -> torch.Tensor:
    group_list = group_list.to(torch.int64)

    if input_scale is None:
        x, x_scale = torch.ops.npu.npu_dynamic_mx_quant(
            input,
            axis=1,
            round_mode="rint",
            dst_type=torch.float8_e4m3fn,
            block_size=32,
            scale_alg=None,
        )
    else:
        x, x_scale = input, input_scale

    # Weight format handling:
    #   is_nd_format=True:  ND storage (H2D layers). Weight is [E, K, N] ND.
    #                      Convert to NZ: transpose→[E,N,K] ND→NZ→[E,N,K] NZ
    #                      →transpose→[E,K,N] NZ. This is the only path that
    #                      allocates extra HBM for the NZ conversion.
    #
    #   is_nz_stored=True: NZ storage (acc_offload pool layers). Weight is
    #                      [E, N, K] NZ (loaded from DRAM via sparse_copy).
    #                      Just transpose (metadata-only) → [E, K, N] NZ.
    #                      No NZ conversion needed, saving HBM and compute.
    #
    #   Both False:         Non-offload path. Weight is already [E, K, N] NZ
    #                      (from process_weights_after_loading). No action.
    if is_nd_format:
        # DRAM offload ND path: weight is ND from DRAM.
        # Convert to NZ format: undo transpose → cast to NZ → re-apply transpose.
        # Split into steps and explicitly del intermediates to reduce HBM
        # peak: without this, compact ND weight + contiguous copy + NZ
        # tensor coexist simultaneously, causing OOM when num_active is
        # large (e.g., ~112 during prefill).
        weight_nd = weight.transpose(1, 2).contiguous().view(torch.uint8)
        weight = torch_npu.npu_format_cast(
            weight_nd,
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        ).transpose(1, 2)
        del weight_nd
    elif is_nz_stored:
        # NZ storage path: weight is [E, N, K] NZ from DRAM.
        # Transpose to [E, K, N] NZ (metadata-only, no data copy).
        # This matches the non-offload path's final format after
        # process_weights_after_loading (NZ cast + transpose).
        weight = weight.transpose(1, 2)

    # Scale: is_contiguous() is reliable here because scales are never
    # cast to NZ format. After process_weights_after_loading, scales are
    # non-contiguous (transposed). DRAM round-trip (.cpu()+.npu()) makes
    # them contiguous again, so the conversion restores the transposed state.
    if weight_scale.is_contiguous():
        weight_scale = (
            weight_scale.permute(0, 2, 1, 3).contiguous()
            .transpose(-3, -2)
        )

    return torch.ops.npu.npu_grouped_matmul(
        [x],
        [weight],
        antiquant_scale=[weight_scale],
        scale_dtype=None,
        scale=None,
        per_token_scale=[x_scale],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=group_list_type,
        output_dtype=output_dtype,
        x_dtype=torch_npu.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
    )[0]