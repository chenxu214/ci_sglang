from __future__ import annotations

from typing import Optional
from typing import TYPE_CHECKING

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

__all__ = ["NPUCompressedTensorsW4A16mxfp4MoE"]

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


# Unpack the weights to FP4 and return them in float32 format
def unpack_uint8_to_fp4_return_float32(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = packed // 16
    # The high 4 bits and low 4 bits are arranged alternately, with the low 4 bits in front.
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    # A 4-digit integer is mapped to mxfp4 based on its value.
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    return fp4_values[unpacked.to(torch.long)]


class NPUCompressedTensorsW4A16mxfp4MoE(CompressedTensorsMoEScheme):

    def __init__(self):
        self.group_size = 32

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

        # Skip NZ format cast and int4pack when MoE DRAM offload is enabled.
        # Both npu_format_cast(format=29) and npu_convert_weight_to_int4pack
        # produce NPU-specific layouts incompatible with CPU round-trip
        # (AICPU Transpose kernel fails with errorCode=0x2a). For offload,
        # weights are stored in ND format (after unpack+transpose) and
        # converted to NZ+int4pack at forward time in w4a16_mxfp4_gmm_npu.
        _skip_nz_cast = getattr(layer, "moe_dram_offload", False)

        layer.w13_weight.data = unpack_uint8_to_fp4_return_float32(layer.w13_weight.data)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        if not _skip_nz_cast:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29, customize_dtype=torch.bfloat16)
            layer.w13_weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.w13_weight.data).contiguous()

        layer.w2_weight.data = unpack_uint8_to_fp4_return_float32(layer.w2_weight.data)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        if not _skip_nz_cast:
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29, customize_dtype=torch.bfloat16)
            layer.w2_weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.w2_weight.data).contiguous()

        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2).contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2).contiguous()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        combine_input = npu_apply_w4a16_mxfp4_moe_deepep(layer, dispatch_output)
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

        # For DRAM offload (ND weights), extract only selected experts
        # before NZ+int4pack conversion. Converting the full
        # [num_local_experts, ...] tensor would allocate a equally large
        # NZ tensor and cause OOM. With top-k=8 and 896 experts, this
        # reduces the converted tensor from ~18 GiB to ~0.16 GiB.
        if w13.is_contiguous():
            unique_ids, inverse_indices = torch.unique(topk_ids, return_inverse=True)
            w13 = w13[unique_ids]
            w2 = w2[unique_ids]
            w13_scale = w13_scale[unique_ids]
            w2_scale = w2_scale[unique_ids]
            topk_ids = inverse_indices.to(torch.int32)

        output = npu_fused_experts_w4a16_mxfp4(
            hidden_states,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            top_k,
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


def npu_fused_experts_w4a16_mxfp4(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
):
    if torch.npu.is_current_stream_capturing():
        return npu_fused_experts_w4a16_mxfp4_decode(
            hidden_states=hidden_states,
            w13=w13,
            w13_weight_scale=w13_weight_scale,
            w2=w2,
            w2_weight_scale=w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            **kwargs,
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

    hidden_states = w4a16_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w13,
        weight_scale=w13_weight_scale,
        group_list_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states = w4a16_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w2,
        weight_scale=w2_weight_scale,
        group_list_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
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


def npu_fused_experts_w4a16_mxfp4_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
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

    hidden_states = w4a16_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w13,
        weight_scale=w13_weight_scale,
        group_list_type=group_list_type,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states = w4a16_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=w2,
        weight_scale=w2_weight_scale,
        group_list_type=group_list_type,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )

    final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=hidden_states,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def npu_apply_w4a16_mxfp4_moe_deepep(
    layer: torch.nn.Module,
    dispatch_output: "DispatchOutput",
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

    hidden_states = npu_apply_without_routing_weights_w4a16_mxfp4(
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    )
    return combine_cls(
        hidden_states=hidden_states,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


def npu_apply_without_routing_weights_w4a16_mxfp4(
    layer,
    hidden_states,
    hidden_states_scale,
    group_list_type,
    group_list,
    output_dtype,
):
    hidden_states = w4a16_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=hidden_states_scale,
        weight=layer.w13_weight,
        weight_scale=layer.w13_weight_scale,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
    )
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states = w4a16_mxfp4_gmm_npu(
        input=hidden_states,
        input_scale=None,
        weight=layer.w2_weight,
        weight_scale=layer.w2_weight_scale,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
    )
    return hidden_states


def w4a16_mxfp4_gmm_npu(
    input: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_list_type: int,
    group_list: torch.Tensor,
    output_dtype=torch.bfloat16,
) -> torch.Tensor:
    group_list = group_list.to(torch.int64)

    # For ND-format weights (from DRAM offload), convert to NZ+int4pack at
    # forward time. NZ format and int4pack are NPU-specific layouts that
    # cannot be stored in DRAM (AICPU Transpose kernel fails on CPU
    # transfer), so DRAM offload weights are stored in ND format (after
    # unpack+transpose) and converted here. This matches the non-offload
    # path's process_weights_after_loading.
    #
    # The is_contiguous() guard distinguishes the two paths:
    #   - ND path (offload): contiguous after unpack+transpose → convert
    #   - NZ path (non-offload): non-contiguous after NZ cast → skip
    if weight.is_contiguous():
        weight = torch_npu.npu_format_cast(
            weight, 29, customize_dtype=torch.bfloat16
        )
        weight = torch_npu.npu_convert_weight_to_int4pack(weight)

    return torch.ops.npu.npu_grouped_matmul(
        [input],
        [weight],
        antiquant_scale=[weight_scale],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=group_list_type,
        output_dtype=output_dtype,
    )[0]