from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode, async_all_to_all
from sglang.srt.utils.common import get_bool_env_var, is_npu

logger = logging.getLogger(__name__)

if is_npu():
    import torch_npu


class FuseEPDispatchOutput(NamedTuple):
    """DeepEP low latency dispatch output."""

    hidden_state: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


class FuseEPCombineInput(NamedTuple):
    """DeepEP low latency combine input."""

    hidden_state: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


class NpuFuseEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.LOW_LATENCY,
    ):
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs
    ) -> DispatchOutput:
        hidden_states, _ = self._get_buffer().fused_deep_moe(
            hidden_states,
            topk_idx=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
            gmm1_permuted_weight=kwargs["gmm1_permuted_weight"],
            gmm1_permuted_weight_scale=kwargs["gmm1_permuted_weight_scale"],
            gmm2_weight=kwargs["gmm2_weight"],
            gmm2_weight_scale=kwargs["gmm2_weight_scale"],
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            num_experts=self.num_experts,
            fuse_mode=envs.SGLANG_NPU_FUSED_MOE_MODE.get(),
        )
        return FuseEPDispatchOutput(hidden_states)

    def combine(self, combine_input: CombineInput, **kwargs) -> torch.Tensor:
        pass

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class NpuDispatcherWithAllToAllVOutput(NamedTuple):
    """AllToAllV dispatch output."""

    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    combine_metadata: MoEAllToAllCombineInput
    dynamic_scale: torch.Tensor | None = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


class MoEAllToAllCombineInput(NamedTuple):
    input_splits: np.ndarray
    output_splits: np.ndarray
    topk_weights: torch.Tensor
    reversed_local_input_permutation_mapping: torch.Tensor
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size
    reversed_global_input_permutation_mapping: torch.Tensor | None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


class NpuDispatcherWithAllToAllV(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.LOW_LATENCY,
    ):
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.ep_rank = get_moe_ep_group().rank_in_group
        self.ep_size = get_moe_ep_group().world_size
        self.ep_group = get_moe_ep_group()

        self.params_bytes = 2

        self.expert_ids_per_ep_rank = (
            torch.arange(
                self.num_experts, dtype=torch.int32, device=torch.npu.current_device()
            )
            % self.num_local_experts
        )

        local_expert_indices_offset = self.ep_rank * self.num_local_experts

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continuous"

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs
    ) -> DispatchOutput:
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)

        # quant
        input_quant = get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT")
        if input_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens
            )
            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = (
                async_all_to_all(
                    dynamic_scale, output_splits, input_splits, self.ep_group
                )
            )
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)
        else:
            dynamic_scale_after_all2all = None

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits, self.ep_group
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Postprocess
        (
            global_input_tokens,
            dynamic_scale_final,
            reversed_global_input_permutation_mapping,
        ) = self._dispatch_postprocess(
            global_input_tokens,
            global_input_tokens_local_experts_indices,
            input_quant,
            dynamic_scale_after_all2all=dynamic_scale_after_all2all,
        )

        return NpuDispatcherWithAllToAllVOutput(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            combine_metadata=MoEAllToAllCombineInput(
                input_splits=input_splits,
                output_splits=output_splits,
                topk_weights=topk_weights,
                reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
                reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
                hidden_shape=hidden_shape,
                hidden_shape_before_permute=hidden_shape_before_permute,
            ),
        )

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        (
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        ) = self._preprocess(topk_ids)
        hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = (
            torch_npu.npu_moe_token_permute(
                tokens=hidden_states,
                indices=topk_ids,
                num_out_tokens=num_out_tokens,
            )
        )

        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        )

    def _preprocess(self, topk_ids: torch.Tensor):
        num_local_tokens_per_expert = torch.histc(
            topk_ids, bins=self.num_experts, min=0, max=self.num_experts
        )

        ep_size = self.ep_size
        num_out_tokens = topk_ids.numel()

        input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )

        num_global_tokens_per_expert = (
            get_moe_ep_group()
            .all_gather(num_local_tokens_per_expert)
            .reshape(ep_size, self.num_experts)
        )

        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        if num_global_tokens_per_local_expert is None:
            raise ValueError(
                "num_global_tokens_per_local_expert must be set before sum."
            )

        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
            )
        else:
            torch.npu.synchronize()

        return (
            num_tokens_per_local_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        )

    def _dispatch_postprocess(
        self,
        global_input_tokens,
        global_input_tokens_local_experts_indices,
        is_quant,
        dynamic_scale_after_all2all=None,
    ):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, dynamic_scale_after_all2all, None

        # Handle quantized case
        if is_quant:
            assert (
                global_input_tokens_local_experts_indices is not None
            ), "global_input_tokens_local_experts_indices must be provided"
            dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                dynamic_scale_after_all2all.unsqueeze(-1),
                global_input_tokens_local_experts_indices,
            )
            dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(-1)

        # Non-quantized case
        global_input_tokens, reversed_global_input_permutation_mapping = (
            torch_npu.npu_moe_token_permute(
                global_input_tokens, global_input_tokens_local_experts_indices
            )
        )
        return (
            global_input_tokens,
            dynamic_scale_after_all2all,
            reversed_global_input_permutation_mapping,
        )

    def combine(self, combine_input) -> torch.Tensor:
        # 1. Preprocess using metadata
        hidden_states = combine_input.hidden_states
        combine_metadata = combine_input.combine_metadata
        hidden_states = self._combine_preprocess(hidden_states, combine_metadata)

        # 2. AllToAll
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            combine_metadata.input_splits,
            combine_metadata.output_splits,
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # 3. Postprocess using metadata
        output = self._combine_postprocess(
            permutated_local_input_tokens, combine_metadata
        )

        return output

    def _combine_preprocess(
        self, hidden_states: torch.Tensor, combine_metadata
    ) -> torch.Tensor:
        # Unpermutation 2: expert output to AlltoAll input
        rev_global = combine_metadata.reversed_global_input_permutation_mapping
        if (
            hidden_states.shape[0] > 0
            and self.num_local_experts > 1
            and rev_global is not None
        ):
            hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, rev_global)
        return hidden_states

    def _combine_postprocess(
        self,
        permutated_local_input_tokens: torch.Tensor,
        combine_metadata,
    ) -> torch.Tensor:
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=combine_metadata.reversed_local_input_permutation_mapping.to(
                torch.int32
            ),
            probs=combine_metadata.topk_weights,
            restore_shape=combine_metadata.hidden_shape_before_permute,
        )
        output = output.view(combine_metadata.hidden_shape)
        return output


class NpuDispatcherWithAllGatherOutput(NamedTuple):
    """AllToAllV dispatch output."""

    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    combine_metadata: MoEAllGatherCombineInput
    dynamic_scale: torch.Tensor | None = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_AG


class MoEAllGatherCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    reversed_local_input_permutation_mapping: torch.Tensor  # full mapping (not sliced), for npu_moe_token_unpermute
    original_shape: torch.Size
    num_original_tokens: int
    total_tokens: int
    ep_rank: int
    ep_size: int
    local_start: int  # start index in full permuted tokens for local experts
    local_end: int    # end index in full permuted tokens for local experts
    top_k: int        # topk value, needed for padding

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_AG


class NpuDispatcherWithAllGather(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.ALLGATHER,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.ep_rank = get_moe_ep_group().rank_in_group
        self.ep_size = get_moe_ep_group().world_size
        self.ep_group = get_moe_ep_group()

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs
    ) -> DispatchOutput:
        input_quant = get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT")

        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        top_k = topk_ids.shape[1]
        original_shape = hidden_states.shape
        # NOTE: hidden_states, topk_ids, topk_weights are already allgathered by
        # _prepare_with_allgather in DeepseekV2MoE.forward_deepep, so we use them directly.
        num_original_tokens = hidden_states.shape[0]

        first_expert_idx = self.ep_rank * self.num_local_experts
        last_expert_idx = first_expert_idx + self.num_local_experts
        global_num_experts = self.num_experts

        # Step 1: use the already-allgathered data (no need to allgather again)
        hidden_states_all = hidden_states
        topk_ids_all = topk_ids
        topk_weights_all = topk_weights
        total_tokens = hidden_states_all.shape[0]
        active_num = total_tokens * top_k

        # Step 2: permute tokens by expert globally
        permuted_tokens, reversed_mapping = torch_npu.npu_moe_token_permute(
            hidden_states_all,
            topk_ids_all,
            num_out_tokens=active_num,
        )
        # permuted_tokens: (total_tokens * top_k, hidden_size), sorted by global expert idx
        # reversed_mapping: (total_tokens * top_k,), encodes (token_idx, topk_slot)

        # Step 3: count tokens per expert globally
        tokens_per_expert = torch.bincount(
            topk_ids_all.flatten(), minlength=global_num_experts
        ).to(torch.int64)

        # Step 4: slice out local expert tokens from the globally permuted output
        cumsum = tokens_per_expert.cumsum(0)
        local_start = 0 if first_expert_idx == 0 else cumsum[first_expert_idx - 1].item()
        local_end = cumsum[last_expert_idx - 1].item()
        local_permuted = permuted_tokens[local_start:local_end]

        # DEBUG: print dispatch stats
        # print(f"[ALLGATHER-DISPATCH] rank={self.ep_rank}/{self.ep_size}, "
        #       f"num_original_tokens={num_original_tokens}, total_tokens={total_tokens}, "
        #       f"active_num={active_num}, top_k={top_k}")
        # print(f"[ALLGATHER-DISPATCH] rank={self.ep_rank}, "
        #       f"first_expert={first_expert_idx}-{last_expert_idx}, "
        #       f"local_start={local_start}, local_end={local_end}, "
        #       f"local_tokens={local_end - local_start}")
        non_zero = [(i, v.item()) for i, v in enumerate(tokens_per_expert) if v.item() > 0]
        # print(f"[ALLGATHER-DISPATCH] rank={self.ep_rank}, "
        #       f"expert_token_counts(nonzero)={non_zero}")

        # Save the FULL reversed_mapping (not sliced) for combine's npu_moe_token_unpermute
        # which requires dim(0) == total_tokens * top_k

        # Step 5: quantization (if enabled)
        if input_quant:
            local_permuted, pertoken_scale = torch_npu.npu_dynamic_quant(local_permuted)
        else:
            pertoken_scale = None

        # Step 6: build group_list for grouped_matmul
        # group_type=0 means direct 1:1 mapping: group_list[i] -> weight[i]
        # So we need only the local expert token counts (num_local_experts entries)
        expert_tokens = tokens_per_expert[first_expert_idx:last_expert_idx]

        return NpuDispatcherWithAllGatherOutput(
            hidden_states=local_permuted,
            dynamic_scale=pertoken_scale if input_quant else None,
            group_list=expert_tokens,
            group_list_type=1,
            combine_metadata=MoEAllGatherCombineInput(
                hidden_states=local_permuted,
                topk_weights=topk_weights_all,
                reversed_local_input_permutation_mapping=reversed_mapping,  # FULL mapping
                original_shape=original_shape,
                num_original_tokens=num_original_tokens,
                total_tokens=total_tokens,
                ep_rank=self.ep_rank,
                ep_size=self.ep_size,
                local_start=local_start,
                local_end=local_end,
                top_k=top_k,
            ),
        )

    def combine(self, combine_input) -> torch.Tensor:
        assert combine_input.original_shape is not None

        # combine_input.hidden_states: (N_local, hidden_size) - GMM output for local experts
        # combine_input.reversed_local_input_permutation_mapping: (total_tokens * top_k,) - FULL mapping from npu_moe_token_permute
        # combine_input.topk_weights: (total_tokens, top_k) - allgathered weights
        # combine_input.total_tokens: total tokens across all ranks after allgather
        # combine_input.num_original_tokens: tokens originally on this rank
        # combine_input.local_start / local_end: indices of local expert tokens in full permuted output

        gmm_output = combine_input.hidden_states
        full_reversed_mapping = combine_input.reversed_local_input_permutation_mapping  # FULL, shape (total_tokens * top_k,)
        topk_weights = combine_input.topk_weights  # (total_tokens, top_k)
        total_tokens = combine_input.total_tokens
        top_k = combine_input.top_k
        hidden_size = gmm_output.shape[-1]
        dtype = gmm_output.dtype
        device = gmm_output.device
        local_start = combine_input.local_start
        local_end = combine_input.local_end
        num_original_tokens = combine_input.num_original_tokens
        ep_size = combine_input.ep_size

        # Step 1: pad local GMM output to full permuted tokens size
        # npu_moe_token_unpermute requires dim(0) == total_tokens * top_k
        full_permuted = torch.zeros(
            (total_tokens * top_k, hidden_size), dtype=dtype, device=device
        )
        full_permuted[local_start:local_end] = gmm_output

        # DEBUG: gmm output stats
        # print(f"[ALLGATHER-COMBINE] rank={combine_input.ep_rank}/{ep_size}, "
        #       f"gmm_output shape={gmm_output.shape}, "
        #       f"min={gmm_output.min().item():.6f}, max={gmm_output.max().item():.6f}, "
        #       f"mean={gmm_output.mean().item():.6f}")
        # print(f"[ALLGATHER-COMBINE] rank={combine_input.ep_rank}, "
        #       f"full_permuted shape={full_permuted.shape}, "
        #       f"local_range=[{local_start}:{local_end}], "
        #       f"nonzero_ratio={(full_permuted.abs().sum(dim=-1) > 0).float().mean().item():.4f}, "
        #       f"mean={full_permuted.mean().item():.6f}")

        # Step 2: unpermute full tensor with weight application
        # This scatters all expert outputs (local=GMM result, remote=zeros) to token positions
        full_output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=full_permuted,
            sorted_indices=full_reversed_mapping.to(torch.int32),
            probs=topk_weights,
            restore_shape=(total_tokens, hidden_size),
        )
        # full_output: (total_tokens, hidden_size) with weighted partial results from local experts

        # DEBUG: after unpermute
        # print(f"[ALLGATHER-COMBINE] rank={combine_input.ep_rank}, "
        #       f"full_output shape={full_output.shape}, dtype={full_output.dtype}, "
        #       f"min={full_output.min().item():.6f}, max={full_output.max().item():.6f}, "
        #       f"mean={full_output.mean().item():.6f}")

        # Step 3: return full_output directly, leaving reduction to _finalize_with_allgather
        # which calls reduce_scatter_tensor to sum partial contributions and scatter back.
        # Each rank's full_output only contains this rank's local expert contributions
        # (remote expert regions are zeros), so reduce_scatter sums across ranks correctly.

        # DEBUG: after combine (before _finalize_with_allgather)
        # print(f"[ALLGATHER-COMBINE] rank={combine_input.ep_rank}, "
        #       f"returning full_output shape={full_output.shape}, "
        #       f"will be reduce_scattered by _finalize_with_allgather")

        return full_output
