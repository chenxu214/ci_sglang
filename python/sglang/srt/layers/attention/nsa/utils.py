# temp NSA debugging environ
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    attn_cp_all_gather_into_tensor,
    get_attention_cp_group,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_rank,
    is_allocation_symmetric,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import ceil_align, ceil_div

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_nsa_seqlens(original_seq_lens, nsa_index_topk: int):
    return original_seq_lens.clamp(max=nsa_index_topk)


def is_nsa_enable_prefill_cp():
    return get_global_server_args().enable_nsa_prefill_context_parallel


def is_nsa_prefill_cp_in_seq_split():
    return (
        is_nsa_enable_prefill_cp()
        and get_global_server_args().nsa_prefill_cp_mode == "in-seq-split"
    )


def is_nsa_prefill_cp_round_robin_split():
    return (
        is_nsa_enable_prefill_cp()
        and get_global_server_args().nsa_prefill_cp_mode == "round-robin-split"
    )


def can_nsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    cp_size = get_attention_cp_size()
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_nsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def nsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    """
    # for round-robin-split, split the tokens evenly according to the rule of token_idx % cp_size.
    |   +-----------before split------------+|
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |
    |   +--------------result-------------------+
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |   +-------------------------+
    """
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    if isinstance(input_, (tuple, list)):
        indices = range(cp_rank, len(input_), cp_size)
        return input_[indices]

    tokens = len(input_)
    if tokens % cp_size != 0:
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
        if cur_len == 0:
            return input_.new_empty(0, *input_.shape[1:])
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]

    # for torch device tensor
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_cp_size = get_attention_cp_size()
    for i in range(sync_group_size):
        global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size)
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
        forward_batch.is_extend_in_batch, global_num_tokens
    )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    elif len(global_num_tokens) > 1:
        tokens = global_num_tokens[get_attention_dp_rank()]
    else:
        tokens = global_num_tokens[0]
    if can_nsa_prefill_cp_round_robin_split(forward_batch):
        tokens = ceil_div(tokens, attn_cp_size)
    return tokens


def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", nsa_cache_seqlens):
    attn_cp_size = get_attention_cp_size()
    needs_cp_pad = attn_cp_size > 1 and can_nsa_prefill_cp_round_robin_split(
        forward_batch
    )
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
    if not needs_cp_pad and not needs_dp_pad:
        return nsa_cache_seqlens
    tokens = cal_padded_tokens(forward_batch)
    pad_len = tokens - nsa_cache_seqlens.shape[0]
    if pad_len > 0:
        nsa_cache_seqlens = torch.cat(
            [
                nsa_cache_seqlens,
                nsa_cache_seqlens.new_zeros(pad_len, *nsa_cache_seqlens.shape[1:]),
            ]
        )
    return nsa_cache_seqlens


@dataclass
class NSAContextParallelMetadata:

    split_list: List[int] = None
    max_rank_len: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    reverse_split_len: List[int] = None
    cp_reverse_index: List[int] = None
    kv_len_prev: List[int] = None
    kv_len_next: List[int] = None
    actual_seq_q_prev: List[int] = None
    actual_seq_q_next: List[int] = None
    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None
    total_seq_lens: torch.Tensor = None
    batch_size: int = 1


def can_cp_split(seq_len: int, cp_size: int, use_nsa: bool, forward_batch):
    if is_nsa_prefill_cp_round_robin_split():
        cur_cp_seq_len = seq_len // cp_size
        assert (
            seq_len % cp_size == 0
        ), f"seq_len {seq_len} is not divisible by cp_size {cp_size} when nsa_prefill_cp_mode is round-robin-split"
    else:
        min_extend_seq_len = min(forward_batch.extend_seq_lens_cpu)
        cur_cp_seq_len = min_extend_seq_len // (cp_size * 2)
    if (
        cur_cp_seq_len != 0
        and cp_size > 1
        and use_nsa
        and forward_batch.forward_mode.is_context_parallel_extend()
        and is_nsa_enable_prefill_cp()
        and all(s >= cp_size for s in forward_batch.extend_seq_lens_cpu)
    ):
        return True
    else:
        return False


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    if is_nsa_prefill_cp_round_robin_split():
        cp_size = get_attention_cp_size()
        assert (
            input_.shape[0] % cp_size == 0
        ), f"Expect input shape 0 can divided by cp size, but got input shape {input_.shape}, cp size {cp_size}"
        return nsa_cp_round_robin_split_data(input_)

    input_list = list(
        torch.split(input_, forward_batch.nsa_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    if is_nsa_prefill_cp_round_robin_split():
        cp_size = get_attention_cp_size()
        assert positions.shape[0] % cp_size == 0, (
            f"Expect positions shape 0 can divided by cp size, but got positions shape {positions.shape}, "
            f"cp size {cp_size}"
        )
        return nsa_cp_round_robin_split_data(positions)

    position_id_list = list(
        torch.split(positions, forward_batch.nsa_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions


@triton.jit
def nsa_cp_round_robin_split_q_seqs_kernel(
    in_seqs_ptr,
    out_seqs_ptr,
    bs_idx_ptr,
    tokens: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
):
    extra_seq = 0
    bs_idx = 0
    for bs in range(tokens):
        cur_len = tl.load(in_seqs_ptr + bs)
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + (cur_len % cp_size > cp_rank)
        if cur_seq > 0:
            tl.store(bs_idx_ptr + bs_idx, bs)
            tl.store(out_seqs_ptr + bs_idx, cur_seq)
            bs_idx += 1
        extra_seq = cur_len - cur_seq * cp_size


def nsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    extra_seq = 0
    q_seqs = []
    for bs, cur_len in enumerate(extend_seqs):
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def nsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    """
    round-robin-split distributes tokens across ranks based on token_idx % cp_size.

    Return:
    ret_q_lens_cpu(List) and ret_q_lens(torch.Tensor): the partitioned length (excluding zeros) on the current cp rank
        for each sequence after distribution across cp ranks.
    bs_idx_cpu(List) and bs_idx(torch.Tensor): marks which sequences are ultimately selected,
        i.e., those with a partitioned length greater than zero.
    """
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    # len(ret_q_lens_cpu) == len(bs_idx_cpu)
    ret_q_lens_cpu, bs_idx_cpu = nsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    grid = (1,)
    nsa_cp_round_robin_split_q_seqs_kernel[grid](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


def nsa_use_prefill_cp(forward_batch, nsa_enable_prefill_cp=None):
    if nsa_enable_prefill_cp is None:
        nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
    if (
        forward_batch.nsa_cp_metadata is not None
        and nsa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    ):
        return True
    else:
        return False


def cp_attn_tp_all_gather_reorganazied_into_tensor(
    input_: torch.Tensor, total_len, attn_tp_size, forward_batch, stream_op
):
    """
    Allgather communication for context_parallel(kv_cache, index_k, hidden_states).
    This implementation mainly consists of three parts:
    Step 1, padding the input shape to unify the shape for allgather communication (the shape must be the same).
    Step 2, allgather communication(async).
    Step 3, removing the padding and reassembling the data according to the actual tokens.
    """
    # step1
    if forward_batch.nsa_cp_metadata is not None and forward_batch.nsa_cp_metadata.max_rank_len:
        max_len = max(forward_batch.nsa_cp_metadata.max_rank_len)
    else:
        max_len = (total_len + attn_tp_size - 1) // attn_tp_size
    pad_size = max_len - input_.shape[0]
    if pad_size > 0:
        input_ = F.pad(input_, (0, 0, 0, pad_size), mode="constant", value=0)
    with use_symmetric_memory(
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        input_tensor_all = torch.empty(
            max_len * attn_tp_size,
            input_.shape[1],
            device=input_.device,
            dtype=input_.dtype,
        )
    # step2
    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_all, input_, stream_op
    )
    # step3
    outputs_list_max = list(
        torch.split(input_tensor_all, forward_batch.nsa_cp_metadata.max_rank_len, dim=0)
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.nsa_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )
    return outputs


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """
    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+

    # for round-robin-split
    |   +-----------before allgather------------+|
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |
    |   +--------------result-------------------+
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |   +-------------------------+
    """
    if is_nsa_prefill_cp_round_robin_split():
        with use_symmetric_memory(
            get_attention_cp_group(), disabled=not is_allocation_symmetric()
        ):
            output_tensor = input_tensor.new_empty(
                (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),
            )
        attn_cp_all_gather_into_tensor(
            output_tensor,
            input_tensor,
        )
        out_shape = output_tensor.shape
        output_tensor = (
            output_tensor.view(cp_size, -1, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )
        return output_tensor

    bs_seq_len, hidden_size = input_tensor.shape
    output_tensor = cp_attn_tp_all_gather_reorganazied_into_tensor(
        input_tensor,
        forward_batch.nsa_cp_metadata.total_seq_lens,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.nsa_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.nsa_cp_metadata.cp_reverse_index], dim=0
    )
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor


def calculate_cp_seq_idx(cp_chunks_len, seqs_len):
    """Used to obtain the index of the seq corresponding
    to each cp block in the forwardbatch, and the starting
    and ending positions of the corresponding seq in the cp block"""
    j = 0
    tuple_len = []  # Only keep this result list
    cumulative = {}  # Used to track cumulative values for each index

    for i in range(len(cp_chunks_len)):
        current_dict = {}
        current_tuples = []
        c_val = cp_chunks_len[i]

        while j < len(seqs_len):
            s_val = seqs_len[j]
            if s_val == c_val:
                idx = j
                current_dict[idx] = s_val
                # Update cumulative value for this index
                cumulative[idx] = cumulative.get(idx, 0) + s_val
                j += 1
                break
            elif s_val > c_val:
                idx = j
                current_dict[idx] = c_val
                # Update cumulative value for this index
                cumulative[idx] = cumulative.get(idx, 0) + c_val
                seqs_len[j] = s_val - c_val
                break
            else:  # s_val < c_val
                idx = j
                current_dict[idx] = s_val
                # Update cumulative value for this index
                cumulative[idx] = cumulative.get(idx, 0) + s_val
                c_val -= s_val
                j += 1

        # Build tuple: (index, historical cumulative, historical+current)
        for idx, val in current_dict.items():
            # Subtract current value to get historical cumulative
            prev_cum = cumulative.get(idx, 0) - val
            current_cum = prev_cum + val
            current_tuples.append((idx, prev_cum, current_cum))

        tuple_len.append(current_tuples)
    return tuple_len


def _prepare_single_seq_zigzag_split(
    extend_len: int,
    prefix_len: int,
    cp_rank: int,
    cp_size: int,
):
    """Prepare zigzag split metadata for a single sequence.

    Returns:
        split_list: list of chunk sizes after splitting into cp_size*2 chunks
        kv_len_prev: prefix_len + cumulative length up to cp_rank chunk
        kv_len_next: prefix_len + cumulative length up to (cp_size*2 - cp_rank - 1) chunk
        actual_seq_q_prev: query length of the prev chunk for this rank
        actual_seq_q_next: query length of the next chunk for this rank
    """
    kv_len = torch.tensor(extend_len)
    cp_segment_num = cp_size * 2
    seq_per_batch = kv_len // cp_segment_num
    split_list = seq_per_batch.repeat_interleave(cp_segment_num).int().tolist()
    remainder = kv_len % cp_segment_num
    if remainder > 0:
        split_list[:remainder] = [x + 1 for x in split_list[:remainder]]

    prefix_sum_list = list(accumulate(split_list))
    kv_len_prev = prefix_len + prefix_sum_list[cp_rank]
    kv_len_next = prefix_len + prefix_sum_list[cp_size * 2 - cp_rank - 1]
    actual_seq_q_prev = split_list[cp_rank]
    actual_seq_q_next = split_list[cp_size * 2 - cp_rank - 1]
    return split_list, kv_len_prev, kv_len_next, actual_seq_q_prev, actual_seq_q_next


def prepare_input_dp_with_cp_dsa(
    kv_len,
    cp_rank,
    cp_size,
    seqs_len,
    extend_seq_lens=None,
    extend_prefix_lens=None,
):
    if is_nsa_prefill_cp_round_robin_split():
        return True
    """prepare_input_dp_with_cp_dsa-zigzag index (multi-batch support)

    For multi-batch, each sequence is independently split into cp_size*2 chunks
    using zigzag rearrangement. The chunks from all sequences are then concatenated
    to form the per-rank data.

    Example (CP_SIZE == 4, batch_size == 2):
    Sequence 0: split into 8 chunks -> zigzag rearrange -> rank gets (chunk0, chunk7)
    Sequence 1: split into 8 chunks -> zigzag rearrange -> rank gets (chunk0, chunk7)
    Per-rank data: [seq0_chunk0, seq0_chunk7, seq1_chunk0, seq1_chunk7]

    Why zigzag rearrange?
    - Attention calculations must follow causal attention principles.
    - Simply slicing by rank order can lead to computational load imbalance:
        * First rank may focus on fewer historical key-value tokens (less computation)
        * Last rank may focus on more tokens (more computation)
    - To mitigate uneven load, the input hidden states needs to be sliced by cp_size*2 and rearranged.
    """
    if extend_seq_lens is None:
        extend_seq_lens = [kv_len]
    if extend_prefix_lens is None:
        extend_prefix_lens = [0] * len(extend_seq_lens)

    batch_size = len(extend_seq_lens)
    device_str = "npu" if is_npu() else "cuda"

    all_split_list = []
    all_kv_len_prev = []
    all_kv_len_next = []
    all_actual_seq_q_prev = []
    all_actual_seq_q_next = []

    for i in range(batch_size):
        prefix_len = extend_prefix_lens[i] if i < len(extend_prefix_lens) else 0
        split_list, kv_len_prev, kv_len_next, actual_seq_q_prev, actual_seq_q_next = (
            _prepare_single_seq_zigzag_split(
                extend_seq_lens[i], prefix_len, cp_rank, cp_size
            )
        )
        all_split_list.extend(split_list)
        all_kv_len_prev.append(kv_len_prev)
        all_kv_len_next.append(kv_len_next)
        all_actual_seq_q_prev.append(actual_seq_q_prev)
        all_actual_seq_q_next.append(actual_seq_q_next)

    kv_len_origin = torch.tensor(sum(extend_seq_lens))

    cp_segment_num = cp_size * 2
    bs_per_cp_group = batch_size

    per_rank_actual_token = []
    for i in range(cp_size):
        per_rank_seq_token = 0
        for batch_id in range(batch_size):
            base = batch_id * cp_segment_num
            per_rank_seq_token += all_split_list[base + i] + all_split_list[
                base + cp_segment_num - i - 1
            ]
        per_rank_actual_token.append(per_rank_seq_token)

    max_per_rank = max(per_rank_actual_token)
    max_rank_len = [max_per_rank] * cp_size

    zigzag_index = []
    for batch_id in range(bs_per_cp_group):
        zigzag_index.extend(
            list(
                range(
                    batch_id * cp_segment_num + cp_rank,
                    batch_id * cp_segment_num + cp_rank + 1,
                    1,
                )
            )
            + list(
                range(
                    batch_id * cp_segment_num + cp_segment_num - cp_rank - 1,
                    batch_id * cp_segment_num + cp_segment_num - cp_rank,
                    1,
                )
            )
        )

    reverse_split_len = []
    for i in range(cp_size):
        for batch_id in range(batch_size):
            base = batch_id * cp_segment_num
            reverse_split_len.append(all_split_list[base + i])
            reverse_split_len.append(all_split_list[base + cp_segment_num - i - 1])

    cp_reverse_index = []
    for batch_id in range(bs_per_cp_group):
        for rank_i in range(cp_size):
            cp_reverse_index.append(rank_i * batch_size * 2 + batch_id * 2)
        for rank_i in range(cp_size - 1, -1, -1):
            cp_reverse_index.append(rank_i * batch_size * 2 + batch_id * 2 + 1)

    kv_len_prev_tensor = torch.tensor(all_kv_len_prev).to(
        device=device_str, dtype=torch.int32
    )
    kv_len_next_tensor = torch.tensor(all_kv_len_next).to(
        device=device_str, dtype=torch.int32
    )
    actual_seq_q_prev_tensor = torch.tensor(all_actual_seq_q_prev).to(
        device=device_str, dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(all_actual_seq_q_next).to(
        device=device_str, dtype=torch.int32
    )

    nsa_cp_metadata = NSAContextParallelMetadata(
        split_list=all_split_list,
        max_rank_len=max_rank_len,
        zigzag_index=zigzag_index,
        per_rank_actual_token=per_rank_actual_token,
        reverse_split_len=reverse_split_len,
        cp_reverse_index=cp_reverse_index,
        kv_len_prev=all_kv_len_prev,
        kv_len_next=all_kv_len_next,
        actual_seq_q_prev=all_actual_seq_q_prev,
        actual_seq_q_next=all_actual_seq_q_next,
        kv_len_prev_tensor=kv_len_prev_tensor,
        kv_len_next_tensor=kv_len_next_tensor,
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,
        total_seq_lens=kv_len_origin,
        batch_size=batch_size,
    )
    return nsa_cp_metadata
