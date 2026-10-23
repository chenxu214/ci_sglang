"""Communication groups for EPLB collectives and weight migration."""

import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.runtime_context import get_device


def get_eplb_device_group():
    # Ascend lazily allocates large buffers for a new HCCL communicator.
    # Reuse TP when it has exactly the legacy WORLD membership and ordering.
    # Subgroup deployments keep the original WORLD communication semantics.
    if get_device().device != "npu" or not dist.is_initialized():
        return None
    try:
        tp_group = get_tp_group()
    except AssertionError:
        # Standalone migration tools do not initialize model-parallel groups.
        return None
    except RuntimeError as error:
        # The runtime-context API raises this instead of the legacy assertion.
        if str(error).startswith("parallel name 'tp_group' has not been written"):
            return None
        raise
    if tp_group.ranks == list(range(dist.get_world_size())):
        return tp_group.device_group
    return None
