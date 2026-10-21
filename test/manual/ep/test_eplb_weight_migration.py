"""Bounded EPLB migration check for CPU/Gloo or Ascend/HCCL.

Example: torchrun --standalone --nproc-per-node=2 \
    test/manual/ep/test_eplb_weight_migration.py --device npu --weight-format nd

Uses synthetic processed W8A8-shaped tensors; it does not load a model.
"""

import argparse
import datetime
import json
import os

import torch
import torch.distributed as dist


def _metadata(mapping, rank):
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    num_logical = max(mapping) + 1
    physical = torch.tensor([mapping, mapping], device=_DEVICE)
    inverse = torch.full((2, num_logical, len(mapping)), -1, device=_DEVICE)
    for expert in range(num_logical):
        slots = [index for index, value in enumerate(mapping) if value == expert]
        inverse[:, expert, : len(slots)] = torch.tensor(slots, device=_DEVICE)
    return ExpertLocationMetadata._init_raw(
        ep_size=2,
        physical_to_logical_map=physical,
        logical_to_all_physical_map=inverse,
        moe_ep_rank=rank,
    )


def _expert_tensor(expert, layer, kind):
    # Values encode both the logical expert and positions inside each tensor.
    if kind < 2:
        values = torch.arange(512).reshape(16, 32)
        return ((values + expert * 7 + layer * 11 + kind) % 127).to(torch.int8)
    values = torch.arange(32).to(torch.float32) / 32
    values = values + expert + layer * 16 + kind
    return values.to(torch.bfloat16 if kind == 2 else torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--weight-format", choices=("nd", "nz"), default="nd")
    args = parser.parse_args()
    if args.device == "cpu" and args.weight_format != "nd":
        parser.error("NZ requires --device npu")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("launch with torchrun --nproc-per-node=2")
    rank = int(os.environ["RANK"])
    if args.device == "npu":
        import torch_npu

        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    global _DEVICE
    _DEVICE = args.device

    from sglang.srt.eplb.expert_location import set_global_expert_location_metadata
    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    publish(
        ServerArgs(
            model_path="dummy",
            device=args.device,
            tp_size=2,
            ep_size=2,
            ep_dispatch_algorithm="static",
        ),
        role="test",
    )
    dist.init_process_group(
        backend="hccl" if args.device == "npu" else "gloo",
        timeout=datetime.timedelta(seconds=60),
    )
    updates = 0
    try:
        for initial, layouts in (
            (
                list(range(8)),
                [[7, 0, 4, 3, 2, 5, 6, 1], [4, 5, 6, 7, 0, 1, 2, 3], list(range(8))],
            ),
            (
                [0, 0, 1, 1, 2, 2, 3, 3],
                [
                    [2, 0, 2, 1, 3, 1, 0, 3],
                    [3, 3, 2, 2, 1, 1, 0, 0],
                    [0, 0, 1, 1, 2, 2, 3, 3],
                ],
            ),
        ):
            current = _metadata(initial, rank)
            set_global_expert_location_metadata(current, allow_overwrite=True)
            map_pointers = {
                name: value.data_ptr()
                for name, value in vars(current).items()
                if isinstance(value, torch.Tensor)
            }
            full_weights = {}
            for layer in range(2):
                full_weights[layer] = []
                for kind in range(4):
                    # The final slot represents a fused shared expert. EPLB
                    # receives only the routed view and must preserve this slot.
                    tensor = torch.stack(
                        [
                            _expert_tensor(expert, layer, kind)
                            for expert in initial[rank * 4 : (rank + 1) * 4] + [31]
                        ]
                    ).to(_DEVICE)
                    if args.weight_format == "nz" and kind < 2:
                        tensor = torch_npu.npu_format_cast(tensor, 29)
                    full_weights[layer].append(tensor)
            routed = {
                layer: [tensor[:4] for tensor in tensors]
                for layer, tensors in full_weights.items()
            }
            weight_pointers = [
                [tensor.data_ptr() for tensor in tensors] for tensors in routed.values()
            ]
            updater = ExpertLocationUpdater()
            if args.device == "cpu":
                # The updater's first-use accelerator cache flush has no CPU
                # equivalent; all migration and metadata operations stay real.
                updater._first_execution = False
            expected_maps = [initial.copy(), initial.copy()]
            for layout in layouts:
                target = _metadata(layout, rank)
                for layer in range(2):
                    missing = updater.update(
                        routed, target, [layer], nnodes=1, rank=rank
                    )
                    assert missing == {}, missing
                    expected_maps[layer] = layout.copy()
                    for lid, tensors in full_weights.items():
                        for kind, tensor in enumerate(tensors):
                            expected = torch.stack(
                                [
                                    _expert_tensor(expert, lid, kind)
                                    for expert in expected_maps[lid][
                                        rank * 4 : (rank + 1) * 4
                                    ]
                                    + [31]
                                ]
                            )
                            torch.testing.assert_close(
                                tensor.cpu(), expected, rtol=0, atol=0
                            )
                    assert weight_pointers == [
                        [tensor.data_ptr() for tensor in tensors]
                        for tensors in routed.values()
                    ]
                    for name, pointer in map_pointers.items():
                        assert getattr(current, name).data_ptr() == pointer, name
                    torch.testing.assert_close(
                        current.physical_to_logical_map_cpu, torch.tensor(expected_maps)
                    )
                    if layer == 1:
                        for name in map_pointers:
                            torch.testing.assert_close(
                                getattr(current, name),
                                getattr(target, name),
                                rtol=0,
                                atol=0,
                            )
                    dist.barrier()
                    updates += 1
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "device": args.device,
                        "weight_format": args.weight_format,
                        "world_size": 2,
                        "layer_updates": updates,
                        "model_loaded": False,
                    }
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
