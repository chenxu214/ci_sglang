"""NPU routing wrapper contracts, using CPU tensors and stub device gates."""

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe import hash_topk, initialize_moe_config, topk
from sglang.srt.runtime_context import get_parallel, publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNpuTopKPadding(unittest.TestCase):
    def setUp(self):
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=2,
                ep_size=2,
                moe_a2a_backend="deepep",
                deepep_mode="normal",
                expert_distribution_recorder_mode="stat",
                expert_distribution_recorder_buffer_size=4,
            ),
            role="test",
        )
        initialize_moe_config()
        for module, attr, value in (
            (topk, "_is_npu", True),
            (topk, "_is_cuda", False),
            (hash_topk, "_is_npu", True),
            (hash_topk, "_is_xpu", False),
            (distribution, "get_device", "cpu"),
        ):
            item = (
                patch.object(module, attr, return_value=value)
                if callable(getattr(module, attr))
                else patch.object(module, attr, value)
            )
            item.start()
            self.addCleanup(item.stop)
        # No sgl_kernel_npu installation or device execution is needed by CPU CI.
        norm = types.ModuleType("sgl_kernel_npu.norm.l1_norm")
        norm.l1_norm = lambda x: x / x.abs().sum(dim=-1, keepdim=True)
        with patch.dict(sys.modules, {norm.__name__: norm}):
            self.npu_topk = importlib.import_module(
                "sglang.srt.hardware_backend.npu.moe.topk"
            )
        self.mapping = torch.tensor([6, 0, 7, 1], dtype=torch.int32)
        self.info = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=self.mapping,
            partial_logical_to_all_physical_map=torch.arange(4).reshape(4, 1),
            partial_logical_to_all_physical_map_num_valid=torch.ones(4),
            num_physical_experts=8,
        )
        self.hidden = torch.zeros(4, 1)
        self.logits = torch.zeros(4, 4)
        self.ids = torch.tensor([[0, 1], [1, 0], [3, 2], [2, 3]], dtype=torch.int32)

    def recorder(self):
        result = distribution.ExpertDistributionRecorder.init_new(
            SimpleNamespace(
                num_layers=1, num_physical_experts=8, num_local_physical_experts=4
            ),
            rank=0,
        )
        distribution.set_global_expert_distribution_recorder(result)
        result.start_record()
        return result

    def gate(self, *args, **kwargs):
        return torch.full((4, 2), 0.5), self.ids.clone(), None

    def route(self, mode, valid, recorder):
        config = topk.TopKConfig(
            top_k=2,
            scoring_func=mode,
            renormalize=False,
            correction_bias=torch.zeros(4) if mode == "sigmoid" else None,
        )
        with (
            patch.object(
                torch.ops.custom,
                "npu_moe_gating_top_k",
                side_effect=self.gate,
                create=True,
            ),
            patch.object(
                torch.ops.npu,
                "npu_moe_gating_top_k_softmax",
                side_effect=self.gate,
                create=True,
            ),
            patch.object(
                torch.ops.npu,
                "npu_moe_gating_top_k",
                side_effect=self.gate,
                create=True,
            ),
            recorder.with_forward_pass(1, SimpleNamespace()),
            recorder.with_current_layer(0),
        ):
            return self.npu_topk.fused_topk_npu(
                self.hidden,
                self.logits,
                config,
                num_token_non_padded=None if valid is None else torch.tensor([valid]),
                expert_location_dispatch_info=self.info,
                layer_id=0,
            )

    def test_fast_paths_mask_dispatch_and_real_recorder_counts(self):
        for mode in ("sqrtsoftplus", "softmax", "sigmoid"):
            for valid in (None, 0, 2, 4):
                with self.subTest(mode=mode, valid=valid):
                    recorder = self.recorder()
                    output = self.route(mode, valid, recorder)
                    n = 4 if valid is None else valid
                    expected = self.mapping[self.ids.long()]
                    expected[n:] = -1
                    torch.testing.assert_close(output.topk_ids, expected)
                    self.assertEqual(output.topk_ids.dtype, self.ids.dtype)
                    torch.testing.assert_close(
                        output.topk_weights, torch.full((4, 2), 0.5)
                    )
                    counts = recorder._accumulator._global_physical_count_of_buffered_step.get_all()
                    expected_counts = torch.bincount(
                        expected[:n].flatten().long(), minlength=8
                    )
                    torch.testing.assert_close(counts.sum(dim=(0, 1)), expected_counts)

    def test_padding_is_sanitized_before_expert_map_lookup(self):
        self.ids[2:] = 99
        recorder = self.recorder()
        output = self.route("sqrtsoftplus", 2, recorder)
        self.assertTrue(torch.all(output.topk_ids[2:] == -1))

    def test_device_row_count_and_int64_buffer_are_kept_in_place(self):
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        pointer = ids.data_ptr()
        valid = torch.tensor([1])
        with patch.object(
            torch.Tensor, "item", side_effect=AssertionError("host read")
        ):
            topk._mask_topk_ids_padded_region(ids, valid)
        torch.testing.assert_close(ids, torch.tensor([[0, 1], [-1, -1]]))
        valid.fill_(0)
        topk._mask_topk_ids_padded_region(ids, valid)
        self.assertTrue(torch.all(ids == -1))
        self.assertEqual(pointer, ids.data_ptr())

    def test_other_npu_dispatchers_keep_existing_padding_contract(self):
        ids = self.ids.clone()
        with patch.object(
            topk,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_deepep=lambda: False),
            create=True,
        ):
            topk._mask_topk_ids_padded_region(ids, torch.tensor([0]))
        torch.testing.assert_close(ids, self.ids)

    def test_hash_topk_masks_dispatch_and_excludes_shared_slots_from_counts(self):
        for shared in (0, 1):
            with self.subTest(shared=shared):
                recorder = self.recorder()
                router = hash_topk.HashTopK(
                    topk=2 + shared,
                    num_experts=4,
                    num_fused_shared_experts=shared,
                    vocab_size=4,
                    layer_id=0,
                )
                with (
                    get_parallel().override(moe_ep_size=2, moe_ep_rank=0),
                    patch.object(
                        hash_topk.envs.SGLANG_OPT_USE_FUSED_HASH_TOPK,
                        "get",
                        return_value=False,
                    ),
                    recorder.with_forward_pass(1, SimpleNamespace()),
                    recorder.with_current_layer(0),
                ):
                    output = router(
                        self.hidden,
                        self.logits,
                        input_ids=torch.arange(4),
                        num_token_non_padded=torch.tensor([2]),
                        expert_location_dispatch_info=self.info,
                    )
                self.assertTrue(torch.all(output.topk_ids[2:] == -1))
                counts = recorder._accumulator._global_physical_count_of_buffered_step.get_all()
                expected = torch.bincount(
                    self.mapping[torch.tensor([[0, 1], [1, 2]])].flatten().long(),
                    minlength=8,
                )
                torch.testing.assert_close(counts.sum(dim=(0, 1)), expected)


if __name__ == "__main__":
    unittest.main()
