"""DeepEP auto-mode statistics across prefill and decode batches."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
from sglang.srt.arg_groups.moe_hook import handle_a2a_moe
from sglang.srt.arg_groups.parallel_hook import (
    handle_eplb_and_dispatch,
    handle_expert_distribution_metrics,
)
from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.runtime_context import get_forward, publish
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDeepEPAutoDistribution(unittest.TestCase):
    def setUp(self):
        args = ServerArgs(
            model_path="dummy",
            device="cpu",
            moe_a2a_backend="deepep",
            deepep_mode="auto",
            expert_distribution_recorder_mode="stat",
            expert_distribution_recorder_buffer_size=4,
        )
        publish(args, role="test")
        device_patch = patch.object(distribution, "get_device", return_value="cpu")
        device_patch.start()
        self.addCleanup(device_patch.stop)
        previous = get_forward().is_extend_in_batch
        self.addCleanup(get_forward().set, "is_extend_in_batch", previous)
        self.metadata = SimpleNamespace(
            num_layers=2,
            num_physical_experts=8,
            num_local_physical_experts=4,
        )

    def gatherer(self, rank=1):
        return distribution._SinglePassGatherer.init_new(self.metadata, rank)

    def test_pd_cli_modes_construct_real_gatherers(self):
        # dummy avoids model/hardware setup; execute the actual graph/MoE hooks
        # in pipeline order before publishing the resolved configuration.
        for role in ("prefill", "decode"):
            for mode, expected in (
                ("auto", distribution._DeepepAutoSinglePassGatherer),
                ("normal", distribution._SelectExpertsSinglePassGatherer),
                ("low_latency", distribution._DeepepLowLatencySinglePassGatherer),
            ):
                with self.subTest(role=role, mode=mode):
                    args = prepare_server_args(
                        [
                            "--model-path",
                            "dummy",
                            "--device",
                            "cpu",
                            "--tp-size",
                            "2",
                            "--moe-a2a-backend",
                            "deepep",
                            "--deepep-mode",
                            mode,
                            "--enable-eplb",
                            "--disaggregation-mode",
                            role,
                        ]
                    )
                    args.resolve_once()
                    parse_cuda_graph_config(args)
                    handle_a2a_moe(args)
                    handle_eplb_and_dispatch(args)
                    handle_expert_distribution_metrics(args)
                    publish(args, role="test")
                    self.assertIs(type(self.gatherer()), expected)

    def test_prefill_counts_physical_ids_and_ignores_padding(self):
        gatherer = self.gatherer()
        get_forward().set("is_extend_in_batch", True)
        gatherer.on_select_experts(0, torch.tensor([[0, 7, -1], [0, 2, 6]]))
        gatherer.on_select_experts(0, torch.tensor([[7, -1, 2]]))
        expected = torch.tensor([[2, 0, 2, 0, 0, 0, 1, 2], [0] * 8])
        torch.testing.assert_close(
            gatherer.collect()["global_physical_count"], expected, check_dtype=False
        )

    def test_decode_counts_receiving_rank_without_double_counting_topk(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                gatherer = self.gatherer(rank)
                get_forward().set("is_extend_in_batch", False)
                # The topk hook runs before low-latency dispatch, too. Its
                # sending-side ids must not be added to receiving-side counts.
                gatherer.on_select_experts(0, torch.tensor([[0, 0, 1], [4, 7, -1]]))
                gatherer.on_deepep_dispatch_low_latency(0, torch.tensor([3, 0, 2, 1]))
                expected = torch.zeros(2, 8, dtype=torch.int32)
                expected[0, 4 * rank : 4 * (rank + 1)] = torch.tensor([3, 0, 2, 1])
                torch.testing.assert_close(
                    gatherer.collect()["global_physical_count"], expected
                )

    def test_accumulation_across_modes_layers_and_reset(self):
        gatherer = self.gatherer()
        get_forward().set("is_extend_in_batch", True)
        gatherer.on_select_experts(0, torch.tensor([[0, 6]]))
        get_forward().set("is_extend_in_batch", False)
        gatherer.on_select_experts(0, torch.tensor([[4, 4]]))
        gatherer.on_deepep_dispatch_low_latency(0, torch.tensor([2, 0, 0, 0]))
        gatherer.on_deepep_dispatch_low_latency(1, torch.tensor([0, 1, 0, 2]))
        expected = torch.tensor([[1, 0, 0, 0, 2, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 2]])
        torch.testing.assert_close(
            gatherer.collect()["global_physical_count"], expected, check_dtype=False
        )
        # Reading a completed pass must not add its counts again.
        torch.testing.assert_close(
            gatherer.collect()["global_physical_count"], expected, check_dtype=False
        )
        gatherer.reset()
        torch.testing.assert_close(
            gatherer.collect()["global_physical_count"],
            torch.zeros(2, 8, dtype=torch.int32),
        )
        # An empty local batch on a DP rank still follows the DP-wide mode.
        get_forward().set("is_extend_in_batch", True)
        gatherer.on_select_experts(1, torch.empty(0, 2, dtype=torch.int64))
        torch.testing.assert_close(
            gatherer.collect()["global_physical_count"],
            torch.zeros(2, 8, dtype=torch.int32),
        )

    def test_recorder_buffers_prefill_and_decode_in_one_window(self):
        recorder = distribution.ExpertDistributionRecorder.init_new(
            self.metadata, rank=1
        )
        recorder.start_record()
        get_forward().set("is_extend_in_batch", True)
        with recorder.with_forward_pass(0, SimpleNamespace()):
            with recorder.with_current_layer(0):
                recorder.on_select_experts(torch.tensor([[0, 7]]))
        get_forward().set("is_extend_in_batch", False)
        with recorder.with_forward_pass(1, SimpleNamespace()):
            with recorder.with_current_layer(0):
                recorder.on_select_experts(torch.tensor([[4, 4]]))
                recorder.on_deepep_dispatch_low_latency(torch.tensor([2, 0, 0, 0]))
        counts = recorder._accumulator._global_physical_count_of_buffered_step.get_all()
        # The fixed buffer also contains unused zero rows.
        torch.testing.assert_close(
            counts.sum(dim=0),
            torch.tensor([[1, 0, 0, 0, 2, 0, 0, 1], [0] * 8], dtype=torch.int64),
        )

    def test_disabled_eplb_keeps_recorder_noop(self):
        args = ServerArgs(model_path="dummy", device="cpu", moe_a2a_backend="deepep")
        publish(args, role="test")
        recorder = distribution.ExpertDistributionRecorder.init_new(
            self.metadata, rank=1
        )
        self.assertIsInstance(recorder, distribution._ExpertDistributionRecorderNoop)
        self.assertFalse(recorder.recording)


if __name__ == "__main__":
    unittest.main()
