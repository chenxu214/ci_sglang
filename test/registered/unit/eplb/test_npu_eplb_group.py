"""Only reuse an initialized NPU group with exactly WORLD rank ordering."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.eplb import comm_utils
from sglang.srt.runtime_context import ParallelContext
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNpuEplbGroup(unittest.TestCase):
    def check_group(self, device, ranks, initialized=True):
        group = object()
        with (
            patch.object(
                comm_utils, "get_device", return_value=SimpleNamespace(device=device)
            ),
            patch.object(comm_utils.dist, "is_initialized", return_value=initialized),
            patch.object(comm_utils.dist, "get_world_size", return_value=4),
            patch.object(
                comm_utils,
                "get_tp_group",
                return_value=SimpleNamespace(ranks=ranks, device_group=group),
            ),
        ):
            return comm_utils.get_eplb_device_group(), group

    def test_npu_full_world_reuses_group(self):
        actual, expected = self.check_group("npu", [0, 1, 2, 3])
        self.assertIs(actual, expected)

    def test_subgroups_or_different_rank_order_keep_world(self):
        for ranks in ([0, 1], [2, 3], [1, 0, 2, 3]):
            with self.subTest(ranks=ranks):
                self.assertIsNone(self.check_group("npu", ranks)[0])

    def test_other_devices_and_uninitialized_dist_keep_world(self):
        for device in ("cpu", "cuda"):
            self.assertIsNone(self.check_group(device, [0, 1, 2, 3])[0])
        self.assertIsNone(self.check_group("npu", [0, 1, 2, 3], initialized=False)[0])

    def test_standalone_npu_tool_without_tp_keeps_world(self):
        with (
            patch.object(
                comm_utils, "get_device", return_value=SimpleNamespace(device="npu")
            ),
            patch.object(comm_utils.dist, "is_initialized", return_value=True),
            patch.object(
                comm_utils, "get_tp_group", side_effect=AssertionError("no TP")
            ),
        ):
            self.assertIsNone(comm_utils.get_eplb_device_group())

    def test_unwritten_runtime_tp_group_keeps_world(self):
        with (
            patch.object(comm_utils, "get_device", return_value=SimpleNamespace(device="npu")),
            patch.object(comm_utils.dist, "is_initialized", return_value=True),
            patch("sglang.srt.distributed.parallel_state.get_parallel", return_value=ParallelContext()),
        ):
            self.assertIsNone(comm_utils.get_eplb_device_group())

    def test_other_runtime_errors_are_not_hidden(self):
        with (
            patch.object(comm_utils, "get_device", return_value=SimpleNamespace(device="npu")),
            patch.object(comm_utils.dist, "is_initialized", return_value=True),
            patch.object(comm_utils, "get_tp_group", side_effect=RuntimeError("group setup failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "group setup failed"):
                comm_utils.get_eplb_device_group()


if __name__ == "__main__":
    unittest.main()
