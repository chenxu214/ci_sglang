"""CPU unit tests for the Ascend MoE dispatcher contracts.

These tests deliberately stop at the Python dispatcher boundary.  The
``torch_npu`` kernels themselves require an Ascend runner and are covered by
hardware integration tests when an NPU job is available.
"""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.moe.mega_moe import (
    mega_moe_activation_args,
    mega_moe_quant_args,
    mega_moe_weight_list,
    run_mega_moe,
)
from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
    AscendOpsCombineInput,
    AscendOpsDispatcher,
    _AllGatherState,
    _AllToAllVState,
    _MegaMoEState,
    _MC2State,
)
from sglang.srt.layers.moe.utils import AscendOpsConfig, DispatcherOutputDtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAscendOpsConfig(CustomTestCase):
    # Protects the configuration boundary for the MegaMoE fused decode path.
    def test_fused_op_is_accepted_during_config_validation(self):
        config = AscendOpsConfig.from_json_str('{"decode":"fused_op"}')
        self.assertEqual(config.decode, "fused_op")

    # Protects the default path used when no Ascend Ops config is supplied.
    def test_low_latency_is_the_default_decode_mode(self):
        config = AscendOpsConfig.from_json_str(None)
        self.assertEqual(config.decode, "low_latency")


class TestAscendOpsDispatcherContracts(CustomTestCase):
    def _dispatcher_for_quant_mode(self, quant_config=None):
        dispatcher = AscendOpsDispatcher.__new__(AscendOpsDispatcher)
        dispatcher.quant_config = quant_config or {}
        return dispatcher

    # Protects mode-specific quantization settings from being ignored in favor
    # of the shared dispatcher dtype.
    def test_mode_specific_dtype_overrides_shared_dtype(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {
                "dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "mxfp8",
            }
        )
        self.assertEqual(
            dispatcher._quant_mode("mc2", "low_latency"),
            4,
        )
        self.assertEqual(dispatcher._quant_mode("routing"), -1)

    # Protects MC2 from receiving a routing-only dispatcher dtype it cannot
    # encode in its own quantization enum.
    def test_unsupported_mc2_dtype_is_rejected(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {"dispatcher_output_dtype": DispatcherOutputDtype.FP8.value}
        )
        with self.assertRaisesRegex(NotImplementedError, "mc2.*fp8"):
            dispatcher._quant_mode("mc2")

    # Protects the MegaMoE interface from using the MC2 quant enum for routing.
    def test_mega_moe_quant_args_follow_dispatch_dtype(self):
        self.assertEqual(
            mega_moe_quant_args({"dispatcher_output_dtype": "bf16"}), (0, None)
        )
        self.assertEqual(
            mega_moe_quant_args({"dispatcher_output_dtype": "int8"}),
            (2, torch.int8),
        )
        mode, dtype = mega_moe_quant_args({"dispatcher_output_dtype": "mxfp8"})
        self.assertEqual(mode, 4)
        self.assertEqual(dtype, torch.float8_e4m3fn)
        self.assertEqual(
            mega_moe_quant_args(
                {
                    "normal_dispatcher_output_dtype": "bf16",
                    "low_latency_dispatcher_output_dtype": "mxfp8",
                },
                "normal",
            ),
            (0, None),
        )

    # Protects model activation settings from being silently changed by fused dispatch.
    def test_mega_moe_activation_args_preserve_silu_clamp(self):
        config = type(
            "Config",
            (),
            {
                "activation": "silu",
                "gemm1_clamp_limit": 7.0,
                "gemm1_alpha": None,
                "swiglu_limit": None,
            },
        )()
        self.assertEqual(mega_moe_activation_args(config), ("swiglustep", 7.0, None))

    # Protects stacked expert weights from being passed as one expert tensor.
    def test_mega_moe_weight_list_unstacks_expert_dimension(self):
        weights = torch.zeros((2, 3, 4))
        result = mega_moe_weight_list(weights)
        self.assertEqual(len(result), 2)
        self.assertEqual(tuple(result[0].shape), (3, 4))

    # Protects four-dimensional MXFP expert scales from being passed as one
    # packed list item instead of one item per expert.
    def test_mega_moe_weight_list_unstacks_packed_scales(self):
        scales = torch.zeros((2, 3, 4, 2))
        result = mega_moe_weight_list(scales, expected_len=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(tuple(result[0].shape), (3, 4, 2))

    # Protects the MegaMoE call boundary and the argument layout passed to the
    # registered sgl-kernel-npu operator.
    def test_mega_moe_dispatch_calls_registered_operator(self):
        mega_moe = Mock(return_value=("output", "aux"))
        fake_moe_module = ModuleType("sgl_kernel_npu.modules.moe")
        fake_moe_module.mega_moe = mega_moe
        fake_parent = ModuleType("sgl_kernel_npu.modules")
        fake_parent.moe = fake_moe_module
        fake_root = ModuleType("sgl_kernel_npu")
        fake_root.modules = fake_parent

        runner_config = SimpleNamespace(
            num_experts=2,
            top_k=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            layer=SimpleNamespace(
                w13_weight=torch.zeros((2, 3, 4)),
                w2_weight=torch.zeros((2, 4, 3)),
                w13_weight_scale=torch.zeros((2, 3, 4, 2)),
                w2_weight_scale=torch.zeros((2, 4, 3, 2)),
            ),
            activation="silu",
            gemm1_clamp_limit=None,
            gemm1_alpha=None,
            swiglu_limit=None,
        )
        hidden_states = torch.zeros((1, 4))
        topk_weights = torch.ones((1, 2))
        topk_ids = torch.zeros((1, 2), dtype=torch.int32)

        with patch.dict(
            "sys.modules",
            {
                "sgl_kernel_npu": fake_root,
                "sgl_kernel_npu.modules": fake_parent,
                "sgl_kernel_npu.modules.moe": fake_moe_module,
            },
        ), patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=False,
        ), patch(
            "sglang.srt.hardware_backend.npu.moe.mega_moe.check_mega_moe_kernel",
        ), patch(
            "sglang.srt.hardware_backend.npu.moe.mega_moe.get_mega_moe_sym_buffer",
            return_value="sym_buffer",
        ), patch(
            "sglang.srt.runtime_context.get_parallel",
        ):
            result = run_mega_moe(
                runner_config,
                {"dispatcher_output_dtype": "bf16"},
                hidden_states,
                topk_weights,
                topk_ids,
            )

        self.assertEqual(result, "output")
        args, kwargs = mega_moe.call_args
        self.assertIs(args[0], hidden_states)
        self.assertIs(args[1], topk_ids)
        self.assertEqual(args[2].dtype, torch.float32)
        self.assertEqual(len(args[3]), 2)
        self.assertEqual(len(args[4]), 2)
        self.assertEqual(args[5], "sym_buffer")
        self.assertEqual(len(kwargs["l1_weights_sf"]), 2)
        self.assertEqual(len(kwargs["l2_weights_sf"]), 2)
        self.assertEqual(kwargs["activation"], "swiglu")
        self.assertNotIn("activation_params", kwargs)

    # Protects MegaMoE's forward state from being confused with communication
    # states when combine receives the runner output.
    def test_mega_moe_forward_state_is_a_valid_combine_state(self):
        dispatcher = self._dispatcher_for_quant_mode()
        hidden_states = torch.zeros((1, 4))
        state = _MegaMoEState(
            topk_weights=torch.ones((1, 2)),
            topk_ids=torch.zeros((1, 2), dtype=torch.int32),
        )

        with patch.object(
            dispatcher, "_combine_megamoe", return_value=hidden_states
        ) as combine:
            result = dispatcher.combine(AscendOpsCombineInput(hidden_states, state))

        self.assertIs(result, hidden_states)
        combine.assert_called_once_with(hidden_states, state)

    # Protects dispatch from executing MegaMoE before the normal runner hooks
    # have received the forward state.
    def test_mega_moe_dispatch_builds_forward_state(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {"dispatcher_output_dtype": "bf16"}
        )
        hidden_states = torch.zeros((1, 4))
        topk_weights = torch.ones((1, 2))
        topk_ids = torch.zeros((1, 2), dtype=torch.int64)

        output = dispatcher._dispatch_fused_op(
            hidden_states, topk_weights, topk_ids
        )

        self.assertEqual(output.exec_mode, "fused_op")
        self.assertIsInstance(output.forward_state, _MegaMoEState)
        self.assertIs(output.forward_state.topk_weights, output.topk_weights)
        self.assertEqual(output.topk_weights.dtype, torch.float32)
        self.assertEqual(output.forward_state.topk_ids.dtype, torch.int32)
        self.assertEqual(output.forward_state.dtype_mode, "low_latency")

    # Protects AllGather row metadata when attention CP/DP creates uneven shards.
    def test_allgather_row_sizes_use_actual_rank_local_rows(self):
        def fake_all_gather(outputs, local_rows, group):
            outputs[0].copy_(local_rows)
            outputs[1].fill_(3)

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.all_gather",
                side_effect=fake_all_gather,
            ),
        ):
            row_sizes = AscendOpsDispatcher._allgather_row_sizes(
                2, group=None, device=torch.device("cpu")
            )

        self.assertEqual(row_sizes, [2, 3])

    # Protects the forward-state handoff: combine must work without a cached
    # dispatch result stored on the dispatcher instance.
    def test_combine_uses_forward_state_instead_of_dispatcher_state(self):
        dispatcher = self._dispatcher_for_quant_mode()
        state = _MC2State(
            num_tokens=1,
            assist_info_for_combine=torch.empty(0),
            ep_send_counts=torch.empty(0),
            x_active_mask=None,
            expand_scales=None,
            topk_ids=torch.zeros((1, 1), dtype=torch.int32),
            topk_weights=torch.ones((1, 1)),
        )
        combine_input = AscendOpsCombineInput(
            hidden_states=torch.zeros((1, 4)),
            forward_state=state,
        )
        with patch.object(
            dispatcher, "_combine_mc2", return_value="combined"
        ) as combine:
            result = dispatcher.combine(combine_input)

        self.assertEqual(result, "combined")
        combine.assert_called_once_with(combine_input.hidden_states, state)

    # Protects the explicit state-to-combine mapping for both prefill paths.
    def test_combine_dispatches_all_prefill_state_types(self):
        dispatcher = self._dispatcher_for_quant_mode()
        hidden_states = torch.zeros((1, 4))
        states = (
            (_AllGatherState(None, [1], torch.bfloat16), "allgather"),
            (
                _AllToAllVState(
                    None,
                    [1],
                    [1],
                    torch.ones((1, 1)),
                    torch.zeros((1, 1), dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    torch.Size((1, 4)),
                ),
                "alltoallv",
            ),
        )
        for state, mode in states:
            with self.subTest(mode=mode):
                combine_name = f"_combine_{mode}"
                with patch.object(
                    dispatcher, combine_name, return_value=mode
                ) as combine:
                    result = dispatcher.combine(
                        AscendOpsCombineInput(hidden_states, state)
                    )
                self.assertEqual(result, mode)
                combine.assert_called_once_with(hidden_states, state)


if __name__ == "__main__":
    unittest.main()
