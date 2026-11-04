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
    get_mega_moe_sym_buffer,
    mega_moe_activation_args,
    mega_moe_mxfp_weights,
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
    def test_mega_moe_retries_uninitialized_ep_comm_without_caching(self):
        """An unavailable EP communicator must not publish a reusable buffer."""
        module = "sglang.srt.hardware_backend.npu.moe.mega_moe"
        cache = {}
        ready = {"initialized": False}
        comm_names = iter(["", "hccl_ep_comm"])
        backend = Mock()
        group = Mock()
        group._get_backend.return_value = backend
        buffer = object()

        def get_comm_name(rank, *, init_comm):
            self.assertEqual(rank, 0)
            self.assertTrue(init_comm)
            name = next(comm_names)
            ready["initialized"] = bool(name)
            return name

        backend.get_hccl_comm_name.side_effect = get_comm_name

        def create_buffer(actual_group, **kwargs):
            self.assertIs(actual_group, group)
            self.assertTrue(ready["initialized"])
            return buffer

        provider = ModuleType("sgl_kernel_npu.modules.moe")
        provider.get_symm_buffer_for_mega_moe = Mock(side_effect=create_buffer)
        config = SimpleNamespace(
            num_experts=256,
            top_k=8,
            hidden_size=6144,
            intermediate_size_per_partition=2048,
        )
        cpu_device = torch.device("cpu")
        npu = SimpleNamespace(
            current_device=lambda: 0,
            synchronize=Mock(side_effect=AssertionError("Unexpected device sync")),
        )
        with (
            patch.dict("sys.modules", {provider.__name__: provider}),
            patch(f"{module}._SYMM_BUFFER_CACHE", cache),
            patch(f"{module}.torch.npu", npu, create=True),
            patch(f"{module}.torch.device", return_value=cpu_device),
            patch(
                f"{module}.torch.distributed.all_reduce",
                side_effect=AssertionError("Unexpected device collective"),
            ),
            patch(f"{module}.torch.distributed.get_rank", return_value=0),
        ):
            with self.assertRaisesRegex(RuntimeError, "comm name is empty"):
                get_mega_moe_sym_buffer(config, {}, group, 3, "normal")
            self.assertEqual(cache, {})
            self.assertEqual(provider.get_symm_buffer_for_mega_moe.call_count, 0)

            self.assertIs(
                get_mega_moe_sym_buffer(config, {}, group, 3, "normal"), buffer
            )
            ready["initialized"] = False
            self.assertIs(
                get_mega_moe_sym_buffer(config, {}, group, 2, "normal"), buffer
            )
            self.assertEqual(ready, {"initialized": False})
            self.assertEqual(provider.get_symm_buffer_for_mega_moe.call_count, 1)

    def _dispatcher_for_quant_mode(self, quant_config=None):
        dispatcher = AscendOpsDispatcher.__new__(AscendOpsDispatcher)
        dispatcher.quant_config = quant_config or {}
        dispatcher.tp_group = object()
        dispatcher.ep_group = object()
        dispatcher.moe_ascend_ops_config = AscendOpsConfig()
        return dispatcher

    def test_fused_op_selects_group_for_prefill_and_decode(self):
        module = "sglang.srt.layers.moe.token_dispatcher.ascend_ops"
        hidden_states = torch.zeros((3, 4))
        topk_output = (
            torch.ones((3, 2)),
            torch.zeros((3, 2), dtype=torch.int32),
            None,
        )
        cases = (
            ("fused_op", "low_latency", True, "tp_group"),
            ("allgather", "fused_op", False, "ep_group"),
            ("fused_op", "fused_op", True, "ep_group"),
            ("fused_op", "fused_op", False, "ep_group"),
        )
        for prefill, decode, is_extend, group_name in cases:
            with (
                self.subTest(prefill=prefill, decode=decode, is_extend=is_extend),
                patch(f"{module}.get_is_extend_in_batch", return_value=is_extend),
                patch(f"{module}.is_dp_attention_enabled", return_value=False),
                patch(
                    f"{module}.get_parallel",
                    return_value=SimpleNamespace(attn_dp_size=1, attn_cp_size=1),
                ),
                patch(f"{module}.dist.get_world_size", return_value=16),
            ):
                dispatcher = self._dispatcher_for_quant_mode()
                dispatcher.moe_ascend_ops_config = AscendOpsConfig(
                    prefill=prefill, decode=decode
                )
                output = dispatcher.dispatch(hidden_states, topk_output)
                self.assertIs(
                    output.forward_state.group, getattr(dispatcher, group_name)
                )
                self.assertEqual(
                    output.forward_state.dtype_mode,
                    "normal" if is_extend else "low_latency",
                )

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
            dispatcher._resolve_quant_mode("mc2", "low_latency"),
            4,
        )
        self.assertEqual(dispatcher._resolve_quant_mode("routing"), -1)

    # Protects MC2 from receiving a routing-only dispatcher dtype it cannot
    # encode in its own quantization enum.
    def test_unsupported_mc2_dtype_is_rejected(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {"dispatcher_output_dtype": DispatcherOutputDtype.FP8.value}
        )
        with self.assertRaisesRegex(NotImplementedError, "mc2.*fp8"):
            dispatcher._resolve_quant_mode("mc2")

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

    def test_mxfp_views_preserve_expert_axes_and_scale_bits(self):
        """A5 must see stacked expert scales as E8M0, not flattened UINT64."""
        w13 = torch.arange(2 * 8 * 64).reshape(2, 8, 64).to(torch.float8_e4m3fn)
        w2 = torch.ones((2, 64, 64), dtype=torch.float8_e4m3fn)
        s13 = torch.arange(2 * 8 * 2, dtype=torch.uint8).reshape(2, 8, 1, 2)
        s2 = torch.full((2, 64, 1, 2), 127, dtype=torch.uint8)
        layer = SimpleNamespace(
            w13_weight=w13.transpose(1, 2),
            w2_weight=w2.transpose(1, 2),
            w13_weight_scale=s13.transpose(1, 2),
            w2_weight_scale=s2.transpose(1, 2),
        )
        weights1, weights2, scales1, scales2 = mega_moe_mxfp_weights(layer)
        for result, expected in ((weights1, w13), (weights2, w2)):
            self.assertEqual(len(result), 1)
            torch.testing.assert_close(result[0].float(), expected.float())
            self.assertEqual(result[0].data_ptr(), expected.data_ptr())
        for result, expected in ((scales1, s13), (scales2, s2)):
            self.assertEqual(result[0].dtype, torch.float8_e8m0fnu)
            self.assertEqual(result[0].data_ptr(), expected.data_ptr())
            torch.testing.assert_close(result[0].view(torch.uint8), expected)
        # Source parameters must still serve standard prefill GMMs.
        self.assertEqual(layer.w13_weight.shape, (2, 64, 8))
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        s13[0, 0, 0, 0] = 129
        self.assertEqual(scales1[0].view(torch.uint8)[0, 0, 0, 0].item(), 129)

    def test_a5_buffer_uses_type_code_abi_and_reuses_across_batch_sizes(self):
        """PR #9569's receive capacity must not depend on local token count."""
        module = "sglang.srt.hardware_backend.npu.moe.mega_moe"
        provider = ModuleType("npu_ops_transformer.ops")
        buffers = []

        def create_buffer(
            group,
            *,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            max_recv_token_num,
            dispatch_quant_mode,
            dispatch_quant_out_type,
        ):
            self.assertEqual(num_max_tokens_per_rank, 0)
            self.assertEqual(intermediate_hidden, 4096)
            self.assertEqual(max_recv_token_num, 196608)
            self.assertEqual(dispatch_quant_mode, 4)
            self.assertEqual(dispatch_quant_out_type, 24)
            buffers.append(object())
            return buffers[-1]

        provider.get_symm_buffer_for_mega_moe = create_buffer
        backend = Mock()
        backend.get_hccl_comm_name.return_value = "ep_comm"
        group = Mock()
        group._get_backend.return_value = backend
        config = SimpleNamespace(
            num_experts=256,
            top_k=8,
            hidden_size=6144,
            intermediate_size_per_partition=2048,
        )
        with (
            patch.dict("sys.modules", {provider.__name__: provider}),
            patch(f"{module}._SYMM_BUFFER_CACHE", {}),
            patch(
                f"{module}.torch.npu",
                SimpleNamespace(current_device=lambda: 0),
                create=True,
            ),
            patch(f"{module}.torch.device", return_value=torch.device("cpu")),
            patch(f"{module}.torch.distributed.get_rank", return_value=0),
        ):
            first = get_mega_moe_sym_buffer(
                config,
                {"normal_dispatcher_output_dtype": "bf16"},
                group,
                1,
                "normal",
                mxfp=True,
            )
            second = get_mega_moe_sym_buffer(
                config,
                {"low_latency_dispatcher_output_dtype": "mxfp8"},
                group,
                128,
                "low_latency",
                mxfp=True,
            )
            self.assertIs(first, second)
            self.assertEqual(len(buffers), 1)

    def test_a5_buffer_accepts_dtype_abi(self):
        """New ops-transformer releases name the dtype argument differently."""
        module = "sglang.srt.hardware_backend.npu.moe.mega_moe"
        provider = ModuleType("npu_ops_transformer.ops")
        buffer = object()

        def create_buffer(
            group,
            *,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            max_recv_token_num,
            dispatch_quant_mode,
            dispatch_quant_out_dtype,
        ):
            self.assertEqual(num_max_tokens_per_rank, 16)
            self.assertEqual(intermediate_hidden, 2048)
            self.assertEqual(dispatch_quant_mode, 4)
            self.assertEqual(dispatch_quant_out_dtype, torch.float8_e4m3fn)
            return buffer

        provider.get_symm_buffer_for_mega_moe = create_buffer
        backend = Mock()
        backend.get_hccl_comm_name.return_value = "ep_comm"
        group = Mock()
        group._get_backend.return_value = backend
        config = SimpleNamespace(
            num_experts=256,
            top_k=8,
            hidden_size=6144,
            intermediate_size_per_partition=2048,
        )
        with (
            patch.dict("sys.modules", {provider.__name__: provider}),
            patch(f"{module}._SYMM_BUFFER_CACHE", {}),
            patch(
                f"{module}.torch.npu",
                SimpleNamespace(current_device=lambda: 0),
                create=True,
            ),
            patch(f"{module}.torch.device", return_value=torch.device("cpu")),
            patch(f"{module}.torch.distributed.get_rank", return_value=0),
        ):
            self.assertIs(
                get_mega_moe_sym_buffer(config, {}, group, 16, mxfp=True), buffer
            )
            self.assertIs(
                get_mega_moe_sym_buffer(config, {}, group, 8, mxfp=True), buffer
            )

    def test_a5_mxfp_forward_matches_expert_reference(self):
        """The fused boundary must preserve expert axes and decode E8M0 bytes."""
        provider = ModuleType("npu_ops_transformer.ops")

        def mega_moe(x, ids, probs, l1, l2, buffer, **kwargs):
            # CPU stand-in for the external operator's documented stacked ABI.
            w1, w2 = l1[0].float(), l2[0].float()
            s1 = kwargs["l1_weights_sf"][0].float().flatten(-2)
            s2 = kwargs["l2_weights_sf"][0].float().flatten(-2)
            w1 = w1 * s1.repeat_interleave(32, dim=-1)
            w2 = w2 * s2.repeat_interleave(32, dim=-1)
            output = torch.zeros_like(x)
            for token in range(x.shape[0]):
                for k in range(ids.shape[1]):
                    expert = ids[token, k].item()
                    gate, up = (x[token] @ w1[expert].T).chunk(2)
                    output[token] += (
                        probs[token, k]
                        * (torch.nn.functional.silu(gate) * up)
                        @ w2[expert].T
                    )
            return output, None

        provider.mega_moe = mega_moe
        w1 = torch.full((2, 128, 64), 0.25, dtype=torch.float8_e4m3fn)
        w2 = torch.full((2, 64, 64), 0.5, dtype=torch.float8_e4m3fn)
        s1 = torch.full((2, 128, 1, 2), 127, dtype=torch.uint8)
        s2 = torch.full((2, 64, 1, 2), 128, dtype=torch.uint8)
        layer = SimpleNamespace(
            w13_weight=w1.transpose(1, 2),
            w2_weight=w2.transpose(1, 2),
            w13_weight_scale=s1.transpose(1, 2),
            w2_weight_scale=s2.transpose(1, 2),
        )
        config = SimpleNamespace(
            layer=layer,
            activation="silu",
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        )
        x = torch.full((1, 64), 0.125)
        module = "sglang.srt.hardware_backend.npu.moe.mega_moe"
        with (
            patch.dict("sys.modules", {provider.__name__: provider}),
            patch(f"{module}.check_mega_moe_kernel", return_value=True),
            patch(f"{module}.get_mega_moe_sym_buffer", return_value=object()),
        ):
            result = run_mega_moe(
                config,
                {},
                x,
                torch.tensor([[0.25, 0.75]]),
                torch.tensor([[1, 0]]),
                object(),
            )
        gate = torch.full((1, 64), 2.0)
        expected = (torch.nn.functional.silu(gate) * gate) @ torch.ones((64, 64))
        torch.testing.assert_close(result, expected)

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
        group = object()

        with (
            patch.dict(
                "sys.modules",
                {
                    "sgl_kernel_npu": fake_root,
                    "sgl_kernel_npu.modules": fake_parent,
                    "sgl_kernel_npu.modules.moe": fake_moe_module,
                },
            ),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.moe.mega_moe.check_mega_moe_kernel",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.moe.mega_moe.get_mega_moe_sym_buffer",
                return_value="sym_buffer",
            ) as get_buffer,
        ):
            result = run_mega_moe(
                runner_config,
                {"dispatcher_output_dtype": "bf16"},
                hidden_states,
                topk_weights,
                topk_ids,
                group=group,
            )

        self.assertEqual(result, "output")
        args, kwargs = mega_moe.call_args
        self.assertIs(args[0], hidden_states)
        self.assertIs(args[1], topk_ids)
        self.assertEqual(args[2].dtype, torch.float32)
        self.assertEqual(len(args[3]), 2)
        self.assertEqual(len(args[4]), 2)
        self.assertEqual(args[5], "sym_buffer")
        self.assertIs(get_buffer.call_args.args[2], group)
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
            group=dispatcher.ep_group,
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
    def test_all_gather_row_sizes_use_actual_rank_local_rows(self):
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
            row_sizes = AscendOpsDispatcher._all_gather_row_sizes(
                2, group=None, device=torch.device("cpu")
            )

        self.assertEqual(row_sizes, [2, 3])

    # Protects the DP+attention-TP all-gather fast path from adding a device
    # row-count collective when scheduler metadata already determines rows.
    def test_all_gather_row_sizes_use_dp_metadata_without_collective(self):
        dispatcher = AscendOpsDispatcher.__new__(AscendOpsDispatcher)
        parallel = SimpleNamespace(attn_dp_size=2, attn_cp_size=1, attn_tp_size=2)
        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "get_dp_global_num_tokens",
                return_value=[8, 4],
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=0,
            ),
        ):
            layout = AscendOpsDispatcher._resolve_row_layout(
                dispatcher, 4, group=None, device=torch.device("cpu"), allow_cp=True
            )

        self.assertEqual(layout.row_sizes, (4, 4, 2, 2))

    # Protects the metadata fast path from silently falling back to a device
    # collective when the upstream attention layout is inconsistent.
    def test_allgather_metadata_row_mismatch_is_rejected(self):
        parallel = SimpleNamespace(attn_dp_size=2, attn_cp_size=1, attn_tp_size=2)
        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "get_dp_global_num_tokens",
                return_value=[8, 4],
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=0,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "local rows disagree"):
                AscendOpsDispatcher._resolve_row_layout(
                    AscendOpsDispatcher.__new__(AscendOpsDispatcher),
                    3,
                    group=None,
                    device=torch.device("cpu"),
                    allow_cp=True,
                )

    # Protects MegaMoE capacity from using the unsplit DP token count when
    # attention TP has already divided the local MoE rows.
    def test_fused_op_pads_to_attention_tp_local_capacity(self):
        dispatcher = self._dispatcher_for_quant_mode()
        hidden_states = torch.zeros((2, 4))
        topk_weights = torch.ones((2, 1))
        topk_ids = torch.zeros((2, 1), dtype=torch.int64)
        parallel = SimpleNamespace(
            attn_dp_rank=1,
            attn_dp_size=2,
            attn_tp_rank=1,
            attn_tp_size=2,
            attn_cp_size=1,
        )
        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "is_dp_attention_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "get_dp_global_num_tokens",
                return_value=[8, 4],
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "get_is_extend_in_batch",
                return_value=True,
            ),
        ):
            output = dispatcher._dispatch_fused_op(
                hidden_states, topk_weights, topk_ids
            )

        self.assertEqual(output.hidden_states.shape[0], 4)
        self.assertEqual(output.forward_state.num_valid_rows, 2)

    # Protects alltoallv count exchange from all-gathering the full global
    # histogram when each rank only needs its local-expert slice.
    def test_alltoallv_counts_exchange_local_expert_slices(self):
        dispatcher = self._dispatcher_for_quant_mode()
        dispatcher.num_experts = 4
        dispatcher.num_local_experts = 2
        topk_ids = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32)

        def fake_all_to_all(output, input, **kwargs):
            output.copy_(input)

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist."
                "all_to_all_single",
                side_effect=fake_all_to_all,
            ) as all_to_all,
        ):
            input_splits, output_splits, recv_counts = dispatcher._resolve_all_to_all_counts(
                topk_ids, group=None
            )

        self.assertEqual(input_splits, [2, 2])
        self.assertEqual(output_splits, [2, 2])
        self.assertEqual(tuple(recv_counts.shape), (2, 2))
        all_to_all.assert_called_once()

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
