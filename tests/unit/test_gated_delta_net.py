import unittest

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")
from types import SimpleNamespace

import torch.nn.functional as F

from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.utils.context import Context, use_context


def make_config():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )


def make_states(layer):
    return (
        torch.zeros(
            layer.conv_dim,
            layer.conv_kernel_size - 1,
        ),
        torch.zeros(
            layer.num_v_heads,
            layer.head_v_dim,
            layer.head_k_dim,
            dtype=torch.float32,
        ),
    )


def reference_forward_sequence(
    layer,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> torch.Tensor:
    mixed_qkv, z, b, a = layer._project_inputs(hidden_states)
    beta, decay = layer._gates(b, a)
    mixed_qkv = layer._causal_conv(mixed_qkv, conv_state)
    query, key, value = mixed_qkv.split(
        [layer.key_dim, layer.key_dim, layer.value_dim],
        dim=-1,
    )
    query = query.view(-1, layer.num_k_heads, layer.head_k_dim)
    key = key.view(-1, layer.num_k_heads, layer.head_k_dim)
    value = value.view(-1, layer.num_v_heads, layer.head_v_dim)
    core = layer._delta_rule(
        query,
        key,
        value,
        decay,
        beta,
        recurrent_state,
    )
    core = layer.norm(
        core.reshape(-1, layer.head_v_dim),
        z.reshape(-1, layer.head_v_dim),
    )
    return layer.out_proj(core.view(-1, layer.value_dim))


def run_with_context(
    layer,
    hidden_states: torch.Tensor,
    is_prefill: bool,
    **kwargs,
) -> torch.Tensor:
    with use_context(Context(is_prefill=is_prefill, **kwargs)):
        return layer(hidden_states)


class GatedDeltaNetStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layer = GatedDeltaNet(make_config())
        with torch.no_grad():
            self.layer.A_log.zero_()

    def test_fresh_prefill_clears_reused_slot(self):
        hidden_states = torch.randn(
            5,
            self.layer.hidden_size,
        )

        expected_conv, expected_recurrent = make_states(self.layer)
        expected = reference_forward_sequence(
            self.layer,
            hidden_states,
            expected_conv,
            expected_recurrent,
        )

        self.layer.state_pool.allocate_state_cache(1)
        self.layer.state_pool.conv_state[0].fill_(7)
        self.layer.state_pool.recurrent_state[0].fill_(11)

        actual = run_with_context(
            self.layer,
            hidden_states,
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, 5),
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_convolution_chunk_state_matches_full_sequence(self):
        inputs = torch.randn(9, self.layer.conv_dim)
        start_state = torch.randn(self.layer.conv_dim, self.layer.conv_kernel_size - 1)
        full = torch.cat((start_state, inputs.T), dim=-1).unsqueeze(0)
        expected = (
            F.silu(F.conv1d(full, self.layer.conv1d.weight, groups=self.layer.conv_dim))
            .squeeze(0)
            .T
        )
        state = start_state.clone()
        chunks = [
            self.layer._causal_conv(chunk, state)
            for chunk in (inputs[:1], inputs[1:3], inputs[3:6], inputs[6:])
        ]
        torch.testing.assert_close(torch.cat(chunks), expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            state, inputs[-(self.layer.conv_kernel_size - 1) :].T, rtol=0, atol=0
        )

    def test_variable_length_batch_keeps_state_aligned(self):
        first = torch.randn(6, self.layer.hidden_size)
        second = torch.randn(4, self.layer.hidden_size)
        first_conv, first_state = make_states(self.layer)
        second_conv, second_state = make_states(self.layer)
        expected_first = reference_forward_sequence(
            self.layer,
            first,
            first_conv,
            first_state,
        )
        expected_second = reference_forward_sequence(
            self.layer,
            second,
            second_conv,
            second_state,
        )

        self.layer.state_pool.allocate_state_cache(2)
        first_chunk = run_with_context(
            self.layer,
            first[:2],
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, 2),
        )
        packed = torch.cat((first[2:], second[:3]))
        mixed_chunk = run_with_context(
            self.layer,
            packed,
            True,
            state_slots=(0, 1),
            state_prefix_lens=(2, 0),
            prefill_q_offsets=(0, 4, 7),
        )

        for actual, expected in (
            (first_chunk, expected_first[:2]),
            (mixed_chunk[:4], expected_first[2:]),
            (mixed_chunk[4:], expected_second[:3]),
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_vectorized_decode_matches_per_slot_reference(self):
        self.layer.state_pool.allocate_state_cache(3)
        with torch.no_grad():
            for state in (
                self.layer.state_pool.conv_state,
                self.layer.state_pool.recurrent_state,
            ):
                state.copy_(torch.randn_like(state))

        expected_conv = self.layer.state_pool.conv_state.clone()
        expected_state = self.layer.state_pool.recurrent_state.clone()
        inputs = torch.randn(2, self.layer.hidden_size)
        slots = (2, 0)
        expected = torch.cat(
            [
                reference_forward_sequence(
                    self.layer,
                    inputs[row : row + 1],
                    expected_conv[slot],
                    expected_state[slot],
                )
                for row, slot in enumerate(slots)
            ]
        )
        actual = run_with_context(
            self.layer, inputs, False, state_slot_ids=torch.tensor(slots, dtype=torch.int64)
        )

        for actual_tensor, expected_tensor in (
            (actual, expected),
            (self.layer.state_pool.conv_state, expected_conv),
            (self.layer.state_pool.recurrent_state, expected_state),
        ):
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                rtol=1e-5,
                atol=1e-5,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_convolution_matches_reference_across_prefill_and_decode(self):
        config = make_config()
        config.linear_value_head_dim = 4  # Channel-last CUDA convolution layout.
        layer = GatedDeltaNet(config).to(device="cuda", dtype=torch.float16)
        layer.state_pool.allocate_state_cache(2)
        with torch.no_grad():
            layer.A_log.zero_()
            layer.state_pool.conv_state[0].copy_(torch.randn_like(layer.state_pool.conv_state[0]))
            for length in (1, 2, 3, 4, 7, 65):
                inputs = torch.randn(length, layer.conv_dim, device="cuda", dtype=torch.float16)
                state = layer.state_pool.conv_state[0]
                full = torch.cat((state.unsqueeze(0), inputs.T.unsqueeze(0)), dim=-1)
                expected = F.conv1d(full, layer.conv1d.weight, groups=layer.conv_dim)
                expected = F.silu(expected[:, :, -length:]).squeeze(0).T
                expected_state = full[0, :, -(layer.conv_kernel_size - 1) :]
                actual = layer._causal_conv(inputs, state)
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(state, expected_state, atol=0, rtol=0)

            hidden = torch.randn(2, layer.hidden_size, device="cuda", dtype=torch.float16)
            slots = (1, 0)
            ref_conv = layer.state_pool.conv_state.clone()
            ref_recurrent = layer.state_pool.recurrent_state.clone()
            expected = torch.cat(
                [
                    reference_forward_sequence(
                        layer, hidden[i : i + 1], ref_conv[slot], ref_recurrent[slot]
                    )
                    for i, slot in enumerate(slots)
                ]
            )
            actual = run_with_context(
                layer,
                hidden,
                False,
                state_slot_ids=torch.tensor(slots, device="cuda", dtype=torch.int32),
                state_slot_indices=torch.tensor(slots, device="cuda", dtype=torch.int64),
            )
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(layer.state_pool.conv_state, ref_conv, atol=0, rtol=0)

    def test_snapshot_resume_matches_uninterrupted_continuation(self):
        inputs = torch.randn(9, self.layer.hidden_size)
        split = 5
        self.layer.state_pool.allocate_state_cache(1)
        run_with_context(
            self.layer,
            inputs[:split],
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, split),
        )
        expected_conv = self.layer.state_pool.conv_state[0].clone()
        expected_state = self.layer.state_pool.recurrent_state[0].clone()
        snapshot = self.layer.state_pool.snapshot_state_slot(0)
        self.assertEqual(snapshot[0].device.type, "cpu")
        self.assertEqual(snapshot[1].device.type, "cpu")
        self.assertEqual(snapshot[1].dtype, torch.float32)
        expected = reference_forward_sequence(
            self.layer,
            inputs[split:],
            expected_conv,
            expected_state,
        )

        self.layer.state_pool.conv_state[0].zero_()
        self.layer.state_pool.recurrent_state[0].zero_()
        self.layer.state_pool.restore_state_slot(0, snapshot)
        torch.testing.assert_close(self.layer.state_pool.conv_state[0], snapshot[0], rtol=0, atol=0)
        torch.testing.assert_close(
            self.layer.state_pool.recurrent_state[0], snapshot[1], rtol=0, atol=0
        )
        resumed = run_with_context(
            self.layer,
            inputs[split:],
            True,
            state_slots=(0,),
            state_prefix_lens=(split,),
            prefill_q_offsets=(0, len(inputs) - split),
        )
        torch.testing.assert_close(resumed, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
