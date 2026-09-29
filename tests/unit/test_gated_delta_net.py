import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
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
            layer.conv_kernel_size,
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
    mixed_qkv = layer.in_proj_qkv(hidden_states)
    z = layer.in_proj_z(hidden_states).view(
        -1,
        layer.num_v_heads,
        layer.head_v_dim,
    )
    beta = layer.in_proj_b(hidden_states).sigmoid()
    decay = (
        -layer.A_log.float().exp()
        * F.softplus(layer.in_proj_a(hidden_states).float() + layer.dt_bias)
    )
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

    def test_recurrent_state_uses_value_key_layout(self):
        self.layer.state_pool.allocate_state_cache(2)
        self.assertEqual(
            tuple(self.layer.state_pool.recurrent_state.shape),
            (
                2,
                self.layer.num_v_heads,
                self.layer.head_v_dim,
                self.layer.head_k_dim,
            ),
        )

        _, temporary_state = self.layer.state_pool.temporary(
            torch.device("cpu"),
            torch.float32,
        )
        self.assertEqual(
            tuple(temporary_state.shape),
            (
                self.layer.num_v_heads,
                self.layer.head_v_dim,
                self.layer.head_k_dim,
            ),
        )

    def test_flashinfer_prefill_uses_value_key_state_layout(self):
        self.layer.state_pool.allocate_state_cache(1)
        expected_state = torch.arange(
            self.layer.num_v_heads
            * self.layer.head_v_dim
            * self.layer.head_k_dim,
            dtype=torch.float32,
        ).view(
            self.layer.num_v_heads,
            self.layer.head_v_dim,
            self.layer.head_k_dim,
        )
        self.layer.state_pool.recurrent_state[0].copy_(expected_state)
        hidden_states = torch.randn(
            2,
            self.layer.hidden_size,
        )

        def fake_flashinfer(**kwargs):
            initial_state = kwargs["initial_state"]
            self.assertEqual(
                tuple(initial_state.shape),
                (
                    1,
                    self.layer.num_v_heads,
                    self.layer.head_v_dim,
                    self.layer.head_k_dim,
                ),
            )
            torch.testing.assert_close(
                initial_state[0],
                expected_state,
                rtol=0,
                atol=0,
            )
            return (
                kwargs["v"].new_zeros(kwargs["v"].shape),
                initial_state + 1,
            )

        with (
            patch.object(
                self.layer,
                "_can_use_flashinfer_prefill",
                return_value=True,
            ),
            patch(
                "nanovllm.layers.gated_delta_net."
                "flashinfer_chunk_gated_delta_rule",
                side_effect=fake_flashinfer,
            ),
        ):
            self.layer._forward_prefill_batch(
                hidden_states,
                state_slots=(0,),
                state_prefix_lens=(1,),
                q_offsets=(0, 2),
                cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            )

        torch.testing.assert_close(
            self.layer.state_pool.recurrent_state[0],
            expected_state + 1,
            rtol=0,
            atol=0,
        )

    def test_chunked_reference_matches_single_chunk(self):
        hidden_states = torch.randn(
            11,
            self.layer.hidden_size,
        )

        full_conv, full_recurrent = make_states(self.layer)
        full = reference_forward_sequence(
            self.layer,
            hidden_states,
            full_conv,
            full_recurrent,
        )

        chunk_conv, chunk_recurrent = make_states(self.layer)
        chunked = torch.cat([
            reference_forward_sequence(
                self.layer,
                hidden_states[:3],
                chunk_conv,
                chunk_recurrent,
            ),
            reference_forward_sequence(
                self.layer,
                hidden_states[3:7],
                chunk_conv,
                chunk_recurrent,
            ),
            reference_forward_sequence(
                self.layer,
                hidden_states[7:],
                chunk_conv,
                chunk_recurrent,
            ),
        ])

        torch.testing.assert_close(
            chunked,
            full,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            chunk_recurrent,
            full_recurrent,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_token_decode_matches_prefill_recurrence(self):
        hidden_states = torch.randn(
            8,
            self.layer.hidden_size,
        )

        full_conv, full_recurrent = make_states(self.layer)
        full = reference_forward_sequence(
            self.layer,
            hidden_states,
            full_conv,
            full_recurrent,
        )

        decode_conv, decode_recurrent = make_states(
            self.layer
        )
        decoded = torch.cat([
            reference_forward_sequence(
                self.layer,
                hidden_states[i:i + 1],
                decode_conv,
                decode_recurrent,
            )
            for i in range(hidden_states.size(0))
        ])

        torch.testing.assert_close(
            decoded,
            full,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            decode_recurrent,
            full_recurrent,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_fresh_prefill_clears_reused_slot(self):
        hidden_states = torch.randn(
            5,
            self.layer.hidden_size,
        )

        expected_conv, expected_recurrent = make_states(
            self.layer
        )
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

    def test_runtime_context_preserves_state_across_chunks(self):
        hidden_states = torch.randn(
            7,
            self.layer.hidden_size,
        )

        expected_conv, expected_recurrent = make_states(
            self.layer
        )
        expected = reference_forward_sequence(
            self.layer,
            hidden_states,
            expected_conv,
            expected_recurrent,
        )

        self.layer.state_pool.allocate_state_cache(1)
        first = run_with_context(
            self.layer,
            hidden_states[:3],
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, 3),
        )
        second = run_with_context(
            self.layer,
            hidden_states[3:],
            True,
            state_slots=(0,),
            state_prefix_lens=(3,),
            prefill_q_offsets=(0, 4),
        )

        torch.testing.assert_close(
            torch.cat([first, second]),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )


    def test_variable_length_batch_keeps_state_aligned(self):
        first_seq = torch.randn(
            6,
            self.layer.hidden_size,
        )
        second_seq = torch.randn(
            4,
            self.layer.hidden_size,
        )

        first_conv, first_recurrent = make_states(self.layer)
        expected_first = reference_forward_sequence(
            self.layer,
            first_seq,
            first_conv,
            first_recurrent,
        )
        second_conv, second_recurrent = make_states(self.layer)
        expected_second = reference_forward_sequence(
            self.layer,
            second_seq,
            second_conv,
            second_recurrent,
        )

        self.layer.state_pool.allocate_state_cache(2)
        first_chunk = run_with_context(
            self.layer,
            first_seq[:2],
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, 2),
        )

        packed = torch.cat([
            first_seq[2:],
            second_seq[:3],
        ])
        mixed_chunk = run_with_context(
            self.layer,
            packed,
            True,
            state_slots=(0, 1),
            state_prefix_lens=(2, 0),
            prefill_q_offsets=(0, 4, 7),
        )

        torch.testing.assert_close(
            first_chunk,
            expected_first[:2],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            mixed_chunk[:4],
            expected_first[2:],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            mixed_chunk[4:],
            expected_second[:3],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_decode_context_requires_only_device_slot_ids(self):
        self.layer.state_pool.allocate_state_cache(1)
        hidden_states = torch.randn(
            1,
            self.layer.hidden_size,
        )
        output = run_with_context(
            self.layer,
            hidden_states,
            False,
            state_slot_ids=torch.tensor([0], dtype=torch.int32),
        )

        self.assertEqual(
            tuple(output.shape),
            (1, self.layer.hidden_size),
        )

    def test_vectorized_decode_matches_per_slot_reference(self):
        self.layer.state_pool.allocate_state_cache(3)
        with torch.no_grad():
            self.layer.state_pool.conv_state.copy_(
                torch.randn_like(self.layer.state_pool.conv_state)
            )
            self.layer.state_pool.recurrent_state.copy_(
                torch.randn_like(self.layer.state_pool.recurrent_state)
            )

        initial_conv = self.layer.state_pool.conv_state.clone()
        initial_recurrent = (
            self.layer.state_pool.recurrent_state.clone()
        )
        hidden_states = torch.randn(
            2,
            self.layer.hidden_size,
        )
        slot_ids = torch.tensor([2, 0], dtype=torch.int32)

        expected_outputs = []
        expected_conv = initial_conv.clone()
        expected_recurrent = initial_recurrent.clone()
        for row, slot_id in enumerate((2, 0)):
            expected_outputs.append(
                reference_forward_sequence(
                self.layer,
                    hidden_states[row : row + 1],
                    expected_conv[slot_id],
                    expected_recurrent[slot_id],
                )
            )
        expected = torch.cat(expected_outputs)

        self.layer.state_pool.conv_state.copy_(initial_conv)
        self.layer.state_pool.recurrent_state.copy_(initial_recurrent)
        actual = self.layer._forward_decode_batch(
            hidden_states,
            slot_ids,
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            self.layer.state_pool.conv_state,
            expected_conv,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            self.layer.state_pool.recurrent_state,
            expected_recurrent,
            rtol=1e-5,
            atol=1e-5,
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 is required",
    )
    def test_cuda_bf16_decode_tracks_slot_permutations(self):
        layer = GatedDeltaNet(make_config()).cuda().to(torch.bfloat16)
        with torch.no_grad():
            layer.A_log.zero_()
        layer.state_pool.allocate_state_cache(3)
        with torch.no_grad():
            layer.state_pool.conv_state.copy_(
                torch.randn_like(layer.state_pool.conv_state)
            )
            layer.state_pool.recurrent_state.copy_(
                torch.randn_like(layer.state_pool.recurrent_state)
            )

        expected_conv = layer.state_pool.conv_state.clone()
        expected_recurrent = layer.state_pool.recurrent_state.clone()
        permutations = (
            (2, 0, 1),
            (1, 2, 0),
            (0, 1, 2),
        )
        for permutation in permutations:
            hidden_states = torch.randn(
                3,
                layer.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            expected_outputs = []
            for row, slot_id in enumerate(permutation):
                expected_outputs.append(
                    reference_forward_sequence(
                        layer,
                        hidden_states[row : row + 1],
                        expected_conv[slot_id],
                        expected_recurrent[slot_id],
                    )
                )
            expected = torch.cat(expected_outputs)
            slot_ids = torch.tensor(
                permutation,
                dtype=torch.int32,
                device="cuda",
            )
            actual = layer._forward_decode_batch(
                hidden_states,
                slot_ids,
            )
            torch.testing.assert_close(
                actual,
                expected,
                rtol=2e-2,
                atol=2e-2,
            )

        torch.testing.assert_close(
            layer.state_pool.conv_state,
            expected_conv,
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            layer.state_pool.recurrent_state,
            expected_recurrent,
            rtol=2e-2,
            atol=2e-2,
        )

    def test_state_slot_snapshot_round_trip(self):
        self.layer.state_pool.allocate_state_cache(1)
        expected_conv = torch.randn_like(
            self.layer.state_pool.conv_state[0]
        )
        expected_recurrent = torch.randn_like(
            self.layer.state_pool.recurrent_state[0]
        )
        self.layer.state_pool.conv_state[0].copy_(expected_conv)
        self.layer.state_pool.recurrent_state[0].copy_(expected_recurrent)

        snapshot = self.layer.state_pool.snapshot_state_slot(0)

        self.layer.state_pool.conv_state[0].zero_()
        self.layer.state_pool.recurrent_state[0].zero_()
        self.layer.state_pool.restore_state_slot(0, snapshot)

        torch.testing.assert_close(
            self.layer.state_pool.conv_state[0],
            expected_conv,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            self.layer.state_pool.recurrent_state[0],
            expected_recurrent,
            rtol=0,
            atol=0,
        )
        self.assertEqual(snapshot[0].device.type, "cpu")
        self.assertEqual(snapshot[1].device.type, "cpu")
        self.assertEqual(
            snapshot[0].dtype,
            self.layer.state_pool.conv_state.dtype,
        )
        self.assertEqual(snapshot[1].dtype, torch.float32)

    def test_snapshot_resume_matches_uninterrupted_continuation(self):
        hidden_states = torch.randn(
            9,
            self.layer.hidden_size,
        )
        split = 5

        self.layer.state_pool.allocate_state_cache(1)
        run_with_context(
            self.layer,
            hidden_states[:split],
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, split),
        )
        reference_conv = self.layer.state_pool.conv_state[0].clone()
        reference_recurrent = (
            self.layer.state_pool.recurrent_state[0].clone()
        )
        snapshot = self.layer.state_pool.snapshot_state_slot(0)

        expected = reference_forward_sequence(
            self.layer,
            hidden_states[split:],
            reference_conv,
            reference_recurrent,
        )

        self.layer.state_pool.conv_state[0].zero_()
        self.layer.state_pool.recurrent_state[0].zero_()
        self.layer.state_pool.restore_state_slot(0, snapshot)
        resumed = run_with_context(
            self.layer,
            hidden_states[split:],
            True,
            state_slots=(0,),
            state_prefix_lens=(split,),
            prefill_q_offsets=(0, hidden_states.size(0) - split),
        )

        torch.testing.assert_close(
            resumed,
            expected,
            rtol=0,
            atol=0,
        )

    def test_state_slot_restore_rejects_recurrent_dtype_mismatch(self):
        self.layer.state_pool.allocate_state_cache(1)
        invalid_snapshot = (
            torch.zeros_like(self.layer.state_pool.conv_state[0]).cpu(),
            torch.zeros_like(
                self.layer.state_pool.recurrent_state[0],
                dtype=torch.bfloat16,
            ).cpu(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "recurrent snapshot dtype",
        ):
            self.layer.state_pool.restore_state_slot(0, invalid_snapshot)

    def test_negative_state_prefix_is_rejected(self):
        self.layer.state_pool.allocate_state_cache(1)
        hidden_states = torch.randn(
            2,
            self.layer.hidden_size,
        )
        with (
            use_context(
                Context(
                    is_prefill=True,
                    state_slots=(0,),
                    state_prefix_lens=(-1,),
                    prefill_q_offsets=(0, 2),
                )
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "invalid GDN state prefix",
            ),
        ):
            self.layer(hidden_states)

if __name__ == "__main__":
    unittest.main()
