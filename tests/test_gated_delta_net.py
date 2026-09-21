import unittest
from types import SimpleNamespace

import torch

from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.utils.context import reset_context, set_context


def make_config():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
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
            layer.head_k_dim,
            layer.head_v_dim,
            dtype=torch.float32,
        ),
    )


class GatedDeltaNetStateTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.layer = GatedDeltaNet(
            make_config(),
            layer_idx=0,
        )
        with torch.no_grad():
            self.layer.A_log.zero_()

    def tearDown(self):
        reset_context()

    def test_chunked_reference_matches_single_chunk(self):
        hidden_states = torch.randn(
            11,
            self.layer.hidden_size,
        )

        full_conv, full_recurrent = make_states(self.layer)
        full = self.layer._forward_sequence(
            hidden_states,
            full_conv,
            full_recurrent,
        )

        chunk_conv, chunk_recurrent = make_states(self.layer)
        chunked = torch.cat([
            self.layer._forward_sequence(
                hidden_states[:3],
                chunk_conv,
                chunk_recurrent,
            ),
            self.layer._forward_sequence(
                hidden_states[3:7],
                chunk_conv,
                chunk_recurrent,
            ),
            self.layer._forward_sequence(
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
        full = self.layer._forward_sequence(
            hidden_states,
            full_conv,
            full_recurrent,
        )

        decode_conv, decode_recurrent = make_states(
            self.layer
        )
        decoded = torch.cat([
            self.layer._forward_sequence(
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
        expected = self.layer._forward_sequence(
            hidden_states,
            expected_conv,
            expected_recurrent,
        )

        self.layer.allocate_state_cache(1)
        self.layer.conv_state[0].fill_(7)
        self.layer.recurrent_state[0].fill_(11)

        set_context(
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, 5),
        )
        actual = self.layer(hidden_states)

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
        expected = self.layer._forward_sequence(
            hidden_states,
            expected_conv,
            expected_recurrent,
        )

        self.layer.allocate_state_cache(1)
        set_context(
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, 3),
        )
        first = self.layer(hidden_states[:3])

        set_context(
            True,
            state_slots=(0,),
            state_prefix_lens=(3,),
            prefill_q_offsets=(0, 4),
        )
        second = self.layer(hidden_states[3:])

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
        expected_first = self.layer._forward_sequence(
            first_seq,
            first_conv,
            first_recurrent,
        )
        second_conv, second_recurrent = make_states(self.layer)
        expected_second = self.layer._forward_sequence(
            second_seq,
            second_conv,
            second_recurrent,
        )

        self.layer.allocate_state_cache(2)
        set_context(
            True,
            state_slots=(0,),
            state_prefix_lens=(0,),
            prefill_q_offsets=(0, 2),
        )
        first_chunk = self.layer(first_seq[:2])

        packed = torch.cat([
            first_seq[2:],
            second_seq[:3],
        ])
        set_context(
            True,
            state_slots=(0, 1),
            state_prefix_lens=(2, 0),
            prefill_q_offsets=(0, 4, 7),
        )
        mixed_chunk = self.layer(packed)

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

    def test_state_slot_snapshot_round_trip(self):
        self.layer.allocate_state_cache(1)
        expected_conv = torch.randn_like(
            self.layer.conv_state[0]
        )
        expected_recurrent = torch.randn_like(
            self.layer.recurrent_state[0]
        )
        self.layer.conv_state[0].copy_(expected_conv)
        self.layer.recurrent_state[0].copy_(expected_recurrent)

        snapshot = self.layer.snapshot_state_slot(0)
        expected_conv_after_restore = expected_conv.to(
            torch.bfloat16
        ).to(expected_conv.dtype)
        expected_recurrent_after_restore = expected_recurrent.to(
            torch.bfloat16
        ).to(expected_recurrent.dtype)

        self.layer.conv_state[0].zero_()
        self.layer.recurrent_state[0].zero_()
        self.layer.restore_state_slot(0, snapshot)

        torch.testing.assert_close(
            self.layer.conv_state[0],
            expected_conv_after_restore,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            self.layer.recurrent_state[0],
            expected_recurrent_after_restore,
            rtol=0,
            atol=0,
        )
        self.assertEqual(snapshot[0].device.type, "cpu")
        self.assertEqual(snapshot[1].device.type, "cpu")
        self.assertEqual(snapshot[0].dtype, torch.bfloat16)
        self.assertEqual(snapshot[1].dtype, torch.bfloat16)

    def test_state_slot_restore_rejects_non_bf16_snapshot(self):
        self.layer.allocate_state_cache(1)
        invalid_snapshot = (
            torch.zeros_like(self.layer.conv_state[0]).cpu(),
            torch.zeros_like(self.layer.recurrent_state[0]).cpu(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "snapshot must use BF16 storage",
        ):
            self.layer.restore_state_slot(0, invalid_snapshot)

    def test_negative_state_prefix_is_rejected(self):
        self.layer.allocate_state_cache(1)
        hidden_states = torch.randn(
            2,
            self.layer.hidden_size,
        )
        set_context(
            True,
            state_slots=(0,),
            state_prefix_lens=(-1,),
            prefill_q_offsets=(0, 2),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "invalid GDN state prefix",
        ):
            self.layer(hidden_states)

if __name__ == "__main__":
    unittest.main()
