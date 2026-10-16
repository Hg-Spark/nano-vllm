import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.gdn_chunk import chunk_gated_delta_rule


def recurrent_oracle(query, key, value, decay, beta, initial_state):
    """Independent FP64 token recurrence in KV layout, including nonzero history."""
    query, key, value, decay, beta = [x.double() for x in (query, key, value, decay, beta)]
    query /= ((query * query).sum(-1, keepdim=True) + 1e-6).sqrt()
    key /= ((key * key).sum(-1, keepdim=True) + 1e-6).sqrt()
    query *= query.shape[-1] ** -0.5
    state = initial_state.double().clone()
    outputs = []
    for t in range(query.shape[1]):
        state *= decay[:, t].exp()[..., None, None]
        prediction = (state * key[:, t, :, :, None]).sum(-2)
        delta = beta[:, t, :, None] * (value[:, t] - prediction)
        state += key[:, t, :, :, None] * delta[:, :, None, :]
        outputs.append((state * query[:, t, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1), state


class ChunkGatedDeltaRuleTest(unittest.TestCase):
    @torch.no_grad()
    def test_padding_boundaries_and_history_match_fp64_recurrence(self):
        torch.manual_seed(11)
        for length in (1, 63, 64, 65, 129):
            for dtype in (torch.float32, torch.float16):
                for nonzero_history in (False, True):
                    with self.subTest(length=length, dtype=dtype, history=nonzero_history):
                        # Non-square state and B=2 expose transposition/broadcast errors.
                        query = torch.randn(2, length, 3, 7, dtype=dtype)
                        key = torch.randn_like(query)
                        value = torch.randn(2, length, 3, 5, dtype=dtype)
                        decay = -torch.rand(2, length, 3) * 0.2
                        beta = torch.rand(2, length, 3)
                        state = torch.zeros(2, 3, 7, 5)
                        if nonzero_history:
                            state.normal_()
                        original = state.clone()
                        expected, expected_state = recurrent_oracle(
                            query,
                            key,
                            value,
                            decay,
                            beta,
                            state,
                        )
                        output, final_state = chunk_gated_delta_rule(
                            query,
                            key,
                            value,
                            decay,
                            beta,
                            state,
                        )
                        self.assertEqual(output.dtype, dtype)
                        self.assertEqual(final_state.dtype, torch.float32)
                        self.assertEqual(tuple(output.shape), tuple(value.shape))
                        self.assertTrue(torch.equal(state, original))
                        tolerance = 2e-3 if dtype == torch.float16 else 2e-5
                        torch.testing.assert_close(
                            output.double(),
                            expected,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                        torch.testing.assert_close(
                            final_state.double(),
                            expected_state,
                            rtol=2e-5,
                            atol=2e-5,
                        )

    @torch.no_grad()
    def test_partitioned_chunk_prefill_matches_recurrent_history(self):
        torch.manual_seed(12)
        query, key = [torch.randn(1, 139, 2, 6) for _ in range(2)]
        value = torch.randn(1, 139, 2, 4)
        decay = -torch.rand(1, 139, 2) * 0.2
        beta = torch.rand(1, 139, 2)
        initial_state = torch.randn(1, 2, 6, 4)
        expected, expected_state = recurrent_oracle(
            query,
            key,
            value,
            decay,
            beta,
            initial_state,
        )
        state, outputs, start = initial_state, [], 0
        for end in (37, 102, 138):
            output, state = chunk_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                decay[:, start:end],
                beta[:, start:end],
                state,
            )
            outputs.append(output)
            start = end
        oracle_tail, state = recurrent_oracle(
            query[:, -1:],
            key[:, -1:],
            value[:, -1:],
            decay[:, -1:],
            beta[:, -1:],
            state,
        )
        outputs.append(oracle_tail.float())
        torch.testing.assert_close(
            torch.cat(outputs, 1).double(),
            expected,
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-5)

    @torch.no_grad()
    def test_grouped_heads_match_explicit_head_expansion(self):
        config = SimpleNamespace(
            hidden_size=16,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=7,
            linear_value_head_dim=5,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
        )
        layer = GatedDeltaNet(config)
        torch.manual_seed(13)
        query, key = [torch.randn(65, 2, 7) for _ in range(2)]
        value = torch.randn(65, 4, 5)
        decay = -torch.rand(65, 4)
        beta = torch.rand(65, 4)
        state = torch.randn(4, 5, 7)
        expected, final_state = chunk_gated_delta_rule(
            query.repeat_interleave(2, 1).unsqueeze(0),
            key.repeat_interleave(2, 1).unsqueeze(0),
            value.unsqueeze(0),
            decay.unsqueeze(0),
            beta.unsqueeze(0),
            state.transpose(-1, -2).unsqueeze(0).contiguous(),
        )
        actual = layer._delta_rule(query, key, value, decay, beta, state)
        torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(
            state,
            final_state[0].transpose(-1, -2),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
