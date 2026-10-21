import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.layers.gated_delta_net import GatedDeltaNet


def fake_causal_conv1d(
    x,
    weight,
    bias=None,
    *,
    initial_states=None,
    return_final_states=False,
    activation=None,
):
    history = weight.shape[-1] - 1
    padded = (
        torch.cat((initial_states, x), dim=-1)
        if initial_states is not None
        else F.pad(x, (history, 0))
    )
    result = F.conv1d(padded, weight.unsqueeze(1), bias, groups=x.shape[1])
    if activation == "silu":
        result = F.silu(result)
    return (result, padded[:, :, -history:].clone()) if return_final_states else result


class GDNConvBatchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layer = GatedDeltaNet(
            SimpleNamespace(
                hidden_size=8,
                linear_num_value_heads=1,
                linear_num_key_heads=1,
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_conv_kernel_dim=4,
                rms_norm_eps=1e-6,
            )
        )

    def _check_against_single_sequence(self, lengths):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        x = torch.randn(offsets[-1], self.layer.conv_dim)
        states = [
            torch.randn(self.layer.conv_dim, self.layer.conv_kernel_size - 1)
            for _ in lengths
        ]
        reference_states = [state.clone() for state in states]

        with patch.dict(
            sys.modules,
            {"causal_conv1d": SimpleNamespace(causal_conv1d_fn=fake_causal_conv1d)},
        ):
            expected = torch.cat(
                [
                    self.layer._causal_conv(
                        x[offsets[index] : offsets[index + 1]],
                        reference_states[index],
                    )
                    for index in range(len(lengths))
                ],
                dim=0,
            )
            actual = self.layer._causal_conv_batch(x, states, tuple(offsets))

        torch.testing.assert_close(actual, expected)
        for state, expected_state in zip(states, reference_states):
            torch.testing.assert_close(state, expected_state)

    def test_equal_length_batch_uses_native_initial_states(self):
        for lengths in ((12, 12, 12), (2, 2, 2), (1, 1)):
            with self.subTest(lengths=lengths):
                self._check_against_single_sequence(lengths)

    def test_variable_length_batch_preserves_packed_fallback(self):
        for lengths in ((7, 12, 19), (1, 3, 2)):
            with self.subTest(lengths=lengths):
                self._check_against_single_sequence(lengths)


if __name__ == "__main__":
    unittest.main()
