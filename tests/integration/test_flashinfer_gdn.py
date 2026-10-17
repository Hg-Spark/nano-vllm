"""Optional SM90+ regression checks against the PyTorch recurrence oracle."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.layers.gated_delta_net import GatedDeltaNet


def supported_gpu():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (9, 10, 12)


@unittest.skipUnless(supported_gpu(), "requires SM90+ FlashInfer GDN kernels")
class FlashInferGDNTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        config = SimpleNamespace(
            hidden_size=256,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
        )
        self.layer = GatedDeltaNet(config).cuda().to(torch.bfloat16)
        with torch.no_grad():
            self.layer.A_log.fill_(0)
        self.layer.state_pool.allocate_state_cache(2)

    @torch.no_grad()
    def test_packed_prefill_matches_fp32_chunked_reference(self):
        conv_state = torch.randn_like(self.layer.state_pool.conv_state) * 0.01
        recurrent_state = torch.randn_like(self.layer.state_pool.recurrent_state) * 0.01
        self.layer.state_pool.conv_state.copy_(conv_state)
        self.layer.state_pool.recurrent_state.copy_(recurrent_state)
        inputs = torch.randn(69, 256, device="cuda", dtype=torch.bfloat16)
        args = ((0, 1), (4, 3), (0, 5, 69))
        with patch.object(self.layer, "_can_use_flashinfer", return_value=False):
            expected = self.layer._forward_prefill_batch(inputs, *args)
        expected_state = self.layer.state_pool.recurrent_state.clone()
        self.layer.state_pool.conv_state.copy_(conv_state)
        self.layer.state_pool.recurrent_state.copy_(recurrent_state)
        self.assertTrue(self.layer._can_use_flashinfer(inputs))
        actual = self.layer._forward_prefill_batch(
            inputs,
            *args,
            offsets=torch.tensor(args[2], device="cuda", dtype=torch.int32),
            slot_ids=torch.tensor(args[0], device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(
            self.layer.state_pool.recurrent_state,
            expected_state,
            atol=5e-2,
            rtol=5e-2,
        )

    @torch.no_grad()
    def test_decode_tracks_slot_permutations_across_steps(self):
        conv_state = torch.randn_like(self.layer.state_pool.conv_state) * 0.01
        recurrent_state = torch.randn_like(self.layer.state_pool.recurrent_state) * 0.01
        self.layer.state_pool.conv_state.copy_(conv_state)
        self.layer.state_pool.recurrent_state.copy_(recurrent_state)

        permutations = ((1, 0), (0, 1), (1, 0))
        batches = [
            (
                torch.randn(2, 256, device="cuda", dtype=torch.bfloat16),
                torch.tensor(slots, device="cuda", dtype=torch.int32),
            )
            for slots in permutations
        ]
        # Compare Delta Rule backends; both paths use causal-conv1d for convolution.
        with patch.object(self.layer, "_can_use_flashinfer", return_value=False):
            expected_outputs = [
                self.layer._forward_decode_batch(inputs, slots) for inputs, slots in batches
            ]
        expected_conv = self.layer.state_pool.conv_state.clone()
        expected_state = self.layer.state_pool.recurrent_state.clone()

        self.layer.state_pool.conv_state.copy_(conv_state)
        self.layer.state_pool.recurrent_state.copy_(recurrent_state)
        for (inputs, slots), expected in zip(batches, expected_outputs):
            self.assertTrue(self.layer._can_use_flashinfer(inputs))
            actual = self.layer._forward_decode_batch(inputs, slots)
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(
            self.layer.state_pool.conv_state, expected_conv, atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(
            self.layer.state_pool.recurrent_state, expected_state, atol=5e-2, rtol=5e-2
        )


if __name__ == "__main__":
    unittest.main()
