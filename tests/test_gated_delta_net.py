import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.layers.gated_delta_net import GatedDeltaNet


def make_config():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        layer_types=["linear_attention"],
    )


def make_states(layer):
    return (
        torch.zeros(layer.conv_dim, layer.conv_kernel_size),
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
        world_size = patch(
            "nanovllm.layers.gated_delta_net.dist.get_world_size",
            return_value=1,
        )
        self.addCleanup(world_size.stop)
        world_size.start()
        self.layer = GatedDeltaNet(make_config(), layer_idx=0)

    def test_chunked_prefill_matches_single_chunk(self):
        hidden_states = torch.randn(11, self.layer.hidden_size)

        full_conv, full_recurrent = make_states(self.layer)
        full = self.layer._forward_sequence(
            hidden_states,
            full_conv,
            full_recurrent,
        )

        chunk_conv, chunk_recurrent = make_states(self.layer)
        chunked = torch.cat(
            [
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
            ],
            dim=0,
        )

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
        torch.testing.assert_close(
            chunk_conv,
            full_conv,
            rtol=0,
            atol=0,
        )

    def test_token_decode_matches_prefill_recurrence(self):
        hidden_states = torch.randn(8, self.layer.hidden_size)

        full_conv, full_recurrent = make_states(self.layer)
        full = self.layer._forward_sequence(
            hidden_states,
            full_conv,
            full_recurrent,
        )

        decode_conv, decode_recurrent = make_states(self.layer)
        decoded = torch.cat(
            [
                self.layer._forward_sequence(
                    hidden_states[i : i + 1],
                    decode_conv,
                    decode_recurrent,
                )
                for i in range(hidden_states.size(0))
            ],
            dim=0,
        )

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


if __name__ == "__main__":
    unittest.main()
