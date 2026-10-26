"""Configuration validation must reject silent numeric coercion."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config


class ConfigStrictTest(unittest.TestCase):
    def test_integer_options_reject_bool_and_fractional_values(self):
        cases = (
            ("max_num_batched_tokens", True),
            ("max_num_seqs", 1.5),
            ("max_model_len", False),
            ("kv_cache_block_size", 16.0),
            ("max_prefix_cache_entries", True),
            ("max_prefix_cache_snapshot_bytes", 1.5),
            ("max_visual_tokens_per_request", True),
            ("decode_graph_batch_sizes", (2.5,)),
            ("decode_graph_batch_sizes", (True,)),
        )
        with tempfile.TemporaryDirectory() as folder:
            for name, invalid in cases:
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        Config(folder, **{name: invalid})

    def test_text_model_does_not_accept_unbounded_visual_tokens(self):
        text = SimpleNamespace(
            model_type="qwen3_5_text",
            max_position_embeddings=4096,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            intermediate_size=128,
            layer_types=("linear_attention", "full_attention"),
            num_hidden_layers=2,
            hidden_act="silu",
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        root = SimpleNamespace(model_type="qwen3_5_text", text_config=text, vision_config=None)
        with tempfile.TemporaryDirectory() as folder:
            with patch("nanovllm.config._load_hf_config", return_value=root):
                self.assertEqual(
                    Config(folder, max_visual_tokens_per_request=0).max_visual_tokens_per_request,
                    0,
                )
                with self.assertRaisesRegex(ValueError, "requires a vision model"):
                    Config(folder, max_visual_tokens_per_request=None)


if __name__ == "__main__":
    unittest.main()
