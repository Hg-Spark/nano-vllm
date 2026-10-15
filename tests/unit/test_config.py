import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config


def make_qwen3_5_text_config():
    return SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        partial_rotary_factor=0.5,
        max_position_embeddings=1024,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
        shared_expert_intermediate_size=5,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
    )


class ConfigTest(unittest.TestCase):
    def load(self, root, *, dtype: str, **kwargs):
        with (
            patch("nanovllm.config.os.path.isdir", return_value=True),
            patch("nanovllm.config._load_hf_config", return_value=root),
        ):
            return Config("/tmp/qwen3_5-moe", dtype=dtype, **kwargs)

    def root(self, text=None, **overrides):
        return SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=text or make_qwen3_5_text_config(),
            quantization_config=None,
            **overrides,
        )

    def test_dtype_and_bounds_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            for invalid in ("auto", "fp32", ""):
                with self.assertRaisesRegex(ValueError, "dtype must be"):
                    Config(directory, dtype=invalid)
            with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
                Config(directory, dtype="bfloat16", max_num_seqs=4, decode_graph_batch_sizes=(1, 1))
            with self.assertRaisesRegex(ValueError, "within"):
                Config(directory, dtype="bfloat16", max_num_seqs=2, decode_graph_batch_sizes=(1, 3))
            with self.assertRaisesRegex(ValueError, "must be one of"):
                Config(directory, dtype="bfloat16", kv_cache_block_size=8)
            with self.assertRaisesRegex(ValueError, "max_visual_tokens_per_request"):
                Config(directory, dtype="float16", max_visual_tokens_per_request=-1)
            with self.assertRaisesRegex(ValueError, "max_visual_tokens_per_request"):
                Config(
                    directory,
                    dtype="float16",
                    max_model_len=10,
                    max_visual_tokens_per_request=11,
                )

    def test_head_dim_fallback_and_invalid_attention_dimensions(self):
        text = make_qwen3_5_text_config()
        del text.head_dim
        config = self.load(self.root(text), dtype="float16")
        self.assertEqual(config.head_dim, 4)
        self.assertEqual(config.max_visual_tokens_per_request, 0)

        for attr, value in (
            ("num_attention_heads", 0),
            ("num_key_value_heads", 0),
            ("num_key_value_heads", 3),
            ("head_dim", -1),
        ):
            with self.subTest(attr=attr, value=value):
                bad = make_qwen3_5_text_config()
                setattr(bad, attr, value)
                with self.assertRaises(ValueError):
                    self.load(self.root(bad), dtype="float16")

    def test_accepts_dense_and_moe_and_multimodal_mrope(self):
        moe_cfg = self.load(self.root(), dtype="bfloat16")
        self.assertTrue(moe_cfg.is_moe)

        text = make_qwen3_5_text_config()
        text.head_dim = 12
        text.partial_rotary_factor = 1.0
        text.rope_parameters = {
            "rope_type": "default",
            "mrope_interleaved": True,
            "mrope_section": [2, 2, 2],
            "partial_rotary_factor": 1.0,
        }
        mm_cfg = self.load(
            self.root(text, vision_config=SimpleNamespace(spatial_merge_size=2)), dtype="bfloat16"
        )
        self.assertIs(mm_cfg.text_config, text)

        dense_text = make_qwen3_5_text_config()
        dense_text.model_type = "qwen3_5_text"
        dense_text.intermediate_size = 12
        for name in (
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ):
            delattr(dense_text, name)
        dense_root = SimpleNamespace(
            model_type="qwen3_5", text_config=dense_text, quantization_config=None
        )
        self.assertFalse(self.load(dense_root, dtype="float16").is_moe)


if __name__ == "__main__":
    unittest.main()
