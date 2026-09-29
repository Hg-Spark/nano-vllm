import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config


def make_qwen35_text_config():
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

    def test_decode_graph_batch_sizes_are_validated_before_model_load(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with self.assertRaisesRegex(
                ValueError,
                "must not contain duplicates",
            ):
                Config(
                    model_dir,
                    max_num_seqs=4,
                    decode_graph_batch_sizes=(1, 1),
                )

            with self.assertRaisesRegex(
                ValueError,
                "within",
            ):
                Config(
                    model_dir,
                    max_num_seqs=2,
                    decode_graph_batch_sizes=(1, 3),
                )

    def test_kvcache_block_size_must_match_flashinfer_pages(self):
        with tempfile.TemporaryDirectory() as model_dir:
            for block_size in (0, -16, 8, 256):
                with self.subTest(block_size=block_size):
                    with self.assertRaisesRegex(
                        ValueError,
                        "must be one of",
                    ):
                        Config(
                            model_dir,
                            kvcache_block_size=block_size,
                        )

    def test_accepts_qwen35_moe(self):
        text = make_qwen35_text_config()
        root = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=text,
            quantization_config=None,
        )
        with (
            patch("nanovllm.config.os.path.isdir", return_value=True),
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=root,
            ),
        ):
            config = Config("/tmp/qwen35-moe")

        self.assertIs(config.root_config, root)
        self.assertIs(config.text_config, text)

    def test_accepts_official_multimodal_mrope(self):
        text = make_qwen35_text_config()
        text.head_dim = 12
        text.partial_rotary_factor = 1.0
        text.rope_parameters = {
            "rope_type": "default",
            "mrope_interleaved": True,
            "mrope_section": [2, 2, 2],
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000.0,
        }
        root = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=text,
            vision_config=SimpleNamespace(spatial_merge_size=2),
            quantization_config=None,
        )
        with (
            patch("nanovllm.config.os.path.isdir", return_value=True),
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=root,
            ),
        ):
            config = Config("/tmp/qwen35-moe")

        self.assertIs(config.text_config, text)

    def test_rejects_unsupported_rope_scaling(self):
        text = make_qwen35_text_config()
        text.rope_parameters = {
            "rope_type": "yarn",
            "mrope_interleaved": True,
            "mrope_section": [1, 0, 0],
            "partial_rotary_factor": 0.5,
        }
        root = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=text,
            vision_config=SimpleNamespace(spatial_merge_size=2),
            quantization_config=None,
        )
        with (
            patch("nanovllm.config.os.path.isdir", return_value=True),
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=root,
            ),
        ):
            with self.assertRaisesRegex(
                NotImplementedError,
                "default RoPE only",
            ):
                Config("/tmp/qwen35-moe")

    def test_rejects_incomplete_multimodal_mrope(self):
        text = make_qwen35_text_config()
        text.rope_parameters = {
            "rope_type": "default",
            "mrope_interleaved": True,
        }
        root = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=text,
            vision_config=SimpleNamespace(spatial_merge_size=2),
            quantization_config=None,
        )
        with (
            patch("nanovllm.config.os.path.isdir", return_value=True),
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=root,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires rope_parameters.mrope_section",
            ):
                Config("/tmp/qwen35-moe")

    def test_sequence_capacity_is_single_active_request_limit(self):
        text = make_qwen35_text_config()
        root = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=text,
            quantization_config=None,
        )
        with (
            patch("nanovllm.config.os.path.isdir", return_value=True),
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=root,
            ),
        ):
            config = Config(
                "/tmp/qwen35-moe",
                max_num_seqs=2,
            )

        self.assertEqual(config.max_num_seqs, 2)

    def test_rejects_dense_qwen35(self):
        dense = SimpleNamespace(
            model_type="qwen3_5",
            text_config=SimpleNamespace(
                model_type="qwen3_5_text",
            ),
            quantization_config=None,
        )
        with (
            patch("nanovllm.config.os.path.isdir", return_value=True),
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=dense,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "only Qwen3.5-MoE checkpoints",
            ):
                Config("/tmp/qwen35-dense")



if __name__ == "__main__":
    unittest.main()
