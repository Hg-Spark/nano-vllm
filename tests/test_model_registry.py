import unittest
from types import SimpleNamespace

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.models.registry import get_model_class
from nanovllm.utils.loader import _map_weight_name


class ModelRegistryTest(unittest.TestCase):

    def test_qwen3_resolution(self):
        config = SimpleNamespace(architectures=["Qwen3ForCausalLM"])
        self.assertIs(get_model_class(config), Qwen3ForCausalLM)

    def test_qwen35_text_resolution(self):
        config = SimpleNamespace(
            architectures=["Qwen3_5ForCausalLM"],
            model_type="qwen3_5_text",
        )
        self.assertIs(get_model_class(config), Qwen3_5ForCausalLM)

    def test_qwen35_multimodal_wrapper_resolution(self):
        config = SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            model_type="qwen3_5",
            text_config=SimpleNamespace(model_type="qwen3_5_text"),
        )
        self.assertIs(get_model_class(config), Qwen3_5ForCausalLM)

    def test_nested_text_config_fallback(self):
        config = SimpleNamespace(
            architectures=["UnknownWrapper"],
            model_type="unknown_wrapper",
            text_config=SimpleNamespace(
                architectures=[],
                model_type="qwen3_5_text",
            ),
        )
        self.assertIs(get_model_class(config), Qwen3_5ForCausalLM)

    def test_model_type_fallback(self):
        config = SimpleNamespace(
            architectures=[],
            model_type="qwen3_5_text",
        )
        self.assertIs(get_model_class(config), Qwen3_5ForCausalLM)

    def test_unknown_architecture_fails_explicitly(self):
        config = SimpleNamespace(
            architectures=["UnknownForCausalLM"],
            model_type="unknown",
        )
        with self.assertRaisesRegex(
            ValueError,
            "unsupported model architecture",
        ):
            get_model_class(config)


class CheckpointNameMappingTest(unittest.TestCase):

    def test_qwen35_language_model_prefix_is_removed(self):
        name = (
            "model.language_model.layers.0.linear_attn."
            "in_proj_qkv.weight"
        )
        self.assertEqual(
            _map_weight_name(Qwen3_5ForCausalLM, name),
            "model.layers.0.linear_attn.in_proj_qkv.weight",
        )

    def test_qwen35_non_text_weights_are_skipped(self):
        self.assertIsNone(
            _map_weight_name(
                Qwen3_5ForCausalLM,
                "model.visual.patch_embed.proj.weight",
            )
        )
        self.assertIsNone(
            _map_weight_name(
                Qwen3_5ForCausalLM,
                "mtp.layers.0.self_attn.q_proj.weight",
            )
        )

    def test_root_lm_head_is_preserved(self):
        self.assertEqual(
            _map_weight_name(Qwen3_5ForCausalLM, "lm_head.weight"),
            "lm_head.weight",
        )


if __name__ == "__main__":
    unittest.main()
