import unittest

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")
import tempfile
from pathlib import Path
from unittest.mock import patch

from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP as HFQwen3_5MLP

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.models.qwen3_5_layers import Qwen3_5MLP
from nanovllm.utils.loader import load_model
from tests.unit.test_qwen3_5_moe import make_text_config


def dense_config():
    config = make_text_config()
    config.model_type = "qwen3_5_text"
    config.intermediate_size = 12
    config.tie_word_embeddings = True
    for name in (
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        delattr(config, name)
    return config


class Qwen3_5DenseTest(unittest.TestCase):
    def test_dense_backbone_has_no_experts_and_shares_output_weight(self):
        model = Qwen3_5ForCausalLM(dense_config())
        self.assertFalse(model.is_moe)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertEqual(len(model.kv_cache_modules()), 1)
        self.assertEqual(len(model.state_cache_modules()), 1)
        for layer in model.model.layers:
            self.assertIsInstance(layer.mlp, Qwen3_5MLP)
            self.assertFalse(hasattr(layer.mlp, "experts"))

    def test_dense_mlp_packs_hf_checkpoint_weights_in_gate_up_order(self):
        config = dense_config()
        hf = HFQwen3_5MLP(config, config.intermediate_size)
        nano = Qwen3_5MLP(config, config.intermediate_size)
        nano.load_state_dict(hf.state_dict())

        expected = torch.cat((hf.gate_proj.weight, hf.up_proj.weight), dim=0)
        nano.pack_projections()
        self.assertFalse(hasattr(nano, "gate_proj"))
        self.assertFalse(hasattr(nano, "up_proj"))
        torch.testing.assert_close(nano.gate_up_proj.weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(nano.down_proj.weight, hf.down_proj.weight, rtol=0, atol=0)

    def test_dense_runtime_does_not_initialize_moe(self):
        from types import SimpleNamespace

        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(is_moe=False)
        runner.model = Qwen3_5ForCausalLM(dense_config())
        with patch("nanovllm.engine.model_runner.initialize_moe_runtime") as initialize:
            runner._init_moe_runtime()
        initialize.assert_not_called()
        self.assertIsNone(runner.moe_workspace)
        self.assertIsNone(runner.moe_output)

    def test_strict_loading_handles_either_tied_name_and_rejects_conflicts(self):
        source = Qwen3_5ForCausalLM(dense_config())
        # Use finite deterministic weights, including GDN's uninitialized A_log.
        weights = {
            name: torch.full_like(param, 0.1 + i * 0.01)
            for i, (name, param) in enumerate(source.named_parameters())
        }
        embed = "model.embed_tokens.weight"
        for mode in ("embedding_only", "head_only", "both", "conflict", "missing_ffn"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                checkpoint = {name: value.clone() for name, value in weights.items()}
                if mode in ("head_only", "both", "conflict"):
                    checkpoint["lm_head.weight"] = checkpoint[embed].clone()
                if mode == "head_only":
                    del checkpoint[embed]
                if mode == "conflict":
                    checkpoint["lm_head.weight"].add_(1)
                if mode == "missing_ffn":
                    del checkpoint["model.layers.0.mlp.gate_proj.weight"]
                save_file(checkpoint, str(Path(folder) / "model.safetensors"))
                model = Qwen3_5ForCausalLM(dense_config())
                if mode in ("conflict", "missing_ffn"):
                    error = ValueError if mode == "conflict" else KeyError
                    message = "conflicting" if mode == "conflict" else "missing model weights"
                    with self.assertRaisesRegex(error, message):
                        load_model(model, folder)
                else:
                    load_model(model, folder)
                    for name, param in model.named_parameters():
                        torch.testing.assert_close(param, weights[name], rtol=0, atol=0)
                    self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)


if __name__ == "__main__":
    unittest.main()
