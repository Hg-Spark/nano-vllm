import unittest

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from safetensors.torch import save_file

from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.models.qwen3_5_layers import Qwen3_5MLP
from nanovllm.utils.loader import load_model
from tests.qwen35_helpers import make_dense_config as dense_config


class Qwen3_5DenseTest(unittest.TestCase):
    def test_dense_backbone_has_no_experts_and_shares_output_weight(self):
        model = Qwen3_5ForCausalLM(dense_config())
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        for layer in model.model.layers:
            self.assertIsInstance(layer.mlp, Qwen3_5MLP)
            self.assertFalse(hasattr(layer.mlp, "experts"))

    def test_text_only_model_does_not_create_vision_tower(self):
        from types import SimpleNamespace

        root = SimpleNamespace(
            model_type="qwen3_5",
            text_config=dense_config(),
            vision_config=object(),
        )
        with patch("nanovllm.models.qwen3_5.AutoModel.from_config") as create_vision:
            model = Qwen3_5ForCausalLM(root, load_vision=False)
        create_vision.assert_not_called()
        self.assertIsNone(model.visual)
        self.assertIn("model.visual.", model.checkpoint_skip_prefixes)

    def test_vision_tower_uses_runtime_dtype(self):
        from types import SimpleNamespace

        root = SimpleNamespace(text_config=dense_config(), vision_config=object())
        with patch(
            "nanovllm.models.qwen3_5.AutoModel.from_config",
            return_value=torch.nn.Linear(1, 1),
        ) as make_visual:
            Qwen3_5ForCausalLM(root)
        make_visual.assert_called_once_with(root.vision_config, dtype=torch.get_default_dtype())

    def test_strict_loading_handles_either_tied_name_and_rejects_conflicts(self):
        source = Qwen3_5ForCausalLM(dense_config())
        # Independent HF shard layout for the fixed make_dense_config() fixture.
        # These offsets must not come from packed_checkpoint_shards(), the
        # production mapping being tested.
        shards = (
            (
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                "model.layers.0.linear_attn.in_proj.weight",
                0,
                16,
            ),
            (
                "model.layers.0.linear_attn.in_proj_z.weight",
                "model.layers.0.linear_attn.in_proj.weight",
                16,
                24,
            ),
            (
                "model.layers.0.linear_attn.in_proj_b.weight",
                "model.layers.0.linear_attn.in_proj.weight",
                24,
                26,
            ),
            (
                "model.layers.0.linear_attn.in_proj_a.weight",
                "model.layers.0.linear_attn.in_proj.weight",
                26,
                28,
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.gate_up_proj.weight",
                0,
                12,
            ),
            ("model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.gate_up_proj.weight", 12, 24),
            (
                "model.layers.1.mlp.gate_proj.weight",
                "model.layers.1.mlp.gate_up_proj.weight",
                0,
                12,
            ),
            ("model.layers.1.mlp.up_proj.weight", "model.layers.1.mlp.gate_up_proj.weight", 12, 24),
        )
        weights = {
            name: torch.full_like(param, 0.1 + i * 0.01)
            for i, (name, param) in enumerate(source.named_parameters())
        }
        for i, (_, target, start, end) in enumerate(shards):
            weights[target][start:end].fill_(i + 1)
        embed = "model.embed_tokens.weight"
        for mode in ("embedding_only", "head_only", "nested_head", "both", "conflict", "missing_ffn", "indexed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                checkpoint = {name: value.clone() for name, value in weights.items()}
                for source_name, target, start, end in shards:
                    checkpoint[source_name] = weights[target][start:end].clone()
                    checkpoint.pop(target, None)
                if mode in ("head_only", "nested_head", "both", "conflict"):
                    name = (
                        "model.language_model.lm_head.weight"
                        if mode == "nested_head"
                        else "lm_head.weight"
                    )
                    checkpoint[name] = checkpoint[embed].clone()
                if mode in ("head_only", "nested_head"):
                    del checkpoint[embed]
                if mode == "conflict":
                    checkpoint["lm_head.weight"].add_(1)
                if mode == "missing_ffn":
                    del checkpoint["model.layers.0.mlp.gate_proj.weight"]
                save_file(checkpoint, str(Path(folder) / "model.safetensors"))
                if mode == "indexed":
                    (Path(folder) / "model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": {name: "model.safetensors" for name in checkpoint}})
                    )
                    save_file({"unrelated": torch.ones(1)}, str(Path(folder) / "backup.safetensors"))
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
