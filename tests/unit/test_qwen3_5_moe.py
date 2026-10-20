import unittest

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.models.qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeTopKRouter,
    Qwen3_5SparseMoeBlock,
)
from nanovllm.utils.loader import _map_weight_name
from tests.qwen35_helpers import make_text_config


class Qwen3_5MoeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = make_text_config()

    def test_router_topk_weights_are_normalized(self):
        router = Qwen3_5MoeTopKRouter(self.config)
        hidden_states = torch.randn(6, self.config.hidden_size)

        routing_weights, selected_experts = router(hidden_states)

        self.assertEqual(
            routing_weights.shape,
            (6, self.config.num_experts_per_tok),
        )
        self.assertEqual(
            selected_experts.shape,
            (6, self.config.num_experts_per_tok),
        )
        torch.testing.assert_close(
            routing_weights.float().sum(dim=-1),
            torch.ones(6),
        )

    def test_flashinfer_requires_buffers_and_capacity(self):
        experts = Qwen3_5MoeExperts(self.config)
        hidden = torch.randn(2, self.config.hidden_size)
        ids = torch.tensor([[0, 1], [2, 3]])
        weights = torch.full((2, 2), 0.5)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            experts(hidden, ids, weights)
        experts.bind_runtime_buffers(torch.empty(16, dtype=torch.uint8), torch.empty(1, 8))
        with self.assertRaisesRegex(RuntimeError, "exceeds allocated"):
            experts(hidden, ids, weights)

    def test_checkpoint_gate_up_rows_are_reordered_for_runtime(self):
        weight = torch.arange(
            self.config.num_experts
            * 2
            * self.config.moe_intermediate_size
            * self.config.hidden_size,
            dtype=torch.float32,
        ).reshape(
            self.config.num_experts,
            2 * self.config.moe_intermediate_size,
            self.config.hidden_size,
        )
        original = weight.clone()
        split = self.config.moe_intermediate_size

        transformed = torch.empty_like(weight)
        Qwen3_5ForCausalLM.copy_checkpoint_weight(
            "model.layers.0.mlp.experts.gate_up_proj", transformed, weight
        )

        torch.testing.assert_close(weight, original)
        torch.testing.assert_close(
            transformed[:, :split],
            original[:, split:],
        )
        torch.testing.assert_close(
            transformed[:, split:],
            original[:, :split],
        )

    def test_shared_expert_initializes_packed_weights(self):
        model = Qwen3_5ForCausalLM(self.config)
        for layer in model.model.layers:
            shared = layer.mlp.shared_expert
            self.assertFalse(hasattr(shared, "gate_proj"))
            self.assertFalse(hasattr(shared, "up_proj"))
            self.assertEqual(
                shared.gate_up_proj.weight.shape[0],
                2 * self.config.shared_expert_intermediate_size,
            )
        shards = model.packed_checkpoint_shards()
        self.assertIn("model.layers.0.mlp.shared_expert.gate_proj.weight", shards)

    def test_model_binds_shared_moe_buffers(self):
        model = Qwen3_5ForCausalLM(self.config)
        workspace = torch.empty(16, dtype=torch.uint8)
        output = torch.empty(8, self.config.hidden_size)
        model.bind_moe_runtime(workspace, output)
        for layer in model.model.layers:
            self.assertIs(layer.mlp.experts._moe_workspace, workspace)
            self.assertIs(layer.mlp.experts._moe_output, output)

    def test_sparse_block_parameter_names_match_checkpoint_layout(self):
        block = Qwen3_5SparseMoeBlock(self.config)
        names = dict(block.named_parameters())

        self.assertIn("gate.weight", names)
        self.assertIn("experts.gate_up_proj", names)
        self.assertIn("experts.down_proj", names)
        self.assertIn("shared_expert.gate_up_proj.weight", names)
        self.assertIn("shared_expert.down_proj.weight", names)
        self.assertIn("shared_expert_gate.weight", names)

    def test_model_exposes_explicit_hybrid_cache_topology(self):
        model = Qwen3_5ForCausalLM(self.config)

        kv_modules = model.kv_cache_modules()
        state_modules = model.state_cache_modules()

        self.assertEqual(len(kv_modules), 1)
        self.assertEqual(len(state_modules), 1)
        self.assertIs(
            kv_modules[0],
            model.model.layers[1].self_attn.attn,
        )
        self.assertIs(
            state_modules[0],
            model.model.layers[0].linear_attn.state_pool,
        )

    def test_text_checkpoint_prefix_mapping(self):
        self.assertEqual(
            _map_weight_name(
                Qwen3_5ForCausalLM,
                "model.language_model.layers.0.mlp.gate.weight",
            ),
            "model.layers.0.mlp.gate.weight",
        )
        self.assertEqual(
            _map_weight_name(
                Qwen3_5ForCausalLM,
                "model.visual.blocks.0.weight",
            ),
            "visual.blocks.0.weight",
        )
        self.assertIsNone(
            _map_weight_name(
                Qwen3_5ForCausalLM,
                "mtp.layers.0.weight",
            )
        )


if __name__ == "__main__":
    unittest.main()
