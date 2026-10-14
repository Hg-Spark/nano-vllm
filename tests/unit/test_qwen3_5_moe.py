import unittest
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.models.qwen3_5_layers import Qwen3_5Attention
from nanovllm.models.qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeTopKRouter,
    Qwen3_5SparseMoeBlock,
)
from nanovllm.utils.loader import _map_weight_name


def make_text_config():
    return SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
        shared_expert_intermediate_size=5,
        layer_types=["linear_attention", "full_attention"],
        num_hidden_layers=2,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        max_position_embeddings=1024,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_bias=False,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
    )


class Qwen3_5MoeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = make_text_config()

    def test_attention_reads_readonly_rope_mapping(self):
        config = make_text_config()
        config.head_dim = 12
        config.rope_parameters = MappingProxyType(
            {
                "partial_rotary_factor": 1.0,
                "rope_theta": 20000.0,
                "mrope_section": (2, 2, 2),
            }
        )
        with patch("nanovllm.models.qwen3_5_layers.get_rope") as get_rope:
            Qwen3_5Attention(config)
        self.assertEqual(get_rope.call_args.kwargs["base"], 20000.0)
        self.assertEqual(get_rope.call_args.kwargs["rotary_dim"], 12)
        self.assertEqual(get_rope.call_args.kwargs["mrope_section"], (2, 2, 2))

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

    def test_flashinfer_forward_argument_contract(self):
        experts = Qwen3_5MoeExperts(self.config)
        workspace = torch.empty(16, dtype=torch.uint8)
        output = torch.empty(8, self.config.hidden_size)
        experts.bind_runtime_buffers(workspace, output)
        hidden = torch.randn(4, self.config.hidden_size)
        ids = torch.tensor([[0, 1], [2, 0], [1, 3], [3, 2]])
        weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]])

        with patch(
            "nanovllm.models.qwen3_5_moe.cutlass_fused_moe",
            side_effect=lambda *args, **kwargs: kwargs["output"].fill_(2),
        ) as fused:
            actual = experts(hidden, ids, weights)

        torch.testing.assert_close(actual, torch.full_like(hidden, 2))
        fused.assert_called_once()
        args, kwargs = fused.call_args
        torch.testing.assert_close(args[0], hidden)
        torch.testing.assert_close(args[1], ids.to(torch.int32))
        torch.testing.assert_close(args[2], weights.float())
        self.assertIs(args[3], experts.gate_up_proj)
        self.assertIs(args[4], experts.down_proj)
        self.assertEqual(kwargs["tune_max_num_tokens"], 8)
        self.assertIs(kwargs["workspace_buffer"], workspace)
        self.assertFalse(kwargs["use_fused_finalize"])

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

        transformed = Qwen3_5ForCausalLM.transform_checkpoint_weight(
            "model.layers.0.mlp.experts.gate_up_proj",
            weight,
        )

        torch.testing.assert_close(
            transformed[:, :split],
            original[:, split:],
        )
        torch.testing.assert_close(
            transformed[:, split:],
            original[:, :split],
        )

    def test_model_packs_shared_expert_projections(self):
        model = Qwen3_5ForCausalLM(self.config)
        expected = [
            torch.cat(
                (
                    layer.mlp.shared_expert.gate_proj.weight,
                    layer.mlp.shared_expert.up_proj.weight,
                ),
                dim=0,
            ).clone()
            for layer in model.model.layers
        ]

        model.pack_linear_projections()

        for layer, weight in zip(model.model.layers, expected):
            shared = layer.mlp.shared_expert
            self.assertFalse(hasattr(shared, "gate_proj"))
            self.assertFalse(hasattr(shared, "up_proj"))
            torch.testing.assert_close(shared.gate_up_proj.weight, weight, rtol=0, atol=0)

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
        self.assertIn("shared_expert.gate_proj.weight", names)
        self.assertIn("shared_expert.up_proj.weight", names)
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
