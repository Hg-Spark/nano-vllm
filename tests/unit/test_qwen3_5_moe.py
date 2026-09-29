import unittest
from types import SimpleNamespace

import torch

from nanovllm.models.qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTopKRouter,
)
from nanovllm.utils.context import Context, use_context
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


class Qwen35MoeTest(unittest.TestCase):

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

    def test_packed_experts_match_tokenwise_reference(self):
        experts = Qwen3_5MoeExperts(self.config)
        hidden_states = torch.randn(4, self.config.hidden_size)
        selected_experts = torch.tensor([
            [0, 1],
            [2, 0],
            [1, 3],
            [3, 2],
        ])
        routing_weights = torch.tensor([
            [0.7, 0.3],
            [0.6, 0.4],
            [0.55, 0.45],
            [0.8, 0.2],
        ])

        actual = experts(
            hidden_states,
            selected_experts,
            routing_weights,
        )

        expected = torch.zeros_like(hidden_states)
        for token_idx in range(hidden_states.size(0)):
            for topk_idx in range(self.config.num_experts_per_tok):
                expert_idx = selected_experts[token_idx, topk_idx]
                gate_up = torch.nn.functional.linear(
                    hidden_states[token_idx],
                    experts.gate_up_proj[expert_idx],
                )
                up, gate = gate_up.chunk(2, dim=-1)
                intermediate = torch.nn.functional.silu(gate) * up
                current = torch.nn.functional.linear(
                    intermediate,
                    experts.down_proj[expert_idx],
                )
                expected[token_idx] += (
                    routing_weights[token_idx, topk_idx]
                    * current
                )

        torch.testing.assert_close(actual, expected)

    def test_decode_expert_path_matches_prefill_reference(self):
        experts = Qwen3_5MoeExperts(self.config)
        hidden_states = torch.randn(5, self.config.hidden_size)
        selected_experts = torch.tensor([
            [0, 1],
            [2, 0],
            [1, 3],
            [3, 2],
            [0, 3],
        ])
        routing_weights = torch.tensor([
            [0.7, 0.3],
            [0.6, 0.4],
            [0.55, 0.45],
            [0.8, 0.2],
            [0.25, 0.75],
        ])

        with use_context(Context(is_prefill=True)):
            expected = experts(
                hidden_states,
                selected_experts,
                routing_weights,
            )
        with use_context(Context(is_prefill=False)):
            actual = experts(
                hidden_states,
                selected_experts,
                routing_weights,
            )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 is required",
    )
    def test_cuda_bf16_decode_experts_match_prefill_reference(self):
        experts = Qwen3_5MoeExperts(self.config).cuda().to(torch.bfloat16)
        hidden_states = torch.randn(
            5,
            self.config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        selected_experts = torch.tensor(
            [
                [0, 1],
                [2, 0],
                [1, 3],
                [3, 2],
                [0, 3],
            ],
            device="cuda",
        )
        routing_weights = torch.tensor(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.55, 0.45],
                [0.8, 0.2],
                [0.25, 0.75],
            ],
            dtype=torch.bfloat16,
            device="cuda",
        )

        with use_context(Context(is_prefill=True)):
            expected = experts(
                hidden_states,
                selected_experts,
                routing_weights,
            )
        with use_context(Context(is_prefill=False)):
            actual = experts(
                hidden_states,
                selected_experts,
                routing_weights,
            )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-2,
            atol=2e-2,
        )

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

        transformed = Qwen3_5MoeForCausalLM.transform_checkpoint_weight(
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

    def test_sparse_block_parameter_names_match_checkpoint_layout(self):
        block = Qwen3_5MoeSparseMoeBlock(self.config)
        names = dict(block.named_parameters())

        self.assertIn("gate.weight", names)
        self.assertIn("experts.gate_up_proj", names)
        self.assertIn("experts.down_proj", names)
        self.assertIn("shared_expert.gate_proj.weight", names)
        self.assertIn("shared_expert.up_proj.weight", names)
        self.assertIn("shared_expert.down_proj.weight", names)
        self.assertIn("shared_expert_gate.weight", names)

    def test_model_exposes_explicit_hybrid_cache_topology(self):
        model = Qwen3_5MoeForCausalLM(self.config)

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
                Qwen3_5MoeForCausalLM,
                "model.language_model.layers.0.mlp.gate.weight",
            ),
            "model.layers.0.mlp.gate.weight",
        )
        self.assertEqual(
            _map_weight_name(
                Qwen3_5MoeForCausalLM,
                "model.visual.blocks.0.weight",
            ),
            "visual.blocks.0.weight",
        )
        self.assertIsNone(
            _map_weight_name(
                Qwen3_5MoeForCausalLM,
                "mtp.layers.0.weight",
            )
        )


if __name__ == "__main__":
    unittest.main()
