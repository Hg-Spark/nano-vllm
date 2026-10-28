"""Real FlashInfer CUTLASS MoE validation; requires a supported CUDA GPU.

These tests deliberately keep the independent expert formula in test code.
No checkpoint download or reference runtime backend is required.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

from nanovllm.models.qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5SparseMoeBlock,
)

SUPPORTED_ARCHS = {89, 90, 100, 103, 107, 110, 120, 121}


def _independent_experts(x, ids, scales, gate_up, down):
    """Checkpoint-independent FP32 per-token SwiGLU oracle on CPU."""
    x = x.detach().float().cpu()
    ids = ids.detach().cpu()
    scales = scales.detach().float().cpu()
    gate_up = gate_up.detach().float().cpu()
    down = down.detach().float().cpu()
    output = torch.zeros_like(x)
    for token in range(x.size(0)):
        for choice in range(ids.size(1)):
            expert = int(ids[token, choice])
            up, gate = F.linear(x[token], gate_up[expert]).chunk(2)
            value = F.linear(F.silu(gate) * up, down[expert])
            output[token] += scales[token, choice] * value
    return output


@unittest.skipUnless(torch.cuda.is_available(), "FlashInfer MoE requires a CUDA GPU")
class FlashInferMoEIntegrationTest(unittest.TestCase):
    HIDDEN = 256
    INTERMEDIATE = 128
    EXPERTS = 16
    TOP_K = 4
    CAPACITY = 128

    def setUp(self):
        major, minor = torch.cuda.get_device_capability()
        if major * 10 + minor not in SUPPORTED_ARCHS:
            self.skipTest(f"FlashInfer CUTLASS MoE does not support SM{major}{minor}")
        torch.manual_seed(19)
        self.config = SimpleNamespace(
            hidden_size=self.HIDDEN,
            moe_intermediate_size=self.INTERMEDIATE,
            shared_expert_intermediate_size=self.INTERMEDIATE,
            num_experts=self.EXPERTS,
            num_experts_per_tok=self.TOP_K,
            hidden_act="silu",
        )

    def _make_experts(self, dtype, *, capacity=CAPACITY):
        experts = Qwen3_5MoeExperts(self.config).cuda().to(dtype)
        with torch.no_grad():
            experts.gate_up_proj.normal_(std=0.05)
            experts.down_proj.normal_(std=0.05)
        self._bind_expert_buffers(experts, dtype, capacity=capacity)
        return experts

    def _bind_expert_buffers(self, experts, dtype, *, capacity=CAPACITY):
        workspace_bytes = cutlass_fused_moe_workspace_size(
            capacity,
            self.HIDDEN,
            self.INTERMEDIATE,
            self.EXPERTS,
            self.TOP_K,
            x_dtype=dtype,
            weight_dtype=dtype,
            output_dtype=dtype,
            use_fused_finalize=False,
            device=torch.device("cuda"),
        )
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        output = torch.empty(capacity, self.HIDDEN, dtype=dtype, device="cuda")
        experts.bind_runtime_buffers(workspace, output)

    def _batch(self, tokens, dtype, *, offset=0):
        x = torch.randn(tokens, self.HIDDEN, dtype=dtype, device="cuda")
        # Distinct experts per token, with a reproducible route permutation.
        ids = (
            torch.arange(tokens, device="cuda")[:, None] * 7
            + torch.arange(self.TOP_K, device="cuda")[None, :]
            + offset
        ) % self.EXPERTS
        weights = torch.rand(tokens, self.TOP_K, device="cuda")
        weights /= weights.sum(dim=-1, keepdim=True)
        return x, ids.to(torch.int64), weights.to(dtype)

    def _assert_reference(self, experts, batch, *, atol):
        x, ids, scales = batch
        expected = _independent_experts(x, ids, scales, experts.gate_up_proj, experts.down_proj)
        actual = experts(x, ids, scales)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual.float().cpu(), expected, atol=atol, rtol=0.05)

    @torch.inference_mode()
    def test_fp16_bf16_prefill_and_decode_against_independent_formula(self):
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with self.subTest(dtype=dtype):
                experts = self._make_experts(dtype)
                # 1-token decode, partial tiles and a packed prefill.
                for tokens in (1, 7, 65):
                    with self.subTest(tokens=tokens):
                        self._assert_reference(
                            experts,
                            self._batch(tokens, dtype, offset=tokens),
                            atol=0.025 if dtype == torch.float16 else 0.07,
                        )

    @torch.inference_mode()
    def test_shared_expert_matches_independent_formula(self):
        dtype = torch.float16
        block = Qwen3_5SparseMoeBlock(self.config).cuda().to(dtype)
        with torch.no_grad():
            for param in block.parameters():
                param.normal_(std=0.05)
        self._bind_expert_buffers(block.experts, dtype)
        x = torch.randn(9, self.HIDDEN, dtype=dtype, device="cuda")
        weights, ids = block.gate(x)
        routed = _independent_experts(
            x, ids, weights, block.experts.gate_up_proj, block.experts.down_proj
        )
        shared = block.shared_expert
        x32 = x.float().cpu()
        gate_up = F.linear(x32, shared.gate_up_proj.weight.float().cpu())
        gate, up = gate_up.chunk(2, dim=-1)
        down = F.linear(F.silu(gate) * up, shared.down_proj.weight.float().cpu())
        shared_gate = F.linear(x32, block.shared_expert_gate.weight.float().cpu()).sigmoid()
        expected = routed + shared_gate * down
        actual = block(x)
        torch.testing.assert_close(actual.float().cpu(), expected, atol=0.035, rtol=0.05)

    @torch.inference_mode()
    def test_cuda_graph_replay_uses_updated_routes(self):
        dtype = torch.float16
        experts = self._make_experts(dtype)
        x_static = torch.empty(4, self.HIDDEN, dtype=dtype, device="cuda")
        ids_static = torch.empty(4, self.TOP_K, dtype=torch.int64, device="cuda")
        scales_static = torch.empty(4, self.TOP_K, dtype=dtype, device="cuda")
        x, ids, scales = self._batch(4, dtype)
        x_static.copy_(x)
        ids_static.copy_(ids)
        scales_static.copy_(scales)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                experts(x_static, ids_static, scales_static)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = experts(x_static, ids_static, scales_static)

        previous = None
        for offset in (0, 5, 11):
            x, ids, scales = self._batch(4, dtype, offset=offset)
            x_static.copy_(x)
            ids_static.copy_(ids)
            scales_static.copy_(scales)
            graph.replay()
            actual = graph_output.clone()
            expected = _independent_experts(x, ids, scales, experts.gate_up_proj, experts.down_proj)
            torch.testing.assert_close(actual.float().cpu(), expected, atol=0.025, rtol=0.05)
            if previous is not None:
                self.assertFalse(torch.equal(actual, previous))
            previous = actual


if __name__ == "__main__":
    unittest.main()
