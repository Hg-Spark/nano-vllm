import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires torch and CUDA")
class Qwen35FlashInferOpsTest(unittest.TestCase):
    @staticmethod
    def _supported_dtypes():
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        return dtypes

    def test_gemma_rmsnorm_matches_reference(self):
        from nanovllm.models.qwen3_5_layers import Qwen3_5RMSNorm

        for dtype in self._supported_dtypes():
            norm = Qwen3_5RMSNorm(128).to(device="cuda", dtype=dtype)
            with torch.no_grad():
                norm.weight.uniform_(-0.2, 0.2)
            inputs = (
                torch.randn(7, 128, device="cuda", dtype=dtype),
                torch.randn(3, 4, 256, device="cuda", dtype=dtype)[..., :128],
            )
            for x in inputs:
                with self.subTest(dtype=dtype, shape=tuple(x.shape)):
                    value = x.float()
                    expected = (
                        value
                        * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + norm.eps)
                        * (1.0 + norm.weight.float())
                    ).to(dtype)
                    torch.testing.assert_close(norm(x), expected, rtol=2e-2, atol=2e-2)

    def test_fused_swiglu_matches_reference(self):
        from nanovllm.models.qwen3_5_layers import Qwen3_5MLP

        config = SimpleNamespace(hidden_size=128, hidden_act="silu")
        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                mlp = Qwen3_5MLP(config, 256).to(device="cuda", dtype=dtype)
                hidden_states = torch.randn(7, 128, device="cuda", dtype=dtype)
                with torch.no_grad():
                    expected = mlp.down_proj(
                        F.silu(mlp.gate_proj(hidden_states)) * mlp.up_proj(hidden_states)
                    )
                    mlp.pack_projections()
                    actual = mlp(hidden_states)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
