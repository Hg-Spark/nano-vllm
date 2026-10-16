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

    def test_gemma_fused_add_rmsnorm_matches_reference(self):
        from nanovllm.models.qwen3_5_layers import Qwen3_5RMSNorm

        for dtype in self._supported_dtypes():
            norm = Qwen3_5RMSNorm(128).to(device="cuda", dtype=dtype)
            with torch.no_grad():
                norm.weight.uniform_(-0.2, 0.2)
                x = torch.randn(5, 128, device="cuda", dtype=dtype)
                residual = torch.randn_like(x)
                summed = (x.float() + residual.float()).to(dtype)
                expected = (
                    summed.float()
                    * torch.rsqrt(summed.float().square().mean(dim=-1, keepdim=True) + norm.eps)
                    * (1.0 + norm.weight.float())
                ).to(dtype)
                actual = norm.forward_with_residual(x, residual)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(residual, summed, rtol=2e-2, atol=2e-2)

    def test_decoder_residual_stream_matches_unfused_reference(self):
        from nanovllm.models.qwen3_5 import Qwen3_5DecoderLayer

        class ScaleAttention(torch.nn.Module):
            def forward(self, positions, hidden_states):
                return hidden_states * 0.125

        config = SimpleNamespace(
            layer_types=["full_attention"],
            model_type="qwen3_5_text",
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=False,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            intermediate_size=256,
        )
        for dtype in self._supported_dtypes():
            layer = Qwen3_5DecoderLayer(config, 0, 32, (32, 10000.0, None)).to(
                device="cuda", dtype=dtype
            )
            layer.self_attn = ScaleAttention()
            layer.mlp = torch.nn.Identity()
            with torch.no_grad():
                layer.input_layernorm.weight.uniform_(-0.2, 0.2)
                layer.post_attention_layernorm.weight.uniform_(-0.2, 0.2)
                initial = torch.randn(4, 128, device="cuda", dtype=dtype)
                reference = initial.clone()
                for _ in range(2):
                    normalized = layer.input_layernorm(reference)
                    merged = reference + normalized * 0.125
                    reference = merged + layer.post_attention_layernorm(merged)
                hidden, residual = initial.clone(), None
                for _ in range(2):
                    hidden, residual = layer(torch.zeros(3, 4, device="cuda"), hidden, residual)
                actual = hidden + residual
                torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)

    def test_fused_swiglu_matches_reference(self):
        from nanovllm.models.qwen3_5_layers import Qwen3_5MLP

        config = SimpleNamespace(hidden_size=128, hidden_act="silu")
        for dtype in self._supported_dtypes():
            with self.subTest(dtype=dtype):
                mlp = Qwen3_5MLP(config, 256).to(device="cuda", dtype=dtype)
                hidden_states = torch.randn(7, 128, device="cuda", dtype=dtype)
                with torch.no_grad():
                    gate, up = mlp.gate_up_proj(hidden_states).chunk(2, dim=-1)
                    expected = mlp.down_proj(F.silu(gate) * up)
                    actual = mlp(hidden_states)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class FlashInferRuntimeOpsTest(unittest.TestCase):
    def test_fp8_kv_append_with_explicit_scales(self):
        from nanovllm.layers.attention import store_kv_cache
        from nanovllm.utils.context import Context

        key = torch.tensor([2.0, -4.0], device="cuda", dtype=torch.float16).repeat(128)
        key = key.reshape(2, 1, 128)
        value = key * 1.5
        k_cache = torch.empty((2, 16, 1, 128), device="cuda", dtype=torch.float8_e4m3fn)
        v_cache = torch.empty_like(k_cache)
        context = Context(
            kv_batch_indices=torch.tensor([0, 0], dtype=torch.int32, device="cuda"),
            kv_positions=torch.tensor([15, 16], dtype=torch.int32, device="cuda"),
            paged_kv_indices=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
            paged_kv_indptr=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
            paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
        )
        store_kv_cache(key, value, k_cache, v_cache, context, k_scale=2.0, v_scale=3.0)
        stored_key = torch.stack((k_cache[1, 15].float(), k_cache[0, 0].float())) * 2.0
        stored_value = torch.stack((v_cache[1, 15].float(), v_cache[0, 0].float())) * 3.0
        torch.testing.assert_close(stored_key, key.float(), atol=0.125, rtol=0.125)
        torch.testing.assert_close(stored_value, value.float(), atol=0.125, rtol=0.125)

    def test_pinned_metadata_upload_reuses_and_grows_cpu_buffers(self):
        from nanovllm.engine.batch_common import PinnedMetadataPool

        pool = PinnedMetadataPool()
        try:
            with torch.device("cuda"):
                for dtype in ("int32", "int64", "float32"):
                    for values in ([1, 2], [3], [4, 5, 6]):
                        with self.subTest(dtype=dtype, values=values):
                            result = pool.to_cuda(dtype, values, dtype)
                            expected = torch.tensor(
                                values, dtype=getattr(torch, dtype), device="cpu"
                            )
                            torch.testing.assert_close(result.cpu(), expected)
                            self.assertTrue(result.is_cuda)
                            self.assertEqual(pool.buffers[dtype].device.type, "cpu")
                            self.assertTrue(pool.buffers[dtype].is_pinned())
        finally:
            pool.clear()


if __name__ == "__main__":
    unittest.main()
