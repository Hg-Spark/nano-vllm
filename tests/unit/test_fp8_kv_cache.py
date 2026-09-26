import unittest
from types import SimpleNamespace

import torch
from flash_attn import flash_attn_func

from nanovllm.engine.cache_runtime import _resolve_kv_cache_dtype
from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.fp8_kv import (
    _materialize_paged_cache,
    fp8_paged_attention_reference,
)


class _Config:
    kv_cache_dtype = "auto"


class FP8KVCacheTest(unittest.TestCase):

    def test_cache_dtype_auto_follows_model_dtype(self):
        config = _Config()
        self.assertEqual(
            _resolve_kv_cache_dtype(config, torch.bfloat16),
            torch.bfloat16,
        )

    def test_cache_dtype_fp8_maps_to_e4m3(self):
        config = _Config()
        config.kv_cache_dtype = "fp8_e4m3"
        self.assertEqual(
            _resolve_kv_cache_dtype(config, torch.bfloat16),
            torch.float8_e4m3fn,
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for FP8 paged-cache reads",
    )
    def test_fp8_materialize_reads_selected_blocks_and_scale(self):
        cache = torch.tensor(
            [
                [[[1.0, 2.0]], [[3.0, 4.0]]],
                [[[9.0, 10.0]], [[11.0, 12.0]]],
                [[[5.0, 6.0]], [[7.0, 8.0]]],
            ],
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        block_table = torch.tensor(
            [2, 0],
            device="cuda",
            dtype=torch.int32,
        )

        actual = _materialize_paged_cache(
            cache,
            block_table,
            seqlen=3,
            scale=2.0,
            output_dtype=torch.bfloat16,
        )
        expected = torch.tensor(
            [
                [[10.0, 12.0]],
                [[14.0, 16.0]],
                [[2.0, 4.0]],
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )

        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for FP8 GQA decode",
    )
    def test_fp8_paged_decode_matches_dense_gqa_reference(self):
        torch.manual_seed(0)
        block_size = 2
        num_kv_heads = 2
        num_q_heads = 4
        head_dim = 64
        k_scale = 0.5
        v_scale = 0.25

        k_dense = torch.randn(
            4,
            block_size,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        v_dense = torch.randn_like(k_dense)
        k_cache = (k_dense / k_scale).to(torch.float8_e4m3fn)
        v_cache = (v_dense / v_scale).to(torch.float8_e4m3fn)
        block_tables = torch.tensor(
            [
                [2, 0, -1],
                [1, 3, 0],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        prefix_lens = (2, 4)
        q = torch.randn(
            2,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        context = SimpleNamespace(
            is_prefill=False,
            block_tables=block_tables,
            state_prefix_lens=prefix_lens,
            prefill_q_offsets=None,
        )

        actual = fp8_paged_attention_reference(
            q,
            k_cache,
            v_cache,
            context,
            softmax_scale=head_dim ** -0.5,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        expected = []
        for idx, prefix_len in enumerate(prefix_lens):
            seqlen = prefix_len + 1
            k_i = _materialize_paged_cache(
                k_cache,
                block_tables[idx],
                seqlen,
                k_scale,
                q.dtype,
            )
            v_i = _materialize_paged_cache(
                v_cache,
                block_tables[idx],
                seqlen,
                v_scale,
                q.dtype,
            )
            expected.append(
                flash_attn_func(
                    q[idx:idx + 1].unsqueeze(1),
                    k_i.unsqueeze(0),
                    v_i.unsqueeze(0),
                    softmax_scale=head_dim ** -0.5,
                    causal=True,
                ).squeeze(1)
            )
        expected = torch.cat(expected, dim=0)

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-2,
            atol=1e-2,
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for the Triton KV store kernel",
    )
    def test_fp8_store_round_trip_with_explicit_scales(self):
        key = torch.tensor(
            [[[2.0, -4.0]]],
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.tensor(
            [[[3.0, -6.0]]],
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_cache = torch.empty(
            (1, 1, 1, 2),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        v_cache = torch.empty_like(k_cache)
        slot_mapping = torch.tensor(
            [0],
            device="cuda",
            dtype=torch.int32,
        )

        store_kvcache(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            k_scale=2.0,
            v_scale=3.0,
        )

        torch.testing.assert_close(
            k_cache.to(torch.bfloat16) * 2.0,
            key,
            atol=0.125,
            rtol=0.125,
        )
        torch.testing.assert_close(
            v_cache.to(torch.bfloat16) * 3.0,
            value,
            atol=0.125,
            rtol=0.125,
        )


if __name__ == "__main__":
    unittest.main()
