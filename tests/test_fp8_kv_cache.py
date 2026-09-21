import unittest

import torch

from nanovllm.engine.model_runner import _resolve_kv_cache_dtype
from nanovllm.layers.attention import store_kvcache


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
