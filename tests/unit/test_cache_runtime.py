import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.engine.cache_runtime import (
    _resolve_kv_cache_dtype,
    _visual_feature_cache_nbytes,
    allocate_runtime_caches,
)


class CacheRuntimeTest(unittest.TestCase):
    def test_reservation_uses_enforced_limit(self):
        config = SimpleNamespace(
            root_config=SimpleNamespace(vision_config=object()),
            max_model_len=512,
            max_num_seqs=4,
            text_config=SimpleNamespace(hidden_size=64),
            max_visual_tokens_per_request=None,
        )
        self.assertEqual(_visual_feature_cache_nbytes(config, torch.float16), 4 * 512 * 64 * 2)
        config.max_visual_tokens_per_request = 32
        self.assertEqual(_visual_feature_cache_nbytes(config, torch.float16), 4 * 32 * 64 * 2)
        config.root_config.vision_config = None
        self.assertEqual(_visual_feature_cache_nbytes(config, torch.float16), 0)

    def test_cache_dtype_mapping(self):
        for requested, expected in (
            ("auto", torch.bfloat16),
            ("fp8_e4m3", torch.float8_e4m3fn),
        ):
            with self.subTest(dtype=requested):
                config = SimpleNamespace(kv_cache_dtype=requested)
                self.assertEqual(_resolve_kv_cache_dtype(config, torch.bfloat16), expected)

    def test_static_kv_budget_uses_available_memory(self):
        layer = SimpleNamespace(k_cache=None, v_cache=None)
        state = SimpleNamespace(
            state_cache_nbytes=lambda slots: 128,
            # 64B recurrent + 64B convolution history per live slot.
            num_v_heads=1,
            head_k_dim=4,
            head_v_dim=4,
            recurrent_state=torch.empty(0, dtype=torch.float32),
            allocate_state_cache=Mock(),
        )
        model = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float16))
            ),
            kv_cache_modules=lambda: [layer],
            state_cache_modules=lambda: [state],
        )
        config = SimpleNamespace(
            gpu_memory_utilization=0.75,
            activation_reserve_bytes=0,
            decode_graph_batch_sizes=(),
            kv_cache_dtype="auto",
            kv_cache_block_size=16,
            max_num_seqs=2,
            max_prefix_cache_entries=0,
            max_prefix_cache_snapshot_bytes=None,
            max_num_batched_tokens=512,
            text_config=SimpleNamespace(num_key_value_heads=1, head_dim=8),
            head_dim=8,
            root_config=SimpleNamespace(vision_config=None),
            kv_cache_k_scale=1.0,
            kv_cache_v_scale=1.0,
        )
        with (
            patch(
                "nanovllm.engine.cache_runtime.torch.cuda.mem_get_info",
                return_value=(4096, 8192),
            ),
            patch(
                "nanovllm.engine.cache_runtime.torch.empty",
                return_value=MagicMock(),
            ) as allocation,
        ):
            _, num_blocks = allocate_runtime_caches(model, config)

        # 8192 * .75 - (8192 - 4096) - 128 = 1920; each block takes 512.
        self.assertEqual(num_blocks, 3)
        allocation.assert_called_once_with(2, 1, 3, 16, 1, 8, device="cuda", dtype=torch.float16)
        state.allocate_state_cache.assert_called_once_with(2)

        # Final-state snapshot stacking reserves temporary GPU memory only.
        config.max_prefix_cache_entries = 2
        config.max_num_batched_tokens = 64
        with (
            patch(
                "nanovllm.engine.cache_runtime.torch.cuda.mem_get_info",
                return_value=(4736, 8192),
            ),
            patch(
                "nanovllm.engine.cache_runtime.torch.empty",
                return_value=MagicMock(),
            ) as allocation,
        ):
            _, num_blocks = allocate_runtime_caches(model, config)
        self.assertEqual(num_blocks, 4)
        self.assertEqual(allocation.call_args.args[2], 4)

        # Graph memory is deducted from KV capacity before graph capture.
        config.max_prefix_cache_entries = 0
        config.decode_graph_batch_sizes = (1,)
        with (
            patch("nanovllm.engine.cache_runtime.DECODE_GRAPH_RESERVE_BYTES_PER_BUCKET", 512),
            patch(
                "nanovllm.engine.cache_runtime.torch.cuda.mem_get_info",
                return_value=(4096, 8192),
            ),
            patch("nanovllm.engine.cache_runtime.torch.empty", return_value=MagicMock()),
        ):
            _, num_blocks = allocate_runtime_caches(model, config)
        self.assertEqual(num_blocks, 2)

        # A reserved peak-activation budget reduces KV capacity before allocation.
        config.decode_graph_batch_sizes = ()
        config.activation_reserve_bytes = 1024
        with (
            patch(
                "nanovllm.engine.cache_runtime.torch.cuda.mem_get_info",
                return_value=(4096, 8192),
            ),
            patch("nanovllm.engine.cache_runtime.torch.empty", return_value=MagicMock()),
        ):
            _, num_blocks = allocate_runtime_caches(model, config)
        self.assertEqual(num_blocks, 1)


if __name__ == "__main__":
    unittest.main()
