import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.engine.cache_runtime import (
    GDNCompletionBuffer,
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
            patch("nanovllm.engine.cache_runtime.DECODE_GRAPH_RESERVE_BYTES_PER_BUCKET", 512),
            patch("nanovllm.engine.cache_runtime.torch.cuda.mem_get_info") as memory,
            patch(
                "nanovllm.engine.cache_runtime.torch.empty", return_value=MagicMock()
            ) as allocation,
        ):
            memory.return_value = (4096, 8192)
            _, num_blocks = allocate_runtime_caches(model, config)
            self.assertEqual(num_blocks, 3)
            allocation.assert_called_once_with(
                2, 1, 3, 16, 1, 8, device="cuda", dtype=torch.float16
            )
            state.allocate_state_cache.assert_called_once_with(2)

            # A temporary snapshot workspace uses available memory, not resident KV capacity.
            config.max_prefix_cache_entries = 2
            config.max_num_batched_tokens = 64
            memory.return_value = (4736, 8192)
            _, num_blocks = allocate_runtime_caches(model, config)
            self.assertEqual(num_blocks, 4)
            self.assertEqual(allocation.call_args.args[2], 4)

            # Graph and activation reservations reduce the KV budget.
            config.max_prefix_cache_entries = 0
            config.decode_graph_batch_sizes = (1,)
            memory.return_value = (4096, 8192)
            _, num_blocks = allocate_runtime_caches(model, config)
            self.assertEqual(num_blocks, 2)

            config.decode_graph_batch_sizes = ()
            config.activation_reserve_bytes = 1024
            _, num_blocks = allocate_runtime_caches(model, config)
            self.assertEqual(num_blocks, 1)

    def test_completion_buffer_rolls_without_host_snapshots_until_finish(self):
        from nanovllm.engine.sequence import Sequence

        class Layer:
            def __init__(self):
                self.conv_state = torch.zeros((2, 3, 4), dtype=torch.bfloat16)
                self.recurrent_state = torch.zeros((2, 2, 4), dtype=torch.float32)

            def slot(self, slot):
                return self.conv_state[slot], self.recurrent_state[slot]

        layers = [Layer(), Layer()]
        cache = GDNCompletionBuffer(layers)
        first = Sequence([1] * 81)
        first.state_slot = 0
        for layer in layers:
            layer.conv_state[0].fill_(1)
            layer.recurrent_state[0].fill_(2)
        cache.update(layers, [(first, 64)])
        self.assertEqual(cache._positions[0], (first.seq_id, 64))

        for layer in layers:
            layer.conv_state[0].fill_(3)
            layer.recurrent_state[0].fill_(4)
        cache.update(layers, [(first, 80)])
        for layer in layers:
            layer.conv_state[0].fill_(9)
            layer.recurrent_state[0].fill_(9)

        captured = cache.take([first])[first.seq_id]
        self.assertEqual(captured.num_tokens, 80)
        for conv, recurrent in captured.layers:
            self.assertTrue(torch.all(conv == 3))
            self.assertTrue(torch.all(recurrent == 4))
            self.assertEqual(conv.device.type, "cpu")
            self.assertEqual(recurrent.device.type, "cpu")
        self.assertEqual(cache.take([first]), {})

        # Reused state slots must never surface the previous request's snapshot.
        second = Sequence([2] * 97)
        second.state_slot = 0
        self.assertEqual(cache.take([second]), {})
        for layer in layers:
            layer.conv_state[0].fill_(5)
            layer.recurrent_state[0].fill_(6)
        cache.update(layers, [(second, 96)])
        self.assertEqual(cache.take([second])[second.seq_id].num_tokens, 96)
        cache.clear()
        self.assertEqual(cache._positions, {})


    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_completion_buffer_uses_device_copies_until_final_export(self):
        from nanovllm.engine.sequence import Sequence

        class Layer:
            def __init__(self):
                self.conv_state = torch.ones((1, 3, 4), device="cuda", dtype=torch.bfloat16)
                self.recurrent_state = torch.ones((1, 2, 4), device="cuda", dtype=torch.float32)

            def slot(self, slot):
                return self.conv_state[slot], self.recurrent_state[slot]

        layer = Layer()
        buffer = GDNCompletionBuffer([layer])
        seq = Sequence([1] * 49)
        seq.state_slot = 0
        buffer.update([layer], [(seq, 32)])
        self.assertEqual(buffer._conv[0].device.type, "cuda")
        self.assertEqual(buffer._recurrent[0].device.type, "cuda")
        layer.conv_state.fill_(7)
        layer.recurrent_state.fill_(8)
        buffer.update([layer], [(seq, 48)])
        layer.conv_state.fill_(9)
        layer.recurrent_state.fill_(9)
        snapshot = buffer.take([seq])[seq.seq_id]
        self.assertEqual(snapshot.num_tokens, 48)
        self.assertEqual(snapshot.layers[0][0].device.type, "cpu")
        self.assertEqual(snapshot.layers[0][1].device.type, "cpu")
        self.assertTrue(torch.all(snapshot.layers[0][0] == 7))
        self.assertTrue(torch.all(snapshot.layers[0][1] == 8))
        buffer.clear()



if __name__ == "__main__":
    unittest.main()
