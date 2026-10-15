import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.layers.attention import Attention, store_kv_cache
from nanovllm.utils.context import Context, use_context


class FlashInferAttentionDispatchTest(unittest.TestCase):
    def test_kv_store_validates_layout_without_python_assertions(self):
        key = torch.randn(2, 1, 8)
        value = torch.randn_like(key)
        cache = torch.empty(2, 4, 1, 8)
        mapping = torch.tensor([0, 1], dtype=torch.int32)
        invalid_inputs = (
            (key, value.transpose(1, 2), cache, cache, mapping, "last dimension"),
            (torch.randn(2, 2, 8)[:, ::2, :], value, cache, cache, mapping, "head strides"),
            (key, value, cache.permute(0, 3, 2, 1), cache, mapping, "page strides"),
            (key, value, cache, cache, mapping[:1], "mapping length"),
        )
        with patch("nanovllm.layers.attention.store_kv_cache_kernel") as kernel:
            for k, v, kc, vc, slots, message in invalid_inputs:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        store_kv_cache(k, v, kc, vc, slots)
        kernel.assert_not_called()

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

        store_kv_cache(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            k_scale=2.0,
            v_scale=3.0,
        )

        torch.testing.assert_close(
            k_cache[0].to(torch.bfloat16) * 2.0,
            key,
            atol=0.125,
            rtol=0.125,
        )
        torch.testing.assert_close(
            v_cache[0].to(torch.bfloat16) * 3.0,
            value,
            atol=0.125,
            rtol=0.125,
        )

    def _attention_with_cache(self, dtype=torch.float32):
        attention = Attention(
            num_heads=2,
            head_dim=8,
            num_kv_heads=1,
        )
        attention.k_cache = torch.empty(4, 4, 1, 8, dtype=dtype)
        attention.v_cache = torch.empty_like(attention.k_cache)
        return attention

    @patch("nanovllm.layers.attention.store_kv_cache")
    def test_paged_phases_use_planned_flashinfer_wrapper(
        self,
        store_kv_cache,
    ):
        for is_prefill, slots in ((True, [0, 1]), (False, [0, 4])):
            with self.subTest(is_prefill=is_prefill):
                store_kv_cache.reset_mock()
                attention = self._attention_with_cache()
                wrapper = Mock()
                q = torch.randn(2, 2, 8)
                k = torch.randn(2, 1, 8)
                v = torch.randn(2, 1, 8)
                wrapper.run.return_value = torch.zeros_like(q)
                context = Context(
                    is_prefill=is_prefill,
                    slot_mapping=torch.tensor(slots, dtype=torch.int32),
                    attention_wrapper=wrapper,
                )

                with use_context(context):
                    actual = attention(q, k, v)

                self.assertIs(actual, wrapper.run.return_value)
                store_kv_cache.assert_called_once()
                args = wrapper.run.call_args.args
                wrapper.run.assert_called_once()
                self.assertIs(args[0], q)
                self.assertIs(args[1][0], attention.k_cache)
                self.assertIs(args[1][1], attention.v_cache)
                self.assertEqual(wrapper.run.call_args.kwargs, {})

    @patch("nanovllm.layers.attention.store_kv_cache")
    def test_fp8_cache_passes_dequant_scales_to_flashinfer(
        self,
        store_kv_cache,
    ):
        attention = self._attention_with_cache(torch.float8_e4m3fn)
        attention.k_scale = 0.5
        attention.v_scale = 0.25
        wrapper = Mock()
        q = torch.randn(1, 2, 8)
        k = torch.randn(1, 1, 8)
        v = torch.randn(1, 1, 8)
        wrapper.run.return_value = torch.zeros_like(q)
        context = Context(
            is_prefill=False,
            slot_mapping=torch.tensor([0], dtype=torch.int32),
            attention_wrapper=wrapper,
        )

        with use_context(context):
            attention(q, k, v)

        wrapper.run.assert_called_once_with(
            q,
            (attention.k_cache, attention.v_cache),
            k_scale=0.5,
            v_scale=0.25,
        )

    def test_missing_paged_cache_is_rejected(self):
        attention = Attention(2, 8, 1)
        q = torch.randn(1, 2, 8)
        k = torch.randn(1, 1, 8)
        v = torch.randn(1, 1, 8)
        with (
            use_context(Context(is_prefill=True)),
            self.assertRaisesRegex(RuntimeError, "requires allocated KV cache"),
        ):
            attention(q, k, v)


if __name__ == "__main__":
    unittest.main()
