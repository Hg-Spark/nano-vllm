import unittest
from unittest.mock import Mock, patch

import torch

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import Context, use_context


class FlashInferAttentionDispatchTest(unittest.TestCase):

    def _attention_with_cache(self, dtype=torch.float32):
        attention = Attention(
            num_heads=2,
            head_dim=8,
            scale=8 ** -0.5,
            num_kv_heads=1,
        )
        attention.k_cache = torch.empty(4, 4, 1, 8, dtype=dtype)
        attention.v_cache = torch.empty_like(attention.k_cache)
        return attention

    @patch("nanovllm.layers.attention.store_kvcache")
    def test_paged_prefill_uses_planned_flashinfer_wrapper(
        self,
        store_kvcache,
    ):
        attention = self._attention_with_cache()
        wrapper = Mock()
        q = torch.randn(2, 2, 8)
        k = torch.randn(2, 1, 8)
        v = torch.randn(2, 1, 8)
        wrapper.run.return_value = torch.zeros_like(q)
        context = Context(
            is_prefill=True,
            slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
            attention_wrapper=wrapper,
        )

        with use_context(context):
            actual = attention(q, k, v)

        self.assertEqual(actual.shape, q.shape)
        store_kvcache.assert_called_once()
        wrapper.run.assert_called_once()
        args = wrapper.run.call_args.args
        self.assertIs(args[0], q)
        self.assertIs(args[1][0], attention.k_cache)
        self.assertIs(args[1][1], attention.v_cache)
        self.assertEqual(wrapper.run.call_args.kwargs, {})

    @patch("nanovllm.layers.attention.store_kvcache")
    def test_decode_uses_same_paged_flashinfer_interface(
        self,
        store_kvcache,
    ):
        attention = self._attention_with_cache()
        wrapper = Mock()
        q = torch.randn(2, 2, 8)
        k = torch.randn(2, 1, 8)
        v = torch.randn(2, 1, 8)
        wrapper.run.return_value = torch.zeros_like(q)
        context = Context(
            is_prefill=False,
            slot_mapping=torch.tensor([0, 4], dtype=torch.int32),
            attention_wrapper=wrapper,
        )

        with use_context(context):
            actual = attention(q, k, v)

        self.assertEqual(actual.shape, q.shape)
        store_kvcache.assert_called_once()
        wrapper.run.assert_called_once_with(
            q,
            (attention.k_cache, attention.v_cache),
        )

    @patch("nanovllm.layers.attention.store_kvcache")
    def test_fp8_cache_passes_dequant_scales_to_flashinfer(
        self,
        store_kvcache,
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

    def test_startup_warmup_uses_dense_sdpa_without_persistent_cache(self):
        attention = Attention(
            num_heads=2,
            head_dim=8,
            scale=8 ** -0.5,
            num_kv_heads=1,
        )
        q = torch.randn(3, 2, 8)
        k = torch.randn(3, 1, 8)
        v = torch.randn(3, 1, 8)

        with use_context(Context(is_prefill=True)):
            actual = attention(q, k, v)

        self.assertEqual(actual.shape, q.shape)


if __name__ == "__main__":
    unittest.main()
