import unittest

import flashinfer
import torch

from nanovllm.engine.batch_common import (
    DecodeBatchLayout,
    DecodePageTableCache,
    PinnedMetadataPool,
    prepare_decode,
    prepare_prefill,
)
from nanovllm.engine.decode_graph import DecodeGraphBucket
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention, store_kv_cache
from nanovllm.utils.context import Context, use_context


def _has_cuda_gpu() -> bool:
    return torch.cuda.is_available()


@unittest.skipUnless(
    _has_cuda_gpu() and torch.cuda.is_bf16_supported(),
    "FlashInfer integration tests require CUDA BF16",
)
class FlashInferPagedAttentionIntegrationTest(unittest.TestCase):
    PAGE_SIZE = 16
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 2
    HEAD_DIM = 256
    SCALE = HEAD_DIM**-0.5
    DTYPE = torch.bfloat16

    def setUp(self):
        torch.manual_seed(0)
        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device="cuda",
        )

    def test_eager_plan_shares_metadata_with_context(self):
        seq = Sequence([1, 2, 3, 4])
        seq.block_table = [0]
        seq.state_slot = 0
        runner = object.__new__(ModelRunner)
        runner.block_size = self.PAGE_SIZE
        runner.attention_spec = (
            self.NUM_Q_HEADS,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            {
                "q_data_type": self.DTYPE,
                "kv_data_type": self.DTYPE,
                "o_data_type": self.DTYPE,
                "pos_encoding_mode": "NONE",
                "sm_scale": self.SCALE,
            },
        )
        runner.metadata_pool = PinnedMetadataPool()
        runner.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="auto"
        )
        runner.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="auto"
        )
        try:
            prefill = prepare_prefill((ScheduledChunk(seq, 0, 4),), 16, runner.metadata_pool)
            prefill_ctx = runner._plan_attention(prefill)
            self.assertIs(prefill_ctx.paged_kv_indptr, runner.prefill_wrapper._paged_kv_indptr_buf)
            self.assertIs(
                prefill_ctx.paged_kv_last_page_len,
                runner.prefill_wrapper._paged_kv_last_page_len_buf,
            )
            self.assertIs(prefill_ctx.qo_indptr, runner.prefill_wrapper._qo_indptr_buf)
            self.assertEqual(prefill_ctx.qo_indptr.tolist(), [0, 4])
            self.assertEqual(prefill_ctx.paged_kv_indptr.tolist(), [0, 1])
            self.assertEqual(prefill_ctx.paged_kv_last_page_len.tolist(), [4])

            seq.append_token(5)
            decode = prepare_decode(
                (ScheduledChunk(seq, 4, 5),),
                16,
                runner.metadata_pool,
                DecodePageTableCache(),
            )
            decode_ctx = runner._plan_attention(decode)
            self.assertIs(decode_ctx.paged_kv_indptr, runner.decode_wrapper._paged_kv_indptr_buf)
            self.assertIs(
                decode_ctx.paged_kv_last_page_len,
                runner.decode_wrapper._paged_kv_last_page_len_buf,
            )
            self.assertEqual(decode_ctx.paged_kv_indptr.tolist(), [0, 1])
            self.assertEqual(decode_ctx.paged_kv_last_page_len.tolist(), [5])
        finally:
            runner.metadata_pool.clear()

    def _new_attention(
        self,
        cache_dtype: torch.dtype,
        num_pages: int = 1,
    ):
        attention = Attention()
        attention.k_cache = torch.empty(
            num_pages,
            self.PAGE_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=cache_dtype,
            device="cuda",
        )
        attention.v_cache = torch.empty_like(attention.k_cache)
        return attention

    def _prefill_wrapper(
        self,
        *,
        q_len: int,
        kv_len: int,
        kv_dtype: torch.dtype,
    ):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            backend="auto",
        )
        qo_indptr = torch.tensor(
            [0, q_len],
            dtype=torch.int32,
            device="cuda",
        )
        kv_indptr = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device="cuda",
        )
        kv_indices = torch.tensor(
            [0],
            dtype=torch.int32,
            device="cuda",
        )
        last_page_len = torch.tensor(
            [kv_len],
            dtype=torch.int32,
            device="cuda",
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            self.NUM_Q_HEADS,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            self.PAGE_SIZE,
            causal=True,
            q_data_type=self.DTYPE,
            kv_data_type=kv_dtype,
            o_data_type=self.DTYPE,
            pos_encoding_mode="NONE",
            sm_scale=self.SCALE,
        )
        return wrapper

    def _decode_wrapper(
        self,
        *,
        kv_len: int,
        kv_dtype: torch.dtype,
    ):
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            backend="auto",
        )
        kv_indptr = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device="cuda",
        )
        kv_indices = torch.tensor(
            [0],
            dtype=torch.int32,
            device="cuda",
        )
        last_page_len = torch.tensor(
            [kv_len],
            dtype=torch.int32,
            device="cuda",
        )
        wrapper.plan(
            kv_indptr,
            kv_indices,
            last_page_len,
            self.NUM_Q_HEADS,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            self.PAGE_SIZE,
            q_data_type=self.DTYPE,
            kv_data_type=kv_dtype,
            o_data_type=self.DTYPE,
            pos_encoding_mode="NONE",
            sm_scale=self.SCALE,
        )
        return wrapper

    def _expand_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        repeats = self.NUM_Q_HEADS // self.NUM_KV_HEADS
        return tensor.repeat_interleave(repeats, dim=1)

    def _attention_reference(self, query, key, value, prefix_len=0):
        key = self._expand_kv(key).float()
        value = self._expand_kv(value).float()
        scores = torch.einsum("qhd,khd->hqk", query.float(), key) * self.SCALE
        q_positions = torch.arange(query.shape[0], device=query.device) + prefix_len
        k_positions = torch.arange(key.shape[0], device=query.device)
        scores.masked_fill_(k_positions[None, None, :] > q_positions[None, :, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("hqk,khd->qhd", weights, value).to(query.dtype)

    def test_prefill_and_decode_match_reference(self):
        attention = self._new_attention(self.DTYPE)

        q = torch.randn(
            4,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )
        k = torch.randn(
            4,
            self.NUM_KV_HEADS + 1,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )[:, : self.NUM_KV_HEADS]
        v = torch.randn(
            4,
            self.NUM_KV_HEADS + 1,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )[:, : self.NUM_KV_HEADS]
        self.assertFalse(k.is_contiguous())
        self.assertFalse(v.is_contiguous())

        prefill_wrapper = self._prefill_wrapper(
            q_len=4,
            kv_len=4,
            kv_dtype=self.DTYPE,
        )
        with use_context(
            Context(
                is_prefill=True,
                kv_batch_indices=torch.zeros(4, dtype=torch.int32, device="cuda"),
                kv_positions=torch.arange(4, dtype=torch.int32, device="cuda"),
                paged_kv_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
                paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                paged_kv_last_page_len=torch.tensor([4], dtype=torch.int32, device="cuda"),
                attention_wrapper=prefill_wrapper,
            )
        ):
            prefill_actual = attention(q, k, v)

        prefill_expected = self._attention_reference(q, k, v)
        torch.testing.assert_close(
            prefill_actual,
            prefill_expected,
            atol=5e-2,
            rtol=5e-2,
        )

        q_decode = torch.randn(
            1,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )
        k_decode = torch.randn(
            1,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )
        v_decode = torch.randn_like(k_decode)

        decode_wrapper = self._decode_wrapper(
            kv_len=5,
            kv_dtype=self.DTYPE,
        )
        with use_context(
            Context(
                is_prefill=False,
                kv_batch_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
                kv_positions=torch.tensor([4], dtype=torch.int32, device="cuda"),
                paged_kv_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
                paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                paged_kv_last_page_len=torch.tensor([5], dtype=torch.int32, device="cuda"),
                attention_wrapper=decode_wrapper,
            )
        ):
            decode_actual = attention(
                q_decode,
                k_decode,
                v_decode,
            )

        decode_expected = self._attention_reference(
            q_decode,
            torch.cat((k, k_decode), dim=0),
            torch.cat((v, v_decode), dim=0),
            prefix_len=4,
        )
        torch.testing.assert_close(
            decode_actual,
            decode_expected,
            atol=5e-2,
            rtol=5e-2,
        )

    def test_chunked_multi_request_prefill_matches_reference(self):
        attention = self._new_attention(
            self.DTYPE,
            num_pages=2,
        )

        prefix_k = torch.randn(
            2,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )
        prefix_v = torch.randn_like(prefix_k)
        attention.k_cache[0, :2].copy_(prefix_k)
        attention.v_cache[0, :2].copy_(prefix_v)

        q_a = torch.randn(
            3,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )
        k_a = torch.randn(
            3,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )
        v_a = torch.randn_like(k_a)
        q_b = torch.randn(
            2,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )
        k_b = torch.randn(
            2,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=self.DTYPE,
            device="cuda",
        )
        v_b = torch.randn_like(k_b)

        q = torch.cat((q_a, q_b), dim=0)
        k = torch.cat((k_a, k_b), dim=0)
        v = torch.cat((v_a, v_b), dim=0)

        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            backend="auto",
        )
        qo_indptr = torch.tensor(
            [0, 3, 5],
            dtype=torch.int32,
            device="cuda",
        )
        kv_indptr = torch.tensor(
            [0, 1, 2],
            dtype=torch.int32,
            device="cuda",
        )
        kv_indices = torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device="cuda",
        )
        last_page_len = torch.tensor(
            [5, 2],
            dtype=torch.int32,
            device="cuda",
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            self.NUM_Q_HEADS,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            self.PAGE_SIZE,
            causal=True,
            q_data_type=self.DTYPE,
            kv_data_type=self.DTYPE,
            o_data_type=self.DTYPE,
            pos_encoding_mode="NONE",
            sm_scale=self.SCALE,
        )

        with use_context(
            Context(
                is_prefill=True,
                kv_batch_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device="cuda"),
                kv_positions=torch.tensor([2, 3, 4, 0, 1], dtype=torch.int32, device="cuda"),
                paged_kv_indices=kv_indices,
                paged_kv_indptr=kv_indptr,
                paged_kv_last_page_len=last_page_len,
                attention_wrapper=wrapper,
            )
        ):
            actual = attention(q, k, v)

        expected_a = self._attention_reference(
            q_a,
            torch.cat((prefix_k, k_a), dim=0),
            torch.cat((prefix_v, v_a), dim=0),
            prefix_len=2,
        )
        expected_b = self._attention_reference(
            q_b,
            k_b,
            v_b,
            prefix_len=0,
        )
        expected = torch.cat((expected_a, expected_b), dim=0)
        torch.testing.assert_close(
            actual,
            expected,
            atol=5e-2,
            rtol=5e-2,
        )

    @unittest.skipUnless(
        _has_cuda_gpu() and torch.cuda.get_device_capability() >= (9, 0),
        "FP8 FlashInfer coverage requires SM90+",
    )
    def test_fp8_prefill_applies_explicit_kv_scales(self):
        attention = self._new_attention(torch.float8_e4m3fn)
        attention.k_scale = 0.25
        attention.v_scale = 0.25

        q = (
            torch.randn(
                4,
                self.NUM_Q_HEADS,
                self.HEAD_DIM,
                dtype=self.DTYPE,
                device="cuda",
            )
            * 0.25
        )
        k = (
            torch.randn(
                4,
                self.NUM_KV_HEADS,
                self.HEAD_DIM,
                dtype=self.DTYPE,
                device="cuda",
            )
            * 0.25
        )
        v = torch.randn_like(k) * 0.25

        wrapper = self._prefill_wrapper(
            q_len=4,
            kv_len=4,
            kv_dtype=torch.float8_e4m3fn,
        )
        with use_context(
            Context(
                is_prefill=True,
                kv_batch_indices=torch.zeros(4, dtype=torch.int32, device="cuda"),
                kv_positions=torch.arange(4, dtype=torch.int32, device="cuda"),
                paged_kv_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
                paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                paged_kv_last_page_len=torch.tensor([4], dtype=torch.int32, device="cuda"),
                attention_wrapper=wrapper,
            )
        ):
            actual = attention(q, k, v)

        expected = self._attention_reference(q, k, v)
        torch.testing.assert_close(
            actual,
            expected,
            atol=1.5e-1,
            rtol=1.5e-1,
        )


@unittest.skipUnless(_has_cuda_gpu(), "FlashInfer KV append Graph test requires CUDA")
class FlashInferKVAppendGraphTest(unittest.TestCase):
    def test_graph_replay_updates_logical_positions_and_noncontiguous_pages(self):
        for cache_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(cache_dtype=cache_dtype):
                self._check_graph_replay(cache_dtype)

    def _check_graph_replay(self, cache_dtype):
        bucket = DecodeGraphBucket(
            2, 64, 16, torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda")
        )
        key = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        k_cache = torch.zeros(4, 16, 1, 128, device="cuda", dtype=cache_dtype)
        v_cache = torch.zeros_like(k_cache)

        def layout(positions, pages, indptr, last_page_len):
            return DecodeBatchLayout(
                input_ids=(1, 2),
                rope_positions=((0, 0),),
                kv_batch_indices=(0, 1),
                kv_positions=positions,
                state_slots=(0, 1),
                paged_kv_indices=pages,
                paged_kv_indptr=indptr,
                paged_kv_last_page_len=last_page_len,
            )

        # This test isolates KV append. DecodeGraphBucket.load() only stages
        # indptr/last-page metadata on CPU; FlashInfer.plan() uploads them in
        # production. Populate the fixed GPU buffers here without planning attention.
        def load_for_append(batch):
            bucket.load(batch)
            bucket.paged_kv_indptr.copy_(
                torch.tensor(batch.paged_kv_indptr, dtype=torch.int32, device="cuda")
            )
            bucket.paged_kv_last_page_len.copy_(
                torch.tensor(batch.paged_kv_last_page_len, dtype=torch.int32, device="cuda")
            )

        initial = layout((0, 5), (2, 1), (0, 1, 2), (1, 6))
        graph = None
        try:
            bucket._seed_capture_inputs()
            self.assertEqual(bucket.kv_batch_indices.tolist(), [0, 1])
            self.assertEqual(bucket.kv_positions.tolist(), [0, 0])
            load_for_append(initial)
            store_kv_cache(key, value, k_cache, v_cache, bucket.context)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                store_kv_cache(key, value, k_cache, v_cache, bucket.context)
            k_cache.zero_()
            v_cache.zero_()
            expected_k, expected_v = torch.zeros_like(k_cache), torch.zeros_like(v_cache)
            position_ptr = bucket.kv_positions.data_ptr()
            pages_ptr = bucket.paged_kv_indices.data_ptr()

            for batch in (
                initial,
                layout((16, 31), (3, 0, 2, 1), (0, 2, 4), (1, 16)),
                layout((2, 7), (1, 0), (0, 1, 2), (3, 8)),
            ):
                key.add_(1)
                value.sub_(1)
                load_for_append(batch)
                graph.replay()
                for request, position in enumerate(batch.kv_positions):
                    page = batch.paged_kv_indices[batch.paged_kv_indptr[request] + position // 16]
                    expected_k[page, position % 16].copy_(key[request])
                    expected_v[page, position % 16].copy_(value[request])
                torch.testing.assert_close(k_cache.float(), expected_k.float())
                torch.testing.assert_close(v_cache.float(), expected_v.float())
                self.assertEqual(bucket.kv_positions.data_ptr(), position_ptr)
                self.assertEqual(bucket.paged_kv_indices.data_ptr(), pages_ptr)
        finally:
            if graph is not None:
                graph.reset()
            bucket.release()


if __name__ == "__main__":
    unittest.main()
