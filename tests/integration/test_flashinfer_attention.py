import unittest

import flashinfer
import torch
import torch.nn.functional as F

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import Context, use_context


def _has_cu130_gpu() -> bool:
    return torch.cuda.is_available() and torch.version.cuda == "13.0"


@unittest.skipUnless(
    _has_cu130_gpu(),
    "FlashInfer integration tests require a CUDA 13.0 GPU",
)
class FlashInferPagedAttentionIntegrationTest(unittest.TestCase):

    PAGE_SIZE = 16
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 2
    HEAD_DIM = 256
    SCALE = HEAD_DIM ** -0.5

    def setUp(self):
        torch.manual_seed(0)
        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device="cuda",
        )

    def _new_attention(
        self,
        cache_dtype: torch.dtype,
        num_pages: int = 1,
    ):
        attention = Attention(
            num_heads=self.NUM_Q_HEADS,
            head_dim=self.HEAD_DIM,
            scale=self.SCALE,
            num_kv_heads=self.NUM_KV_HEADS,
        )
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
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
            o_data_type=torch.bfloat16,
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
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
            o_data_type=torch.bfloat16,
            pos_encoding_mode="NONE",
            sm_scale=self.SCALE,
        )
        return wrapper

    def _expand_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        repeats = self.NUM_Q_HEADS // self.NUM_KV_HEADS
        return tensor.repeat_interleave(repeats, dim=1)

    def _prefill_reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        k = self._expand_kv(k)
        v = self._expand_kv(v)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            scale=self.SCALE,
        )
        return out.squeeze(0).transpose(0, 1)

    def _decode_reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        k = self._expand_kv(k).float()
        v = self._expand_kv(v).float()
        scores = torch.einsum(
            "qhd,khd->hqk",
            q.float(),
            k,
        ) * self.SCALE
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum(
            "hqk,khd->qhd",
            probs,
            v,
        ).to(q.dtype)

    def test_bf16_prefill_and_decode_match_reference(self):
        attention = self._new_attention(torch.bfloat16)

        q = torch.randn(
            4,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn(
            4,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        v = torch.randn_like(k)

        prefill_wrapper = self._prefill_wrapper(
            q_len=4,
            kv_len=4,
            kv_dtype=torch.bfloat16,
        )
        with use_context(
            Context(
                is_prefill=True,
                slot_mapping=torch.arange(
                    4,
                    dtype=torch.int32,
                    device="cuda",
                ),
                attention_wrapper=prefill_wrapper,
            )
        ):
            prefill_actual = attention(q, k, v)

        prefill_expected = self._prefill_reference(q, k, v)
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
            dtype=torch.bfloat16,
            device="cuda",
        )
        k_decode = torch.randn(
            1,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        v_decode = torch.randn_like(k_decode)

        decode_wrapper = self._decode_wrapper(
            kv_len=5,
            kv_dtype=torch.bfloat16,
        )
        with use_context(
            Context(
                is_prefill=False,
                slot_mapping=torch.tensor(
                    [4],
                    dtype=torch.int32,
                    device="cuda",
                ),
                attention_wrapper=decode_wrapper,
            )
        ):
            decode_actual = attention(
                q_decode,
                k_decode,
                v_decode,
            )

        decode_expected = self._decode_reference(
            q_decode,
            torch.cat((k, k_decode), dim=0),
            torch.cat((v, v_decode), dim=0),
        )
        torch.testing.assert_close(
            decode_actual,
            decode_expected,
            atol=5e-2,
            rtol=5e-2,
        )

    def test_chunked_multi_request_prefill_matches_reference(self):
        attention = self._new_attention(
            torch.bfloat16,
            num_pages=2,
        )

        prefix_k = torch.randn(
            2,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        prefix_v = torch.randn_like(prefix_k)
        attention.k_cache[0, :2].copy_(prefix_k)
        attention.v_cache[0, :2].copy_(prefix_v)

        q_a = torch.randn(
            3,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k_a = torch.randn(
            3,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        v_a = torch.randn_like(k_a)
        q_b = torch.randn(
            2,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k_b = torch.randn(
            2,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
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
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            pos_encoding_mode="NONE",
            sm_scale=self.SCALE,
        )

        with use_context(
            Context(
                is_prefill=True,
                slot_mapping=torch.tensor(
                    [2, 3, 4, 16, 17],
                    dtype=torch.int32,
                    device="cuda",
                ),
                attention_wrapper=wrapper,
            )
        ):
            actual = attention(q, k, v)

        def chunk_reference(
            query,
            key,
            value,
            prefix_len,
        ):
            key = self._expand_kv(key).float()
            value = self._expand_kv(value).float()
            scores = torch.einsum(
                "qhd,khd->hqk",
                query.float(),
                key,
            ) * self.SCALE
            q_pos = torch.arange(
                query.size(0),
                device="cuda",
            ) + prefix_len
            k_pos = torch.arange(
                key.size(0),
                device="cuda",
            )
            mask = k_pos[None, :] <= q_pos[:, None]
            scores = scores.masked_fill(
                ~mask.unsqueeze(0),
                float("-inf"),
            )
            probs = torch.softmax(scores, dim=-1)
            return torch.einsum(
                "hqk,khd->qhd",
                probs,
                value,
            ).to(query.dtype)

        expected_a = chunk_reference(
            q_a,
            torch.cat((prefix_k, k_a), dim=0),
            torch.cat((prefix_v, v_a), dim=0),
            prefix_len=2,
        )
        expected_b = chunk_reference(
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
        _has_cu130_gpu()
        and torch.cuda.get_device_capability() >= (12, 0),
        "FP8 FlashInfer coverage targets Blackwell SM120+",
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
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.25
        )
        k = (
            torch.randn(
                4,
                self.NUM_KV_HEADS,
                self.HEAD_DIM,
                dtype=torch.bfloat16,
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
                slot_mapping=torch.arange(
                    4,
                    dtype=torch.int32,
                    device="cuda",
                ),
                attention_wrapper=wrapper,
            )
        ):
            actual = attention(q, k, v)

        expected = self._prefill_reference(q, k, v)
        torch.testing.assert_close(
            actual,
            expected,
            atol=1.5e-1,
            rtol=1.5e-1,
        )


if __name__ == "__main__":
    unittest.main()
