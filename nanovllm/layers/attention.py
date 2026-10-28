from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.profiler import record_function

from nanovllm.utils.context import Context, get_context

if TYPE_CHECKING:
    from nanovllm.engine.batch_common import PinnedMetadataPool, PreparedBatch


def flashinfer_attention_spec(
    config,
    model_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> tuple[int, int, int, dict]:
    head_dim = config.head_dim
    return (
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
        {
            "q_data_type": model_dtype,
            "kv_data_type": kv_dtype,
            "o_data_type": model_dtype,
            "pos_encoding_mode": "NONE",
            "sm_scale": head_dim**-0.5,
        },
    )


def plan_flashinfer_attention(
    batch: PreparedBatch,
    metadata_pool: PinnedMetadataPool,
    prefill_wrapper,
    decode_wrapper,
    attention_spec: tuple,
    block_size: int,
) -> Context:
    context = batch.context
    layout = batch.layout
    suffix = "prefill" if context.is_prefill else "decode"
    kv_indptr = metadata_pool.host_view(f"{suffix}_kv_indptr", len(layout.paged_kv_indptr))
    last_page_len = metadata_pool.host_view(
        f"{suffix}_page_len", len(layout.paged_kv_last_page_len)
    )
    num_qo_heads, num_kv_heads, head_dim, common = attention_spec
    if context.is_prefill:
        prefill_wrapper.plan(
            metadata_pool.host_view("prefill_q_offsets", len(layout.q_offsets)),
            kv_indptr,
            context.paged_kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            block_size,
            causal=True,
            **common,
        )
        wrapper = prefill_wrapper
    else:
        decode_wrapper.plan(
            kv_indptr,
            context.paged_kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            block_size,
            **common,
        )
        wrapper = decode_wrapper
    metadata_pool.record_plan_transfer()
    # Eager FlashInfer 0.7.0 owns the device copies of this metadata.
    # Share those buffers with KV append and GDN instead of uploading a
    # duplicate Context copy before plan(). Graph buckets are separate.
    return replace(
        context,
        attention_wrapper=wrapper,
        qo_indptr=wrapper._qo_indptr_buf if context.is_prefill else None,
        paged_kv_indptr=wrapper._paged_kv_indptr_buf,
        paged_kv_last_page_len=wrapper._paged_kv_last_page_len_buf,
    )


def store_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context: Context,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    from flashinfer.page import append_paged_kv_cache

    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if is_fp8:
        # FlashInfer append copies one dtype; quantize before the copy and keep
        # the original dequantization scales for the Attention wrapper.
        key = (key.float() / k_scale).clamp_(-448.0, 448.0).to(k_cache.dtype)
        value = (value.float() / v_scale).clamp_(-448.0, 448.0).to(v_cache.dtype)
    append_paged_kv_cache(
        key,
        value,
        context.kv_batch_indices,
        context.kv_positions,
        (k_cache, v_cache),
        context.paged_kv_indices,
        context.paged_kv_indptr,
        context.paged_kv_last_page_len,
        kv_layout="NHD",
    )


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = 1.0
        self.v_scale = 1.0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        wrapper = context.attention_wrapper
        with record_function("nanovllm::kv_cache_store"):
            store_kv_cache(
                k,
                v,
                k_cache,
                v_cache,
                context,
                self.k_scale,
                self.v_scale,
            )

        run_kwargs = {}
        if k_cache.dtype == torch.float8_e4m3fn:
            run_kwargs["k_scale"] = self.k_scale
            run_kwargs["v_scale"] = self.v_scale

        profile_range = (
            "nanovllm::full_attention_prefill"
            if context.is_prefill
            else "nanovllm::full_attention_decode"
        )
        with record_function(profile_range):
            return wrapper.run(
                q,
                (k_cache, v_cache),
                **run_kwargs,
            )
