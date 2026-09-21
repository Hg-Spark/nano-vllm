import torch
from torch import nn
from torch.profiler import record_function
import triton
import triton.language as tl

from flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from nanovllm.utils.context import get_context


FP8_E4M3_MAX = 448.0


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_scale,
    v_scale,
    D: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    offsets = tl.arange(0, D)
    key = tl.load(key_ptr + idx * key_stride + offsets)
    value = tl.load(value_ptr + idx * value_stride + offsets)
    if IS_FP8:
        key = tl.maximum(
            tl.minimum(key / k_scale, 448.0),
            -448.0,
        )
        value = tl.maximum(
            tl.minimum(value / v_scale, 448.0),
            -448.0,
        )

    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if is_fp8 != (v_cache.dtype == torch.float8_e4m3fn):
        raise RuntimeError("K/V cache dtypes must match")
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        k_scale,
        v_scale,
        D,
        IS_FP8=is_fp8,
    )


def _materialize_paged_cache(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seqlen: int,
    scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Gather one request's live pages and dequantize into compute dtype."""
    if seqlen <= 0:
        raise ValueError("paged cache length must be positive")
    block_size = cache.shape[1]
    num_blocks = (seqlen + block_size - 1) // block_size
    block_ids = block_table_row[:num_blocks].to(torch.long)
    dense = cache.index_select(0, block_ids).flatten(0, 1)[:seqlen]
    return dense.to(output_dtype) * scale


def fp8_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    softmax_scale: float,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    """Correctness path for FP8 paged KV.

    Persistent KV remains FP8. Only pages touched by the current request are
    materialized to the query dtype before FlashAttention. This path is kept
    deliberately simple; profiling decides whether a fused GQA/paged decode
    kernel is justified.
    """
    if context.block_tables is None or context.state_prefix_lens is None:
        raise RuntimeError(
            "FP8 paged attention requires block-table metadata"
        )

    outputs = []
    if context.is_prefill:
        if context.prefill_q_offsets is None:
            raise RuntimeError(
                "FP8 prefill requires packed query offsets"
            )
        q_offsets = context.prefill_q_offsets
        for idx, prefix_len in enumerate(
            context.state_prefix_lens
        ):
            q_start = q_offsets[idx]
            q_end = q_offsets[idx + 1]
            q_i = q[q_start:q_end]
            k_len = prefix_len + (q_end - q_start)
            k_i = _materialize_paged_cache(
                k_cache,
                context.block_tables[idx],
                k_len,
                k_scale,
                q.dtype,
            )
            v_i = _materialize_paged_cache(
                v_cache,
                context.block_tables[idx],
                k_len,
                v_scale,
                q.dtype,
            )
            outputs.append(
                flash_attn_func(
                    q_i.unsqueeze(0),
                    k_i.unsqueeze(0),
                    v_i.unsqueeze(0),
                    softmax_scale=softmax_scale,
                    causal=True,
                ).squeeze(0)
            )
        return torch.cat(outputs, dim=0)

    for idx, prefix_len in enumerate(
        context.state_prefix_lens
    ):
        k_len = prefix_len + 1
        k_i = _materialize_paged_cache(
            k_cache,
            context.block_tables[idx],
            k_len,
            k_scale,
            q.dtype,
        )
        v_i = _materialize_paged_cache(
            v_cache,
            context.block_tables[idx],
            k_len,
            v_scale,
            q.dtype,
        )
        outputs.append(
            flash_attn_func(
                q[idx:idx + 1].unsqueeze(1),
                k_i.unsqueeze(0),
                v_i.unsqueeze(0),
                softmax_scale=softmax_scale,
                causal=True,
            ).squeeze(1)
        )
    return torch.cat(outputs, dim=0)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
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
        use_fp8_cache = (
            k_cache.numel()
            and k_cache.dtype == torch.float8_e4m3fn
        )

        if k_cache.numel() and v_cache.numel():
            with record_function("nanovllm::kv_cache_store"):
                store_kvcache(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    context.slot_mapping,
                    self.k_scale,
                    self.v_scale,
                )

        if context.is_prefill:
            if context.block_tables is not None:
                if use_fp8_cache:
                    return fp8_paged_attention_reference(
                        q,
                        k_cache,
                        v_cache,
                        context,
                        self.scale,
                        self.k_scale,
                        self.v_scale,
                    )
                k, v = k_cache, v_cache
            return flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )

        with record_function(
            "nanovllm::full_attention_decode"
        ):
            if use_fp8_cache:
                return fp8_paged_attention_reference(
                    q,
                    k_cache,
                    v_cache,
                    context,
                    self.scale,
                    self.k_scale,
                    self.v_scale,
                )

            return flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            ).squeeze(1)
