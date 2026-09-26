import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


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


def _warmup_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Dense startup-only path used before persistent KV allocation."""
    if q.size(0) != k.size(0) or k.size(0) != v.size(0):
        raise RuntimeError("warmup attention expects one dense sequence")
    if q.size(1) % k.size(1) != 0:
        raise RuntimeError("query heads must be divisible by KV heads")

    if q.size(1) != k.size(1):
        repeats = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    q_t = q.transpose(0, 1).unsqueeze(0)
    k_t = k.transpose(0, 1).unsqueeze(0)
    v_t = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        is_causal=True,
        scale=softmax_scale,
    )
    return out.squeeze(0).transpose(0, 1)


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

        if k_cache.numel() and v_cache.numel():
            if context.slot_mapping is None:
                raise RuntimeError(
                    "paged attention requires KV slot mapping"
                )
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

        wrapper = context.attention_wrapper
        if wrapper is None:
            if k_cache.numel() or v_cache.numel():
                raise RuntimeError(
                    "persistent KV cache requires a planned FlashInfer wrapper"
                )
            if not context.is_prefill:
                raise RuntimeError(
                    "decode requires FlashInfer paged attention"
                )
            return _warmup_attention(q, k, v, self.scale)

        if not k_cache.numel() or not v_cache.numel():
            raise RuntimeError(
                "FlashInfer paged attention requires allocated KV cache"
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
