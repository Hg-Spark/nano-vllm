import torch

from flash_attn import flash_attn_func


def _materialize_paged_cache(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seqlen: int,
    scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Gather one request's live FP8 pages into compute dtype."""
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
    """Correctness-first FP8 paged KV read path.

    Persistent KV stays FP8. Only pages touched by the current request are
    materialized to the query dtype. A fused kernel should replace this only
    when profiling proves decode attention is a material bottleneck.
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
