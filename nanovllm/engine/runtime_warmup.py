import math
from collections.abc import Callable

import torch

from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import Context, use_context


def _warmup_prefill(config, run_batch: Callable) -> None:
    token_budget = min(
        config.max_num_batched_tokens,
        config.max_model_len * config.max_num_seqs,
    )
    if token_budget <= 0:
        raise ValueError("warmup token budget must be positive")

    chunks = []
    remaining = token_budget
    while remaining > 0 and len(chunks) < config.max_num_seqs:
        seq_len = min(config.max_model_len, remaining)
        seq = Sequence([0] * seq_len)
        chunks.append(ScheduledChunk(seq, 0, seq_len))
        remaining -= seq_len
    run_batch(tuple(chunks), True)


def _warmup_decode_moe(model, config, decode_graph_buckets) -> None:
    layers = model.model.layers
    if not layers:
        return
    hidden_size = config.text_config.hidden_size
    model_dtype = next(model.parameters()).dtype
    batch_sizes = sorted({
        config.max_num_seqs,
        *decode_graph_buckets.keys(),
    })
    with torch.inference_mode(), use_context(Context(is_prefill=False)):
        for batch_size in batch_sizes:
            hidden_states = torch.zeros(
                batch_size,
                hidden_size,
                device="cuda",
                dtype=model_dtype,
            )
            layers[0].mlp(hidden_states)


def _warmup_vision(model, config) -> None:
    vision_config = getattr(config.root_config, "vision_config", None)
    if model.visual is None or vision_config is None:
        return

    merge = int(vision_config.spatial_merge_size)
    max_visual_tokens = max(1, config.max_model_len)
    merged_h = max(1, math.isqrt(max_visual_tokens))
    merged_w = max(1, max_visual_tokens // merged_h)
    grid_h = merged_h * merge
    grid_w = merged_w * merge

    patch_size = vision_config.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)
    temporal_patch = vision_config.temporal_patch_size
    if isinstance(temporal_patch, (tuple, list)):
        temporal_patch = temporal_patch[0]
    patch_width = (
        int(vision_config.in_channels)
        * patch_h
        * patch_w
        * int(temporal_patch)
    )
    pixel_values = torch.zeros(
        grid_h * grid_w,
        patch_width,
        device="cuda",
        dtype=next(model.visual.parameters()).dtype,
    )
    grid_thw = torch.tensor(
        [[1, grid_h, grid_w]],
        dtype=torch.long,
        device="cuda",
    )
    with torch.inference_mode():
        model.encode_image(pixel_values, grid_thw)


def warmup_runtime(model, config, decode_graph_buckets, run_batch: Callable) -> None:
    """Exercise peak runtime paths before persistent KV-cache allocation."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _warmup_decode_moe(model, config, decode_graph_buckets)
    _warmup_vision(model, config)
    _warmup_prefill(config, run_batch)
    torch.cuda.empty_cache()
