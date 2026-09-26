import warnings

import torch
from torch.profiler import record_function

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


def _resolve_kv_cache_dtype(
    config: Config,
    model_dtype: torch.dtype,
) -> torch.dtype:
    if config.kv_cache_dtype == "auto":
        return model_dtype
    if config.kv_cache_dtype == "fp8_e4m3":
        return torch.float8_e4m3fn
    raise ValueError(
        f"unsupported KV cache dtype: {config.kv_cache_dtype}"
    )


def allocate_runtime_caches(
    model,
    config: Config,
) -> tuple[torch.Tensor, int]:
    """Allocate physical KV pages and per-layer GDN state pools."""
    free, total = torch.cuda.mem_get_info()
    used = total - free
    stats = torch.cuda.memory_stats()
    peak = stats["allocated_bytes.all.peak"]
    current = stats["allocated_bytes.all.current"]

    kv_modules = model.kv_cache_modules()
    state_modules = model.state_cache_modules()
    if not kv_modules:
        raise RuntimeError(
            "Qwen3.5-MoE config contains no full-attention layers"
        )
    if not state_modules:
        raise RuntimeError(
            "Qwen3.5-MoE config contains no GDN layers"
        )

    state_cache_bytes = sum(
        module.state_cache_nbytes(config.max_num_seqs)
        for module in state_modules
    )
    model_param = next(model.parameters())
    cache_dtype = _resolve_kv_cache_dtype(
        config,
        model_param.dtype,
    )
    cache_element_size = (
        1
        if cache_dtype == torch.float8_e4m3fn
        else model_param.element_size()
    )
    block_bytes = (
        2
        * len(kv_modules)
        * config.kvcache_block_size
        * config.text_config.num_key_value_heads
        * config.text_config.head_dim
        * cache_element_size
    )
    cache_budget = int(
        total * config.gpu_memory_utilization
        - used
        - peak
        + current
        - state_cache_bytes
    )
    num_kvcache_blocks = cache_budget // block_bytes
    if num_kvcache_blocks <= 0:
        raise RuntimeError(
            "insufficient GPU memory after reserving "
            "Qwen3.5-MoE recurrent state pools"
        )

    kv_cache = torch.empty(
        2,
        len(kv_modules),
        num_kvcache_blocks,
        config.kvcache_block_size,
        config.text_config.num_key_value_heads,
        config.text_config.head_dim,
        device="cuda",
        dtype=cache_dtype,
    )
    for layer_id, module in enumerate(kv_modules):
        module.k_cache = kv_cache[0, layer_id]
        module.v_cache = kv_cache[1, layer_id]
        module.k_scale = config.kv_cache_k_scale
        module.v_scale = config.kv_cache_v_scale

    if (
        cache_dtype == torch.float8_e4m3fn
        and config.kv_cache_k_scale == 1.0
        and config.kv_cache_v_scale == 1.0
    ):
        warnings.warn(
            "FP8 KV cache is using uncalibrated K/V scales of 1.0; "
            "validate logits/generation accuracy for the target model.",
            RuntimeWarning,
            stacklevel=2,
        )

    for module in state_modules:
        module.allocate_state_cache(config.max_num_seqs)

    return kv_cache, num_kvcache_blocks


def capture_gdn_state(
    model,
    seq: Sequence,
    prefix_tokens: int,
) -> GDNStateSnapshot:
    if seq.state_slot < 0:
        raise RuntimeError(
            f"sequence {seq.seq_id} has no state slot to snapshot"
        )
    physical_prefix = (
        seq.committed_tokens + seq.num_scheduled_tokens
    )
    if prefix_tokens != physical_prefix:
        raise RuntimeError(
            f"sequence {seq.seq_id} snapshot boundary mismatch: "
            f"requested={prefix_tokens}, physical={physical_prefix}"
        )

    with record_function("nanovllm::gdn_snapshot_d2h"):
        layers = tuple(
            module.snapshot_state_slot(seq.state_slot)
            for module in model.state_cache_modules()
        )

    return GDNStateSnapshot(
        num_tokens=prefix_tokens,
        layers=layers,
    )


def restore_gdn_state(
    model,
    seq: Sequence,
    snapshot: GDNStateSnapshot,
) -> None:
    if seq.state_slot < 0:
        raise RuntimeError(
            f"sequence {seq.seq_id} has no state slot to restore"
        )
    if snapshot.num_tokens != seq.committed_tokens:
        raise RuntimeError(
            f"sequence {seq.seq_id} restore boundary mismatch: "
            f"snapshot={snapshot.num_tokens}, "
            f"logical={seq.committed_tokens}"
        )

    modules = model.state_cache_modules()
    if len(snapshot.layers) != len(modules):
        raise RuntimeError(
            "GDN snapshot layer count does not match model"
        )
    for module, layer_snapshot in zip(modules, snapshot.layers):
        module.restore_state_slot(
            seq.state_slot,
            layer_snapshot,
        )


def restore_pending_states(model, seqs: list[Sequence]) -> None:
    for seq in seqs:
        snapshot = seq.pending_state_snapshot
        if snapshot is None:
            continue
        restore_gdn_state(model, seq, snapshot)
        seq.pending_state_snapshot = None
