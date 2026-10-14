import logging
import warnings

import torch
from torch.profiler import record_function

from nanovllm.config import Config
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot

logger = logging.getLogger(__name__)


def _resolve_kv_cache_dtype(
    config: Config,
    model_dtype: torch.dtype,
) -> torch.dtype:
    if config.kv_cache_dtype == "auto":
        return model_dtype
    if config.kv_cache_dtype == "fp8_e4m3":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported KV cache dtype: {config.kv_cache_dtype}")


def _visual_feature_cache_nbytes(
    config: Config,
    model_dtype: torch.dtype,
) -> int:
    vision_config = getattr(config.root_config, "vision_config", None)
    if vision_config is None:
        return 0
    # Images can remain resident across chunked prefill and multiple scheduler
    # steps. Reserve for concurrent live visual features rather than the
    # current token budget; decreasing this bound without an admission cap
    # could overcommit memory when allocating persistent KV blocks.
    # A configured cap is enforced when an image request is admitted.
    # Without it, retain the existing conservative worst-case reservation.
    max_visual_tokens = (
        config.max_model_len
        if config.max_visual_tokens_per_request is None
        else config.max_visual_tokens_per_request
    )
    return (
        config.max_num_seqs
        * max_visual_tokens
        * config.text_config.hidden_size
        * torch.empty((), dtype=model_dtype).element_size()
    )


def allocate_runtime_caches(
    model,
    config: Config,
) -> tuple[torch.Tensor, int]:
    """Allocate physical KV pages and per-layer GDN state pools."""
    free, total = torch.cuda.mem_get_info()
    used = total - free
    kv_modules = model.kv_cache_modules()
    state_modules = model.state_cache_modules()
    if not kv_modules:
        raise RuntimeError("Qwen3.5 config contains no full-attention layers")
    if not state_modules:
        raise RuntimeError("Qwen3.5 config contains no GDN layers")

    state_cache_bytes = sum(
        module.state_cache_nbytes(config.max_num_seqs) for module in state_modules
    )
    model_param = next(model.parameters())
    cache_dtype = _resolve_kv_cache_dtype(
        config,
        model_param.dtype,
    )
    cache_element_size = 1 if cache_dtype == torch.float8_e4m3fn else model_param.element_size()
    block_bytes = (
        2
        * len(kv_modules)
        * config.kv_cache_block_size
        * config.text_config.num_key_value_heads
        * config.text_config.head_dim
        * cache_element_size
    )
    visual_feature_cache_bytes = _visual_feature_cache_nbytes(
        config,
        model_param.dtype,
    )
    cache_budget = int(
        total * config.gpu_memory_utilization
        - used
        - state_cache_bytes
        - visual_feature_cache_bytes
    )
    num_kv_cache_blocks = cache_budget // block_bytes
    logger.info(
        "Static KV capacity: KV_budget=%d MiB, KV_blocks=%d, "
        "GDN_reserve=%d MiB, vision_reserve=%d MiB",
        cache_budget // (1024**2),
        num_kv_cache_blocks,
        state_cache_bytes // (1024**2),
        visual_feature_cache_bytes // (1024**2),
    )
    if num_kv_cache_blocks <= 0:
        raise RuntimeError("insufficient GPU memory after reserving Qwen3.5 recurrent state pools")

    kv_cache = torch.empty(
        2,
        len(kv_modules),
        num_kv_cache_blocks,
        config.kv_cache_block_size,
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

    return kv_cache, num_kv_cache_blocks


def capture_gdn_state(
    model,
    chunk: ScheduledChunk,
) -> GDNStateSnapshot:
    seq = chunk.seq
    if seq.state_slot < 0:
        raise RuntimeError(f"sequence {seq.seq_id} has no state slot to snapshot")
    if chunk.start != seq.committed_tokens or chunk.end > len(seq):
        raise RuntimeError(f"sequence {seq.seq_id} snapshot boundary mismatch")

    with record_function("nanovllm::gdn_snapshot_d2h"):
        layers = tuple(
            module.snapshot_state_slot(seq.state_slot) for module in model.state_cache_modules()
        )

    return GDNStateSnapshot(
        num_tokens=chunk.end,
        layers=layers,
    )


def restore_gdn_state(
    model,
    seq: Sequence,
    snapshot: GDNStateSnapshot,
) -> None:
    if seq.state_slot < 0:
        raise RuntimeError(f"sequence {seq.seq_id} has no state slot to restore")
    if snapshot.num_tokens != seq.committed_tokens:
        raise RuntimeError(
            f"sequence {seq.seq_id} restore boundary mismatch: "
            f"snapshot={snapshot.num_tokens}, "
            f"logical={seq.committed_tokens}"
        )

    modules = model.state_cache_modules()
    if len(snapshot.layers) != len(modules):
        raise RuntimeError("GDN snapshot layer count does not match model")
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
