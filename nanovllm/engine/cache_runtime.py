import logging
import warnings

import torch

from nanovllm.config import Config
from nanovllm.engine.schedule import GDN_CHECKPOINT_INTERVAL
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot

logger = logging.getLogger(__name__)

# Separate CUDA Graph buckets have private capture pools. Reserve their memory
# before sizing the static KV cache rather than relying on utilization headroom.
DECODE_GRAPH_RESERVE_BYTES_PER_BUCKET = 256 * 1024**2


def _resolve_kv_cache_dtype(
    config: Config,
    model_dtype: torch.dtype,
) -> torch.dtype:
    return model_dtype if config.kv_cache_dtype == "auto" else torch.float8_e4m3fn


def _visual_feature_cache_nbytes(
    config: Config,
    model_dtype: torch.dtype,
) -> int:
    vision_config = getattr(config.root_config, "vision_config", None)
    if vision_config is None or config.max_visual_tokens_per_request == 0:
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
        * model_dtype.itemsize
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
    # Each request captures at most two selected checkpoints. Those states remain
    # live across layers and are stacked once for D2H. FlashInfer's unselected
    # intermediate recurrent states are temporary to a single GDN layer.
    checkpoint_workspace_bytes = 0
    snapshot_size = sum(module.state_cache_nbytes(1) for module in state_modules)
    if config.max_prefix_cache_entries > 0 and (
        config.max_prefix_cache_snapshot_bytes is None
        or config.max_prefix_cache_snapshot_bytes >= snapshot_size
    ):
        selected_per_seq = min(2, config.max_prefix_cache_entries, config.max_gdn_checkpoints)
        if config.max_prefix_cache_snapshot_bytes is not None and snapshot_size:
            selected_per_seq = min(
                selected_per_seq,
                config.max_prefix_cache_snapshot_bytes // snapshot_size,
            )
        selected_slots = min(
            selected_per_seq * config.max_num_seqs,
            config.max_num_seqs + config.max_num_batched_tokens // GDN_CHECKPOINT_INTERVAL,
        )
        captured_bytes = 2 * selected_slots * snapshot_size
        intermediate_bytes = (config.max_num_batched_tokens // GDN_CHECKPOINT_INTERVAL) * max(
            module.num_v_heads
            * module.head_k_dim
            * module.head_v_dim
            * module.recurrent_state.element_size()
            for module in state_modules
        )
        checkpoint_workspace_bytes = captured_bytes + intermediate_bytes
    model_param = model.model.embed_tokens.weight
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
        * config.head_dim
        * cache_element_size
    )
    visual_feature_cache_bytes = _visual_feature_cache_nbytes(
        config,
        model_param.dtype,
    )
    graph_reserve_bytes = (
        len(config.decode_graph_batch_sizes) * DECODE_GRAPH_RESERVE_BYTES_PER_BUCKET
    )
    cache_budget = int(
        total * config.gpu_memory_utilization
        - used
        - state_cache_bytes
        - checkpoint_workspace_bytes
        - visual_feature_cache_bytes
        - graph_reserve_bytes
        - config.activation_reserve_bytes
    )
    num_kv_cache_blocks = cache_budget // block_bytes
    logger.info(
        "Static KV capacity: KV_budget=%d MiB, KV_blocks=%d, "
        "GDN_reserve=%d MiB, checkpoint_reserve=%d MiB, "
        "vision_reserve=%d MiB, graph_reserve=%d MiB, "
        "activation_reserve=%d MiB",
        cache_budget // (1024**2),
        num_kv_cache_blocks,
        state_cache_bytes // (1024**2),
        checkpoint_workspace_bytes // (1024**2),
        visual_feature_cache_bytes // (1024**2),
        graph_reserve_bytes // (1024**2),
        config.activation_reserve_bytes // (1024**2),
    )
    if num_kv_cache_blocks <= 0:
        raise RuntimeError(
            "insufficient GPU memory after reserving Qwen3.5 runtime caches and activations"
        )

    kv_cache = torch.empty(
        2,
        len(kv_modules),
        num_kv_cache_blocks,
        config.kv_cache_block_size,
        config.text_config.num_key_value_heads,
        config.head_dim,
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


def _validate_gdn_restore(seq: Sequence, snapshot: GDNStateSnapshot, modules) -> None:
    if seq.state_slot < 0:
        raise RuntimeError(f"sequence {seq.seq_id} has no state slot to restore")
    if snapshot.num_tokens != seq.committed_tokens:
        raise RuntimeError(
            f"sequence {seq.seq_id} restore boundary mismatch: "
            f"snapshot={snapshot.num_tokens}, logical={seq.committed_tokens}"
        )
    if not modules or len(snapshot.layers) != len(modules):
        raise RuntimeError("GDN snapshot layer count does not match model")
    for layer_index, (module, (conv, recurrent)) in enumerate(zip(modules, snapshot.layers)):
        if (
            seq.state_slot >= module.conv_state.shape[0]
            or seq.state_slot >= module.recurrent_state.shape[0]
        ):
            raise RuntimeError(
                f"sequence {seq.seq_id} GDN layer {layer_index} state slot out of range"
            )
        for name, source, target in (
            ("conv", conv, module.conv_state),
            ("recurrent", recurrent, module.recurrent_state),
        ):
            if source.shape != target.shape[1:] or source.dtype != target.dtype:
                raise RuntimeError(
                    f"sequence {seq.seq_id} GDN layer {layer_index} {name} "
                    f"snapshot shape/dtype {tuple(source.shape)}/{source.dtype} "
                    f"does not match state cache {tuple(target.shape[1:])}/{target.dtype}"
                )


class GDNRestoreStaging:
    """Bounded pinned staging for batched, two-transfer GDN checkpoint restores.

    CPU snapshots retain independent ownership; this temporary staging is never
    exposed to PrefixCache. A CUDA event guards reuse across batches.
    """

    def __init__(self, max_staging_bytes: int = 64 * 1024**2) -> None:
        if max_staging_bytes <= 0:
            raise ValueError("max_staging_bytes must be positive")
        self.max_staging_bytes = max_staging_bytes
        self._conv_host: torch.Tensor | None = None
        self._recurrent_host: torch.Tensor | None = None
        self._transfer_done: torch.cuda.Event | None = None
        self._pending = False

    def _wait(self) -> None:
        if self._pending:
            self._transfer_done.synchronize()
            self._pending = False

    def clear(self) -> None:
        self._wait()
        self._conv_host = None
        self._recurrent_host = None
        self._transfer_done = None

    def restore(self, model, pending: list[tuple[Sequence, GDNStateSnapshot]]) -> None:
        if not pending:
            return
        modules = model.state_cache_modules()
        for seq, snapshot in pending:
            _validate_gdn_restore(seq, snapshot, modules)
        first = pending[0][1].layers[0]
        device = modules[0].conv_state.device
        conv_shape, recurrent_shape = first[0].shape, first[1].shape
        conv_dtype, recurrent_dtype = first[0].dtype, first[1].dtype
        homogeneous = all(
            conv.shape == conv_shape
            and recurrent.shape == recurrent_shape
            and conv.dtype == conv_dtype
            and recurrent.dtype == recurrent_dtype
            and conv.device.type == "cpu"
            and recurrent.device.type == "cpu"
            for _, snapshot in pending
            for conv, recurrent in snapshot.layers
        ) and all(
            module.conv_state.device == device and module.recurrent_state.device == device
            for module in modules
        )
        per_snapshot_bytes = len(modules) * (
            first[0].numel() * first[0].element_size() + first[1].numel() * first[1].element_size()
        )
        if (
            not homogeneous
            or device.type != "cuda"
            or per_snapshot_bytes == 0
            or per_snapshot_bytes > self.max_staging_bytes
        ):
            # Preserve support for heterogeneous/custom layer layouts and CPU tests.
            for seq, snapshot in pending:
                for module, layer_snapshot in zip(modules, snapshot.layers):
                    module.restore_state_slot(seq.state_slot, layer_snapshot)
            return

        max_group_size = max(1, self.max_staging_bytes // per_snapshot_bytes)
        num_layers = len(modules)
        for start in range(0, len(pending), max_group_size):
            group = pending[start : start + max_group_size]
            count = len(group) * num_layers
            self._wait()
            if (
                self._conv_host is None
                or self._conv_host.shape[0] < count
                or self._conv_host.shape[1:] != conv_shape
                or self._conv_host.dtype != conv_dtype
            ):
                self._conv_host = torch.empty(
                    (count, *conv_shape), dtype=conv_dtype, pin_memory=True
                )
            if (
                self._recurrent_host is None
                or self._recurrent_host.shape[0] < count
                or self._recurrent_host.shape[1:] != recurrent_shape
                or self._recurrent_host.dtype != recurrent_dtype
            ):
                self._recurrent_host = torch.empty(
                    (count, *recurrent_shape), dtype=recurrent_dtype, pin_memory=True
                )
            conv_host = self._conv_host[:count]
            recurrent_host = self._recurrent_host[:count]
            for group_index, (_, snapshot) in enumerate(group):
                for layer_index, (conv, recurrent) in enumerate(snapshot.layers):
                    offset = group_index * num_layers + layer_index
                    conv_host[offset].copy_(conv)
                    recurrent_host[offset].copy_(recurrent)
            # Only two host-to-device transfers per bounded group; scatter from
            # these temporary device tensors on the same stream before forward.
            try:
                conv_gpu = conv_host.to(device, non_blocking=True)
                recurrent_gpu = recurrent_host.to(device, non_blocking=True)
                for group_index, (seq, _) in enumerate(group):
                    for layer_index, module in enumerate(modules):
                        offset = group_index * num_layers + layer_index
                        module.restore_state_slot(
                            seq.state_slot, (conv_gpu[offset], recurrent_gpu[offset])
                        )
            finally:
                # Guard pinned storage even if an H2D transfer or scatter fails
                # after earlier asynchronous copies have been queued.
                if self._transfer_done is None:
                    self._transfer_done = torch.cuda.Event()
                self._transfer_done.record(torch.cuda.current_stream(device))
                self._pending = True


def restore_pending_states(model, seqs: list[Sequence], staging: GDNRestoreStaging) -> None:
    pending = [
        (seq, seq.pending_state_snapshot) for seq in seqs if seq.pending_state_snapshot is not None
    ]
    staging.restore(model, pending)
    # Clear only after every restore was successfully scheduled.
    for seq, _ in pending:
        seq.pending_state_snapshot = None
