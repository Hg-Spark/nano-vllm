from dataclasses import dataclass

import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model


@dataclass(frozen=True, slots=True)
class PrefillBatchLayout:
    """CPU control-plane description of one packed prefill batch."""

    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    q_offsets: tuple[int, ...]
    k_offsets: tuple[int, ...]
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: tuple[int, ...]
    state_slots: tuple[int, ...]
    state_prefix_lens: tuple[int, ...]
    use_block_tables: bool


@dataclass(frozen=True, slots=True)
class GDNStateSnapshot:
    """Host checkpoint for all GDN layers at one committed prefix."""

    num_tokens: int
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]


def build_prefill_batch_layout(
    seqs: list[Sequence],
    block_size: int,
) -> PrefillBatchLayout:
    """Pack variable-length prefill chunks while preserving request alignment.

    Index i consistently refers to one request across q_offsets/k_offsets,
    state_slots, state_prefix_lens and the corresponding block-table row.
    """
    if not seqs:
        raise ValueError("prefill batch must not be empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    has_block_tables = [bool(seq.block_table) for seq in seqs]
    if any(has_block_tables) and not all(has_block_tables):
        raise RuntimeError(
            "prefill batch cannot mix allocated and unallocated requests"
        )

    input_ids: list[int] = []
    positions: list[int] = []
    q_offsets = [0]
    k_offsets = [0]
    slot_mapping: list[int] = []
    state_slots: list[int] = []
    state_prefix_lens: list[int] = []
    seen_state_slots: set[int] = set()
    max_seqlen_q = 0
    max_seqlen_k = 0
    use_block_tables = False

    for seq in seqs:
        if seq.num_scheduled_tokens <= 0:
            raise RuntimeError(
                f"sequence {seq.seq_id} has no scheduled prefill tokens"
            )
        if seq.num_cached_tokens != seq.num_state_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} KV/state prefix mismatch: "
                f"kv={seq.num_cached_tokens}, "
                f"state={seq.num_state_tokens}"
            )

        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        if end > seq.num_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} prefill range [{start}, {end}) "
                f"exceeds token count {seq.num_tokens}"
            )
        seqlen_k = end

        if seq.block_table:
            if seq.state_slot < 0:
                raise RuntimeError(
                    f"sequence {seq.seq_id} has KV blocks without a state slot"
                )
            if seq.state_slot in seen_state_slots:
                raise RuntimeError(
                    f"duplicate state slot {seq.state_slot} in prefill batch"
                )
            seen_state_slots.add(seq.state_slot)
            required_blocks = (end + block_size - 1) // block_size
            if len(seq.block_table) < required_blocks:
                raise RuntimeError(
                    f"sequence {seq.seq_id} block table is too short: "
                    f"need {required_blocks}, have {len(seq.block_table)}"
                )
        elif seq.state_slot >= 0:
            raise RuntimeError(
                f"sequence {seq.seq_id} has a state slot without KV blocks"
            )

        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))
        q_offsets.append(q_offsets[-1] + seqlen_q)
        k_offsets.append(k_offsets[-1] + seqlen_k)
        max_seqlen_q = max(max_seqlen_q, seqlen_q)
        max_seqlen_k = max(max_seqlen_k, seqlen_k)
        state_slots.append(seq.state_slot)
        state_prefix_lens.append(seq.num_state_tokens)
        use_block_tables = use_block_tables or start > 0

        if not seq.block_table:
            continue

        start_block = start // block_size
        end_block = (end + block_size - 1) // block_size
        for block_idx in range(start_block, end_block):
            physical_block = seq.block_table[block_idx]
            slot_start = physical_block * block_size
            if block_idx == start_block:
                slot_start += start % block_size

            if block_idx == end_block - 1:
                slot_end = (
                    physical_block * block_size
                    + end
                    - block_idx * block_size
                )
            else:
                slot_end = physical_block * block_size + block_size
            slot_mapping.extend(range(slot_start, slot_end))

    if all(has_block_tables) and len(slot_mapping) != len(input_ids):
        raise RuntimeError(
            "prefill slot mapping must contain one entry per packed token"
        )

    return PrefillBatchLayout(
        input_ids=tuple(input_ids),
        positions=tuple(positions),
        q_offsets=tuple(q_offsets),
        k_offsets=tuple(k_offsets),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=tuple(slot_mapping),
        state_slots=tuple(state_slots),
        state_prefix_lens=tuple(state_prefix_lens),
        use_block_tables=use_block_tables,
    )


def _resolve_dtype(config) -> torch.dtype:
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return dtype or torch.bfloat16


class ModelRunner:

    def __init__(self, config: Config):
        self.config = config
        self.hf_config = config.text_config
        self.block_size = config.kvcache_block_size

        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        dtype = _resolve_dtype(self.hf_config)
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")
        try:
            self.model = Qwen3_5MoeForCausalLM(self.hf_config)
            load_model(self.model, config.model)
            self.sampler = Sampler()
            self.warmup_model()
            self.allocate_cache()
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)

    def exit(self):
        torch.cuda.synchronize()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seq_len = min(
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
            32,
        )
        if seq_len <= 0:
            raise ValueError("warmup sequence length must be positive")
        seq = Sequence([0] * seq_len)
        seq.num_scheduled_tokens = seq_len
        self.run([seq], True)
        torch.cuda.empty_cache()

    def allocate_cache(self):
        config = self.config
        hf_config = self.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        stats = torch.cuda.memory_stats()
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]

        kv_modules = self.model.kv_cache_modules()
        state_modules = self.model.state_cache_modules()
        if not kv_modules:
            raise RuntimeError(
                "Qwen3.5-MoE config contains no full-attention layers"
            )
        if not state_modules:
            raise RuntimeError(
                "Qwen3.5-MoE config contains no GDN layers"
            )

        state_cache_bytes = sum(
            module.state_cache_nbytes(
                config.max_num_state_slots
            )
            for module in state_modules
        )
        model_param = next(self.model.parameters())
        block_bytes = (
            2
            * len(kv_modules)
            * self.block_size
            * hf_config.num_key_value_heads
            * hf_config.head_dim
            * model_param.element_size()
        )
        cache_budget = int(
            total * config.gpu_memory_utilization
            - used
            - peak
            + current
            - state_cache_bytes
        )
        config.num_kvcache_blocks = (
            cache_budget // block_bytes
        )
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(
                "insufficient GPU memory after reserving "
                "Qwen3.5-MoE recurrent state pools"
            )

        self.kv_cache = torch.empty(
            2,
            len(kv_modules),
            config.num_kvcache_blocks,
            self.block_size,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            device="cuda",
            dtype=model_param.dtype,
        )
        for layer_id, module in enumerate(kv_modules):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]

        for module in state_modules:
            module.allocate_state_cache(
                config.max_num_state_slots
            )

    def capture_gdn_state(
        self,
        seq: Sequence,
        prefix_tokens: int,
    ) -> GDNStateSnapshot:
        if seq.state_slot < 0:
            raise RuntimeError(
                f"sequence {seq.seq_id} has no state slot to snapshot"
            )
        physical_prefix = (
            seq.num_state_tokens + seq.num_scheduled_tokens
        )
        if prefix_tokens != physical_prefix:
            raise RuntimeError(
                f"sequence {seq.seq_id} snapshot boundary mismatch: "
                f"requested={prefix_tokens}, physical={physical_prefix}"
            )

        layers = tuple(
            module.snapshot_state_slot(seq.state_slot)
            for module in self.model.state_cache_modules()
        )
        return GDNStateSnapshot(
            num_tokens=prefix_tokens,
            layers=layers,
        )

    def restore_gdn_state(
        self,
        seq: Sequence,
        snapshot: GDNStateSnapshot,
    ) -> None:
        if seq.state_slot < 0:
            raise RuntimeError(
                f"sequence {seq.seq_id} has no state slot to restore"
            )
        if snapshot.num_tokens != seq.num_state_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} restore boundary mismatch: "
                f"snapshot={snapshot.num_tokens}, "
                f"logical={seq.num_state_tokens}"
            )

        modules = self.model.state_cache_modules()
        if len(snapshot.layers) != len(modules):
            raise RuntimeError(
                "GDN snapshot layer count does not match model"
            )
        for module, layer_snapshot in zip(
            modules,
            snapshot.layers,
        ):
            module.restore_state_slot(
                seq.state_slot,
                layer_snapshot,
            )

    def _restore_pending_states(
        self,
        seqs: list[Sequence],
    ) -> None:
        for seq in seqs:
            snapshot = seq.pending_state_snapshot
            if snapshot is None:
                continue
            if not isinstance(snapshot, GDNStateSnapshot):
                raise RuntimeError(
                    "invalid pending GDN state snapshot"
                )
            self.restore_gdn_state(seq, snapshot)
            seq.pending_state_snapshot = None

    def prepare_block_tables(
        self,
        seqs: list[Sequence],
    ) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table
            + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(
            block_tables,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)

    def prepare_prefill(
        self,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout = build_prefill_batch_layout(
            seqs,
            self.block_size,
        )
        block_tables = (
            self.prepare_block_tables(seqs)
            if layout.use_block_tables
            else None
        )

        cu_seqlens_q = torch.tensor(
            layout.q_offsets,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            layout.k_offsets,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(
            layout.slot_mapping,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=layout.max_seqlen_q,
            max_seqlen_k=layout.max_seqlen_k,
            slot_mapping=slot_mapping_tensor,
            block_tables=block_tables,
            state_slots=layout.state_slots,
            state_prefix_lens=layout.state_prefix_lens,
            prefill_q_offsets=layout.q_offsets,
            prefill_k_offsets=layout.k_offsets,
        )

        return (
            torch.tensor(
                layout.input_ids,
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True),
            torch.tensor(
                layout.positions,
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True),
        )

    def prepare_decode(
        self,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seen_slots: set[int] = set()
        for seq in seqs:
            if seq.num_cached_tokens != seq.num_state_tokens:
                raise RuntimeError(
                    f"sequence {seq.seq_id} KV/state prefix mismatch: "
                    f"kv={seq.num_cached_tokens}, "
                    f"state={seq.num_state_tokens}"
                )
            if seq.num_cached_tokens != len(seq) - 1:
                raise RuntimeError(
                    f"sequence {seq.seq_id} decode prefix mismatch: "
                    f"committed={seq.num_cached_tokens}, "
                    f"expected={len(seq) - 1}"
                )
            if seq.state_slot < 0:
                raise RuntimeError(
                    f"sequence {seq.seq_id} decode has no state slot"
                )
            if seq.state_slot in seen_slots:
                raise RuntimeError(
                    f"duplicate state slot {seq.state_slot} in decode batch"
                )
            seen_slots.add(seq.state_slot)

        input_ids = [seq.last_token for seq in seqs]
        positions = [len(seq) - 1 for seq in seqs]
        context_lens = [len(seq) for seq in seqs]
        state_slots = tuple(
            seq.state_slot for seq in seqs
        )
        state_prefix_lens = tuple(
            seq.num_state_tokens for seq in seqs
        )
        slot_mapping = [
            seq.block_table[-1] * self.block_size
            + seq.last_block_num_tokens
            - 1
            for seq in seqs
        ]

        set_context(
            is_prefill=False,
            slot_mapping=torch.tensor(
                slot_mapping,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True),
            context_lens=torch.tensor(
                context_lens,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True),
            block_tables=self.prepare_block_tables(seqs),
            state_slots=state_slots,
            state_prefix_lens=state_prefix_lens,
        )
        return (
            torch.tensor(
                input_ids,
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True),
            torch.tensor(
                positions,
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True),
        )

    def _sample_indices(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> list[int]:
        if not is_prefill:
            return list(range(len(seqs)))
        return [
            idx
            for idx, seq in enumerate(seqs)
            if (
                seq.num_cached_tokens
                + seq.num_scheduled_tokens
                == seq.num_tokens
            )
        ]

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
        )

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> list[int | None]:
        try:
            if not is_prefill and any(
                seq.pending_state_snapshot is not None
                for seq in seqs
            ):
                raise RuntimeError(
                    "pending GDN prefix restore is valid only for prefill"
                )
            self._restore_pending_states(seqs)

            if is_prefill:
                input_ids, positions = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)

            hidden_states = self.run_model(
                input_ids,
                positions,
            )
            sample_indices = self._sample_indices(
                seqs,
                is_prefill,
            )
            results: list[int | None] = [
                None
                for _ in seqs
            ]
            if sample_indices:
                logits = self.model.compute_logits(
                    hidden_states,
                    sample_indices if is_prefill else None,
                )
                temperatures = torch.tensor(
                    [
                        seqs[idx].temperature
                        for idx in sample_indices
                    ],
                    dtype=torch.float32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                sampled = self.sampler(
                    logits,
                    temperatures,
                ).tolist()
                for idx, token_id in zip(
                    sample_indices,
                    sampled,
                ):
                    results[idx] = token_id
            return results
        finally:
            reset_context()
