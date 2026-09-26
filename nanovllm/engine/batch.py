from dataclasses import dataclass

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context


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


def build_prefill_batch_layout(
    seqs: list[Sequence],
    block_size: int,
) -> PrefillBatchLayout:
    """Pack variable-length chunks while preserving request alignment."""
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

        start = seq.committed_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        if end > seq.num_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} prefill range [{start}, {end}) "
                f"exceeds token count {seq.num_tokens}"
            )

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
        k_offsets.append(k_offsets[-1] + end)
        max_seqlen_q = max(max_seqlen_q, seqlen_q)
        max_seqlen_k = max(max_seqlen_k, end)
        state_slots.append(seq.state_slot)
        state_prefix_lens.append(seq.committed_tokens)
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


def prepare_block_tables(seqs: list[Sequence]) -> torch.Tensor:
    max_len = max(len(seq.block_table) for seq in seqs)
    rows = [
        seq.block_table + [-1] * (max_len - len(seq.block_table))
        for seq in seqs
    ]
    return torch.tensor(
        rows,
        dtype=torch.int32,
        pin_memory=True,
    ).cuda(non_blocking=True)


def prepare_prefill(
    seqs: list[Sequence],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = build_prefill_batch_layout(seqs, block_size)
    block_tables = (
        prepare_block_tables(seqs)
        if layout.use_block_tables
        else None
    )

    set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor(
            layout.q_offsets,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True),
        cu_seqlens_k=torch.tensor(
            layout.k_offsets,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True),
        max_seqlen_q=layout.max_seqlen_q,
        max_seqlen_k=layout.max_seqlen_k,
        slot_mapping=torch.tensor(
            layout.slot_mapping,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True),
        block_tables=block_tables,
        state_slots=layout.state_slots,
        state_prefix_lens=layout.state_prefix_lens,
        prefill_q_offsets=layout.q_offsets,
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
    seqs: list[Sequence],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seen_slots: set[int] = set()
    for seq in seqs:
        if seq.committed_tokens != len(seq) - 1:
            raise RuntimeError(
                f"sequence {seq.seq_id} decode prefix mismatch: "
                f"committed={seq.committed_tokens}, "
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
    state_slots = tuple(seq.state_slot for seq in seqs)
    state_prefix_lens = tuple(seq.committed_tokens for seq in seqs)
    slot_mapping = [
        seq.block_table[-1] * block_size
        + (len(seq) - 1) % block_size
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
        block_tables=prepare_block_tables(seqs),
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
