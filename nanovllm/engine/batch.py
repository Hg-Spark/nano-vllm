from dataclasses import dataclass

import torch

from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import Context


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    context: Context


@dataclass(frozen=True, slots=True)
class PrefillBatchLayout:
    """CPU control-plane description of one packed prefill batch."""

    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    q_offsets: tuple[int, ...]
    kv_lens: tuple[int, ...]
    slot_mapping: tuple[int, ...]
    state_slots: tuple[int, ...]
    state_prefix_lens: tuple[int, ...]
    paged_kv_indptr: tuple[int, ...]
    paged_kv_indices: tuple[int, ...]
    paged_kv_last_page_len: tuple[int, ...]
    use_paged_kv: bool


def _paged_kv_metadata(
    seqs: list[Sequence],
    kv_lens: list[int] | tuple[int, ...],
    block_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(seqs) != len(kv_lens):
        raise ValueError("sequence/KV-length count mismatch")

    indptr = [0]
    indices: list[int] = []
    last_page_len: list[int] = []
    for seq, kv_len in zip(seqs, kv_lens):
        if kv_len <= 0:
            raise RuntimeError("paged KV sequence length must be positive")
        num_pages = (kv_len + block_size - 1) // block_size
        if len(seq.block_table) < num_pages:
            raise RuntimeError(
                f"sequence {seq.seq_id} block table is too short: "
                f"need {num_pages}, have {len(seq.block_table)}"
            )
        indices.extend(seq.block_table[:num_pages])
        indptr.append(len(indices))
        last_page_len.append((kv_len - 1) % block_size + 1)

    return tuple(indptr), tuple(indices), tuple(last_page_len)


def build_prefill_batch_layout(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
) -> PrefillBatchLayout:
    """Pack variable-length chunks while preserving request alignment."""
    if not chunks:
        raise ValueError("prefill batch must not be empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    has_block_tables = [bool(chunk.seq.block_table) for chunk in chunks]
    if any(has_block_tables) and not all(has_block_tables):
        raise RuntimeError(
            "prefill batch cannot mix allocated and unallocated requests"
        )

    input_ids: list[int] = []
    positions: list[int] = []
    q_offsets = [0]
    kv_lens: list[int] = []
    slot_mapping: list[int] = []
    state_slots: list[int] = []
    state_prefix_lens: list[int] = []
    seen_state_slots: set[int] = set()

    for chunk in chunks:
        seq = chunk.seq
        start = chunk.start
        end = chunk.end
        seqlen_q = chunk.num_tokens
        if start != seq.committed_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} scheduled prefix changed"
            )
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
        kv_lens.append(end)
        state_slots.append(seq.state_slot)
        state_prefix_lens.append(seq.committed_tokens)

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

    use_paged_kv = all(has_block_tables)
    if use_paged_kv and len(slot_mapping) != len(input_ids):
        raise RuntimeError(
            "prefill slot mapping must contain one entry per packed token"
        )

    if use_paged_kv:
        paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            _paged_kv_metadata(
                [chunk.seq for chunk in chunks],
                kv_lens,
                block_size,
            )
        )
    else:
        paged_kv_indptr = ()
        paged_kv_indices = ()
        paged_kv_last_page_len = ()

    return PrefillBatchLayout(
        input_ids=tuple(input_ids),
        positions=tuple(positions),
        q_offsets=tuple(q_offsets),
        kv_lens=tuple(kv_lens),
        slot_mapping=tuple(slot_mapping),
        state_slots=tuple(state_slots),
        state_prefix_lens=tuple(state_prefix_lens),
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        use_paged_kv=use_paged_kv,
    )


def _cuda_int32(values: tuple[int, ...] | list[int]) -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=torch.int32,
        pin_memory=True,
    ).cuda(non_blocking=True)


def prepare_prefill(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
) -> PreparedBatch:
    layout = build_prefill_batch_layout(chunks, block_size)

    context = Context(
        is_prefill=True,
        slot_mapping=_cuda_int32(layout.slot_mapping),
        qo_indptr=_cuda_int32(layout.q_offsets),
        paged_kv_indptr=(
            _cuda_int32(layout.paged_kv_indptr)
            if layout.use_paged_kv
            else None
        ),
        paged_kv_indices=(
            _cuda_int32(layout.paged_kv_indices)
            if layout.use_paged_kv
            else None
        ),
        paged_kv_last_page_len=(
            _cuda_int32(layout.paged_kv_last_page_len)
            if layout.use_paged_kv
            else None
        ),
        state_slots=layout.state_slots,
        state_prefix_lens=layout.state_prefix_lens,
        prefill_q_offsets=layout.q_offsets,
    )

    return PreparedBatch(
        input_ids=torch.tensor(
            layout.input_ids,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True),
        positions=torch.tensor(
            layout.positions,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True),
        context=context,
    )


def prepare_decode(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
) -> PreparedBatch:
    if not chunks:
        raise ValueError("decode batch must not be empty")

    seen_slots: set[int] = set()
    seqs = [chunk.seq for chunk in chunks]
    for chunk in chunks:
        seq = chunk.seq
        if (
            chunk.start != seq.committed_tokens
            or chunk.end != len(seq)
            or chunk.num_tokens != 1
        ):
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
        if not seq.block_table:
            raise RuntimeError(
                f"sequence {seq.seq_id} decode has no KV blocks"
            )
        seen_slots.add(seq.state_slot)

    input_ids = [seq.last_token for seq in seqs]
    positions = [len(seq) - 1 for seq in seqs]
    kv_lens = [len(seq) for seq in seqs]
    state_slots = tuple(seq.state_slot for seq in seqs)
    state_prefix_lens = tuple(seq.committed_tokens for seq in seqs)
    slot_mapping = [
        seq.block_table[(len(seq) - 1) // block_size] * block_size
        + (len(seq) - 1) % block_size
        for seq in seqs
    ]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
        _paged_kv_metadata(seqs, kv_lens, block_size)
    )

    context = Context(
        is_prefill=False,
        slot_mapping=_cuda_int32(slot_mapping),
        paged_kv_indptr=_cuda_int32(paged_kv_indptr),
        paged_kv_indices=_cuda_int32(paged_kv_indices),
        paged_kv_last_page_len=_cuda_int32(paged_kv_last_page_len),
        state_slots=state_slots,
        state_prefix_lens=state_prefix_lens,
    )
    return PreparedBatch(
        input_ids=torch.tensor(
            input_ids,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True),
        positions=torch.tensor(
            positions,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True),
        context=context,
    )
