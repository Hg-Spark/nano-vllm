from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import Context

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class ImageCopySpan:
    seq: Sequence
    packed_dst_start: int
    feature_src_start: int
    length: int


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    context: Context
    image_copy_spans: tuple[ImageCopySpan, ...] = ()


@dataclass(frozen=True, slots=True)
class PrefillBatchLayout:
    input_ids: tuple[int, ...]
    rope_positions: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    q_offsets: tuple[int, ...]
    image_copy_spans: tuple[ImageCopySpan, ...]
    slot_mapping: tuple[int, ...]
    state_slots: tuple[int, ...]
    state_prefix_lens: tuple[int, ...]
    paged_kv_indptr: tuple[int, ...]
    paged_kv_indices: tuple[int, ...]
    paged_kv_last_page_len: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DecodeBatchLayout:
    input_ids: tuple[int, ...]
    rope_positions: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    slot_mapping: tuple[int, ...]
    state_slots: tuple[int, ...]
    paged_kv_indptr: tuple[int, ...]
    paged_kv_indices: tuple[int, ...]
    paged_kv_last_page_len: tuple[int, ...]


def _paged_kv_metadata(
    seqs: list[Sequence],
    kv_lens: list[int] | tuple[int, ...],
    block_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    indptr = [0]
    indices: list[int] = []
    last_page_len: list[int] = []
    for seq, kv_len in zip(seqs, kv_lens):
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


def _append_rope_positions(
    destination: list[list[int]],
    seq: Sequence,
    start: int,
    end: int,
) -> None:
    for index in range(start, end):
        position = seq.position_at(index)
        for axis in range(3):
            destination[axis].append(position[axis])


def _pinned_cpu_tensor(values, dtype: torch.dtype) -> torch.Tensor:
    import torch

    return torch.tensor(values, dtype=dtype, pin_memory=True)


def _cuda_int32(values: tuple[int, ...] | list[int]) -> torch.Tensor:
    import torch

    return _pinned_cpu_tensor(values, torch.int32).cuda(non_blocking=True)


def _cuda_int64(values: tuple[int, ...] | tuple[tuple[int, ...], ...]) -> torch.Tensor:
    import torch

    return _pinned_cpu_tensor(values, torch.int64).cuda(non_blocking=True)


def build_prefill_batch_layout(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
) -> PrefillBatchLayout:
    if not chunks or block_size <= 0:
        raise ValueError("prefill batch must be non-empty with positive block_size")

    input_ids: list[int] = []
    rope_positions: list[list[int]] = [[], [], []]
    q_offsets = [0]
    image_copy_spans: list[ImageCopySpan] = []
    kv_lens: list[int] = []
    slot_mapping: list[int] = []
    state_slots: list[int] = []
    state_prefix_lens: list[int] = []
    seen_slots: set[int] = set()

    for chunk in chunks:
        seq, start, end = chunk.seq, chunk.start, chunk.end
        if not seq.block_table:
            if seq.state_slot >= 0:
                raise RuntimeError(f"sequence {seq.seq_id} has a state slot without KV blocks")
            raise RuntimeError("prefill requires allocated KV blocks")
        if seq.state_slot < 0:
            raise RuntimeError(f"sequence {seq.seq_id} has KV blocks without a state slot")
        if seq.state_slot in seen_slots:
            raise RuntimeError(f"duplicate state slot {seq.state_slot} in prefill batch")
        seen_slots.add(seq.state_slot)

        packed_start = q_offsets[-1]
        input_ids.extend(seq[start:end])
        _append_rope_positions(rope_positions, seq, start, end)
        q_offsets.append(packed_start + chunk.num_tokens)
        kv_lens.append(end)
        state_slots.append(seq.state_slot)
        state_prefix_lens.append(seq.committed_tokens)

        if seq.image_state is not None:
            overlap_start = max(start, seq.image_state.image_start)
            overlap_end = min(end, seq.image_state.image_end)
            if overlap_start < overlap_end:
                image_copy_spans.append(
                    ImageCopySpan(
                        seq=seq,
                        packed_dst_start=packed_start + overlap_start - start,
                        feature_src_start=overlap_start - seq.image_state.image_start,
                        length=overlap_end - overlap_start,
                    )
                )

        for pos in range(start, end):
            slot_mapping.append(seq.block_table[pos // block_size] * block_size + pos % block_size)

    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _paged_kv_metadata(
        [chunk.seq for chunk in chunks], kv_lens, block_size
    )
    return PrefillBatchLayout(
        input_ids=tuple(input_ids),
        rope_positions=tuple(tuple(axis) for axis in rope_positions),
        q_offsets=tuple(q_offsets),
        image_copy_spans=tuple(image_copy_spans),
        slot_mapping=tuple(slot_mapping),
        state_slots=tuple(state_slots),
        state_prefix_lens=tuple(state_prefix_lens),
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )


def build_decode_batch_layout(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
) -> DecodeBatchLayout:
    if not chunks:
        raise ValueError("decode batch must not be empty")
    seqs = [chunk.seq for chunk in chunks]
    rope_positions: list[list[int]] = [[], [], []]
    for seq in seqs:
        _append_rope_positions(rope_positions, seq, len(seq) - 1, len(seq))

    kv_lens = [len(seq) for seq in seqs]
    slot_mapping = tuple(
        seq.block_table[(len(seq) - 1) // block_size] * block_size + (len(seq) - 1) % block_size
        for seq in seqs
    )
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _paged_kv_metadata(
        seqs, kv_lens, block_size
    )
    return DecodeBatchLayout(
        input_ids=tuple(seq.last_token for seq in seqs),
        rope_positions=tuple(tuple(axis) for axis in rope_positions),
        slot_mapping=slot_mapping,
        state_slots=tuple(seq.state_slot for seq in seqs),
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )
