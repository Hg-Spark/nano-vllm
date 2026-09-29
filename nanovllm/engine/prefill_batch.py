import torch

from nanovllm.engine.batch_common import (
    ImageCopySpan,
    PrefillBatchLayout,
    PreparedBatch,
    _append_rope_positions,
    _cuda_int32,
    _cuda_int64_2d,
    _paged_kv_metadata,
)
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.utils.context import Context


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
    rope_positions: list[list[int]] = [[], [], []]
    q_offsets = [0]
    image_copy_spans: list[ImageCopySpan] = []
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

        packed_start = q_offsets[-1]
        input_ids.extend(seq[start:end])
        _append_rope_positions(rope_positions, seq, start, end)
        q_offsets.append(packed_start + seqlen_q)
        kv_lens.append(end)
        state_slots.append(seq.state_slot)
        state_prefix_lens.append(seq.committed_tokens)

        image_state = seq.image_state
        if image_state is not None:
            overlap_start = max(start, image_state.image_start)
            overlap_end = min(end, image_state.image_end)
            if overlap_start < overlap_end:
                image_copy_spans.append(
                    ImageCopySpan(
                        seq=seq,
                        packed_dst_start=(
                            packed_start + overlap_start - start
                        ),
                        feature_src_start=(
                            overlap_start - image_state.image_start
                        ),
                        length=overlap_end - overlap_start,
                    )
                )

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

    has_paged_kv = all(has_block_tables)
    if has_paged_kv and len(slot_mapping) != len(input_ids):
        raise RuntimeError(
            "prefill slot mapping must contain one entry per packed token"
        )

    if has_paged_kv:
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


def prepare_prefill(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
) -> PreparedBatch:
    layout = build_prefill_batch_layout(chunks, block_size)

    has_paged_kv = bool(layout.paged_kv_indptr)
    context = Context(
        is_prefill=True,
        slot_mapping=_cuda_int32(layout.slot_mapping),
        qo_indptr=_cuda_int32(layout.q_offsets),
        paged_kv_indptr=(
            _cuda_int32(layout.paged_kv_indptr)
            if has_paged_kv
            else None
        ),
        paged_kv_indices=(
            _cuda_int32(layout.paged_kv_indices)
            if has_paged_kv
            else None
        ),
        paged_kv_last_page_len=(
            _cuda_int32(layout.paged_kv_last_page_len)
            if has_paged_kv
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
        positions=_cuda_int64_2d(layout.rope_positions),
        context=context,
        image_copy_spans=layout.image_copy_spans,
    )


