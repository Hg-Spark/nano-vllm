import torch

from nanovllm.engine.batch_common import (
    DecodeBatchLayout,
    PreparedBatch,
    _cuda_int32,
    _cuda_int64_2d,
    _paged_kv_metadata,
)
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.utils.context import Context


def build_decode_batch_layout(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
) -> DecodeBatchLayout:
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

    input_ids = tuple(seq.last_token for seq in seqs)
    rope_positions = [[], [], []]
    for seq in seqs:
        position = seq.position_at(len(seq) - 1)
        for axis in range(3):
            rope_positions[axis].append(position[axis])

    kv_lens = [len(seq) for seq in seqs]
    state_slots = tuple(seq.state_slot for seq in seqs)
    slot_mapping = tuple(
        seq.block_table[(len(seq) - 1) // block_size] * block_size
        + (len(seq) - 1) % block_size
        for seq in seqs
    )
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
        _paged_kv_metadata(seqs, kv_lens, block_size)
    )

    return DecodeBatchLayout(
        input_ids=input_ids,
        rope_positions=tuple(tuple(axis) for axis in rope_positions),
        slot_mapping=slot_mapping,
        state_slots=state_slots,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )


def prepare_decode(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
) -> PreparedBatch:
    layout = build_decode_batch_layout(chunks, block_size)
    context = Context(
        is_prefill=False,
        slot_mapping=_cuda_int32(layout.slot_mapping),
        paged_kv_indptr=_cuda_int32(layout.paged_kv_indptr),
        paged_kv_indices=_cuda_int32(layout.paged_kv_indices),
        paged_kv_last_page_len=_cuda_int32(
            layout.paged_kv_last_page_len
        ),
        state_slot_ids=_cuda_int32(layout.state_slots),
    )
    return PreparedBatch(
        input_ids=torch.tensor(
            layout.input_ids,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True),
        positions=_cuda_int64_2d(layout.rope_positions),
        context=context,
    )
