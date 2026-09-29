from dataclasses import dataclass

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import Context


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
    """CPU control-plane description of one packed prefill batch."""

    input_ids: tuple[int, ...]
    rope_positions: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
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
    """CPU control-plane description of one single-token decode batch."""

    input_ids: tuple[int, ...]
    rope_positions: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
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


def _cuda_int32(values: tuple[int, ...] | list[int]) -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=torch.int32,
        pin_memory=True,
    ).cuda(non_blocking=True)


def _cuda_int64_2d(
    values: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=torch.int64,
        pin_memory=True,
    ).cuda(non_blocking=True)


