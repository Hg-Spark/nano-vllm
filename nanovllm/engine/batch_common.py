from __future__ import annotations

from contextlib import contextmanager
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
    kv_batch_indices: tuple[int, ...]
    kv_positions: tuple[int, ...]
    state_slots: tuple[int, ...]
    state_prefix_lens: tuple[int, ...]
    paged_kv_indptr: tuple[int, ...]
    paged_kv_indices: tuple[int, ...]
    paged_kv_last_page_len: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DecodeBatchLayout:
    input_ids: tuple[int, ...]
    rope_positions: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    kv_batch_indices: tuple[int, ...]
    kv_positions: tuple[int, ...]
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
    image = seq.image_state
    if image is None or start >= seq.num_prompt_tokens:
        delta = image.rope_delta if image is not None else 0
        for axis in destination:
            axis.extend(range(start + delta, end + delta))
        return

    prompt_end = min(end, seq.num_prompt_tokens)
    for axis, positions in zip(destination, image.rope_positions):
        axis.extend(positions[start:prompt_end])
    if prompt_end < end:
        for axis in destination:
            axis.extend(range(prompt_end + image.rope_delta, end + image.rope_delta))


def stage_pinned(host: torch.Tensor, values) -> torch.Tensor:
    """Copy eager or graph metadata into an existing page-locked host buffer."""
    import torch

    source = torch.as_tensor(values, dtype=host.dtype, device="cpu")
    staged = host.view(-1)[: source.numel()].view(source.shape)
    staged.copy_(source)
    return staged


class PinnedMetadataPool:
    """Reusable page-locked staging for eager batch metadata."""

    def __init__(self) -> None:
        self.buffers = {}
        self._transfer_done = None
        self._pending = False
        self._batching = False

    def _wait_for_transfers(self) -> None:
        if self._pending:
            self._transfer_done.synchronize()
            self._pending = False

    @contextmanager
    def batch(self):
        """Protect all pinned metadata buffers using one event per batch."""
        import torch

        self._wait_for_transfers()
        self._batching = True
        try:
            yield self.to_cuda
        finally:
            self._batching = False
            if self._transfer_done is not None:
                self._transfer_done.record(torch.cuda.current_stream())
                self._pending = True

    def to_cuda(self, name: str, values, dtype: str) -> torch.Tensor:
        import torch

        if not self._batching:
            self._wait_for_transfers()
        source = torch.as_tensor(values, dtype=getattr(torch, dtype), device="cpu")
        required = source.numel()
        host = self.buffers.get(name)
        if host is None or host.numel() < required:
            host = torch.empty(required, dtype=source.dtype, device="cpu", pin_memory=True)
            self.buffers[name] = host
        staged = host[:required].view(source.shape)
        staged.copy_(source)
        result = staged.cuda(non_blocking=True)
        if self._transfer_done is None:
            self._transfer_done = torch.cuda.Event()
        if not self._batching:
            self._transfer_done.record(torch.cuda.current_stream())
            self._pending = True
        return result

    def clear(self) -> None:
        self._wait_for_transfers()
        self._transfer_done = None
        self.buffers.clear()


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
    kv_batch_indices: list[int] = []
    kv_positions: list[int] = []
    state_slots: list[int] = []
    state_prefix_lens: list[int] = []

    for batch_index, chunk in enumerate(chunks):
        seq, start, end = chunk.seq, chunk.start, chunk.end
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

        kv_batch_indices.extend([batch_index] * chunk.num_tokens)
        kv_positions.extend(range(start, end))

    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _paged_kv_metadata(
        [chunk.seq for chunk in chunks], kv_lens, block_size
    )
    return PrefillBatchLayout(
        input_ids=tuple(input_ids),
        rope_positions=tuple(tuple(axis) for axis in rope_positions),
        q_offsets=tuple(q_offsets),
        image_copy_spans=tuple(image_copy_spans),
        kv_batch_indices=tuple(kv_batch_indices),
        kv_positions=tuple(kv_positions),
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
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _paged_kv_metadata(
        seqs, kv_lens, block_size
    )
    return DecodeBatchLayout(
        input_ids=tuple(seq[-1] for seq in seqs),
        rope_positions=tuple(tuple(axis) for axis in rope_positions),
        kv_batch_indices=tuple(range(len(seqs))),
        kv_positions=tuple(len(seq) - 1 for seq in seqs),
        state_slots=tuple(seq.state_slot for seq in seqs),
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )


def prepare_prefill(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
    host_pool: PinnedMetadataPool,
) -> PreparedBatch:
    layout = build_prefill_batch_layout(chunks, block_size)
    with host_pool.batch() as upload:
        checkpoint_interval = next(
            (chunk.checkpoint_interval for chunk in chunks if chunk.checkpoint_interval), 0
        )
        checkpoint_offsets = [0]
        for chunk in chunks:
            count = chunk.num_tokens // checkpoint_interval if checkpoint_interval else 0
            checkpoint_offsets.append(checkpoint_offsets[-1] + count)
        context = Context(
            is_prefill=True,
            prefill_checkpoint_interval=checkpoint_interval,
            prefill_checkpoint_positions=tuple(chunk.checkpoint_positions for chunk in chunks),
            prefill_checkpoint_offsets=tuple(checkpoint_offsets),
            checkpoint_cu_starts=(
                upload("gdn_checkpoint_offsets", checkpoint_offsets, "int64")
                if checkpoint_interval
                else None
            ),
            kv_batch_indices=upload("prefill_kv_batch_indices", layout.kv_batch_indices, "int32"),
            kv_positions=upload("prefill_kv_positions", layout.kv_positions, "int32"),
            qo_indptr=upload("prefill_q_offsets", layout.q_offsets, "int32"),
            paged_kv_indptr=upload("prefill_kv_indptr", layout.paged_kv_indptr, "int32"),
            paged_kv_indices=upload("prefill_kv_indices", layout.paged_kv_indices, "int32"),
            paged_kv_last_page_len=upload(
                "prefill_page_len", layout.paged_kv_last_page_len, "int32"
            ),
            state_slots=layout.state_slots,
            state_slot_ids=upload("prefill_state_slots", layout.state_slots, "int32"),
            state_prefix_lens=layout.state_prefix_lens,
            prefill_q_offsets=layout.q_offsets,
        )
        return PreparedBatch(
            input_ids=upload("prefill_input_ids", layout.input_ids, "int64"),
            positions=upload(
                "prefill_positions",
                layout.rope_positions[:1]
                if all(chunk.seq.image_state is None for chunk in chunks)
                else layout.rope_positions,
                "int64",
            ),
            context=context,
            image_copy_spans=layout.image_copy_spans,
        )


def prepare_decode(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
    host_pool: PinnedMetadataPool,
) -> PreparedBatch:
    layout = build_decode_batch_layout(chunks, block_size)
    with host_pool.batch() as upload:
        context = Context(
            is_prefill=False,
            kv_batch_indices=upload("decode_kv_batch_indices", layout.kv_batch_indices, "int32"),
            kv_positions=upload("decode_kv_positions", layout.kv_positions, "int32"),
            paged_kv_indptr=upload("decode_kv_indptr", layout.paged_kv_indptr, "int32"),
            paged_kv_indices=upload("decode_kv_indices", layout.paged_kv_indices, "int32"),
            paged_kv_last_page_len=upload(
                "decode_page_len", layout.paged_kv_last_page_len, "int32"
            ),
            state_slot_ids=upload("decode_state_slots", layout.state_slots, "int32"),
        )
        return PreparedBatch(
            input_ids=upload("decode_input_ids", layout.input_ids, "int64"),
            positions=upload(
                "decode_positions",
                layout.rope_positions[:1]
                if all(chunk.seq.image_state is None for chunk in chunks)
                else layout.rope_positions,
                "int64",
            ),
            context=context,
        )
