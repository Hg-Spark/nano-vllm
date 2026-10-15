from nanovllm.engine.batch_common import (
    ImageCopySpan,
    PrefillBatchLayout,
    PreparedBatch,
    PinnedMetadataPool,
    _cuda_int32,
    _cuda_int64,
    build_prefill_batch_layout,
)
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.utils.context import Context

__all__ = ["ImageCopySpan", "PrefillBatchLayout", "build_prefill_batch_layout", "prepare_prefill"]


def prepare_prefill(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
    host_pool: PinnedMetadataPool | None = None,
) -> PreparedBatch:
    layout = build_prefill_batch_layout(chunks, block_size)
    context = Context(
        is_prefill=True,
        slot_mapping=_cuda_int32(layout.slot_mapping, pool=host_pool, key="prefill_slots"),
        qo_indptr=_cuda_int32(layout.q_offsets, pool=host_pool, key="prefill_q_offsets"),
        paged_kv_indptr=_cuda_int32(layout.paged_kv_indptr, pool=host_pool, key="prefill_kv_indptr"),
        paged_kv_indices=_cuda_int32(layout.paged_kv_indices, pool=host_pool, key="prefill_kv_indices"),
        paged_kv_last_page_len=_cuda_int32(layout.paged_kv_last_page_len, pool=host_pool, key="prefill_page_len"),
        state_slots=layout.state_slots,
        state_slot_ids=_cuda_int32(layout.state_slots, pool=host_pool, key="prefill_state_slots"),
        state_prefix_lens=layout.state_prefix_lens,
        prefill_q_offsets=layout.q_offsets,
    )
    return PreparedBatch(
        input_ids=_cuda_int64(layout.input_ids, pool=host_pool, key="prefill_input_ids"),
        positions=_cuda_int64(layout.rope_positions, pool=host_pool, key="prefill_positions"),
        context=context,
        image_copy_spans=layout.image_copy_spans,
    )
