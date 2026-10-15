from nanovllm.engine.batch_common import (
    DecodeBatchLayout,
    PinnedMetadataPool,
    PreparedBatch,
    _cuda_int32,
    _cuda_int64,
    build_decode_batch_layout,
)
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.utils.context import Context

__all__ = ["DecodeBatchLayout", "build_decode_batch_layout", "prepare_decode"]


def prepare_decode(
    chunks: tuple[ScheduledChunk, ...],
    block_size: int,
    host_pool: PinnedMetadataPool | None = None,
) -> PreparedBatch:
    layout = build_decode_batch_layout(chunks, block_size)
    context = Context(
        is_prefill=False,
        slot_mapping=_cuda_int32(layout.slot_mapping, pool=host_pool, key="decode_slots"),
        paged_kv_indptr=_cuda_int32(layout.paged_kv_indptr, pool=host_pool, key="decode_kv_indptr"),
        paged_kv_indices=_cuda_int32(
            layout.paged_kv_indices, pool=host_pool, key="decode_kv_indices"
        ),
        paged_kv_last_page_len=_cuda_int32(
            layout.paged_kv_last_page_len, pool=host_pool, key="decode_page_len"
        ),
        state_slot_ids=_cuda_int32(layout.state_slots, pool=host_pool, key="decode_state_slots"),
    )
    return PreparedBatch(
        input_ids=_cuda_int64(layout.input_ids, pool=host_pool, key="decode_input_ids"),
        positions=_cuda_int64(layout.rope_positions, pool=host_pool, key="decode_positions"),
        context=context,
    )
