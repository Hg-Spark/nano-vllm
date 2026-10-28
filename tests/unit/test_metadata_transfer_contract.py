"""Host-only FlashInfer plan metadata must not be uploaded before planning."""

import unittest
from contextlib import contextmanager

from nanovllm.engine.batch_common import (
    DecodePageTableCache,
    prepare_decode,
    prepare_prefill,
)
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence


class RecordingPool:
    def __init__(self):
        self.uploads = []
        self.host_stages = []

    @contextmanager
    def batch(self):
        yield self.to_cuda

    def stage_host(self, name, values, dtype):
        self.host_stages.append(name)
        return tuple(values)

    def to_cuda(self, name, values, dtype):
        self.uploads.append(name)
        return tuple(values)

    def to_cuda_cached(self, name, values, dtype, key):
        return self.to_cuda(name, values, dtype)


class MetadataTransferContractTest(unittest.TestCase):
    def test_prefill_plan_only_metadata_stays_on_host(self):
        seq = Sequence([1, 2, 3, 4])
        seq.block_table = [3]
        seq.state_slot = 0
        pool = RecordingPool()
        prepared = prepare_prefill((ScheduledChunk(seq, 0, 4),), 16, pool)

        self.assertEqual(
            pool.host_stages,
            ["prefill_q_offsets", "prefill_kv_indptr", "prefill_page_len"],
        )
        self.assertFalse(
            {"prefill_q_offsets", "prefill_kv_indptr", "prefill_page_len"} & set(pool.uploads)
        )
        self.assertIsNone(prepared.context.qo_indptr)
        self.assertIsNone(prepared.context.paged_kv_indptr)
        self.assertIsNone(prepared.context.paged_kv_last_page_len)
        self.assertEqual(prepared.context.paged_kv_indices, (3,))

    def test_decode_plan_only_metadata_stays_on_host(self):
        seq = Sequence([1, 2, 3, 4])
        seq.block_table = [5]
        seq.state_slot = 0
        pool = RecordingPool()
        prepared = prepare_decode((ScheduledChunk(seq, 3, 4),), 16, pool, DecodePageTableCache())

        self.assertEqual(pool.host_stages, ["decode_kv_indptr", "decode_page_len"])
        self.assertFalse({"decode_kv_indptr", "decode_page_len"} & set(pool.uploads))
        self.assertIsNone(prepared.context.paged_kv_indptr)
        self.assertIsNone(prepared.context.paged_kv_last_page_len)
        self.assertEqual(prepared.context.paged_kv_indices, (5,))


if __name__ == "__main__":
    unittest.main()
