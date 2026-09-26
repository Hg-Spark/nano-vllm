import unittest

from nanovllm.engine.batch import build_prefill_batch_layout
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence


class PrefillBatchLayoutTest(unittest.TestCase):

    def test_mixed_fresh_and_chunked_sequences_are_packed_by_request(self):
        first = Sequence(list(range(6)))
        first.block_table = [7, 9]
        first.state_slot = 4
        first.committed_tokens = 2

        second = Sequence([10, 11])
        second.block_table = [3]
        second.state_slot = 1

        layout = build_prefill_batch_layout(
            (
                ScheduledChunk(first, 2, 5),
                ScheduledChunk(second, 0, 2),
            ),
            block_size=4,
        )

        self.assertEqual(layout.input_ids, (2, 3, 4, 10, 11))
        self.assertEqual(layout.positions, (2, 3, 4, 0, 1))
        self.assertEqual(layout.q_offsets, (0, 3, 5))
        self.assertEqual(layout.kv_lens, (5, 2))
        self.assertEqual(layout.state_slots, (4, 1))
        self.assertEqual(layout.state_prefix_lens, (2, 0))
        self.assertEqual(layout.slot_mapping, (30, 31, 36, 12, 13))
        self.assertTrue(layout.use_paged_kv)
        self.assertEqual(layout.paged_kv_indptr, (0, 2, 3))
        self.assertEqual(layout.paged_kv_indices, (7, 9, 3))
        self.assertEqual(layout.paged_kv_last_page_len, (1, 2))

    def test_fresh_variable_lengths_use_the_same_paged_kv_path(self):
        first = Sequence([1, 2])
        first.block_table = [1]
        first.state_slot = 0

        second = Sequence([3, 4, 5])
        second.block_table = [2]
        second.state_slot = 1

        layout = build_prefill_batch_layout(
            (
                ScheduledChunk(first, 0, 2),
                ScheduledChunk(second, 0, 3),
            ),
            block_size=4,
        )

        self.assertEqual(layout.q_offsets, (0, 2, 5))
        self.assertEqual(layout.kv_lens, (2, 3))
        self.assertEqual(layout.state_prefix_lens, (0, 0))
        self.assertTrue(layout.use_paged_kv)
        self.assertEqual(layout.paged_kv_indptr, (0, 1, 2))
        self.assertEqual(layout.paged_kv_indices, (1, 2))
        self.assertEqual(layout.paged_kv_last_page_len, (2, 3))
        self.assertEqual(len(layout.slot_mapping), 5)

    def test_warmup_without_persistent_caches_is_supported(self):
        seq = Sequence([0, 0, 0])

        layout = build_prefill_batch_layout(
            (ScheduledChunk(seq, 0, 3),),
            block_size=4,
        )

        self.assertEqual(layout.input_ids, (0, 0, 0))
        self.assertEqual(layout.state_slots, (-1,))
        self.assertEqual(layout.state_prefix_lens, (0,))
        self.assertEqual(layout.slot_mapping, ())
        self.assertFalse(layout.use_paged_kv)
        self.assertEqual(layout.paged_kv_indices, ())

    def test_duplicate_state_slots_are_rejected(self):
        first = Sequence([1])
        first.block_table = [0]
        first.state_slot = 2

        second = Sequence([2])
        second.block_table = [1]
        second.state_slot = 2

        with self.assertRaisesRegex(RuntimeError, "duplicate state slot"):
            build_prefill_batch_layout(
                (
                    ScheduledChunk(first, 0, 1),
                    ScheduledChunk(second, 0, 1),
                ),
                block_size=4,
            )

    def test_state_slot_without_kv_blocks_is_rejected(self):
        seq = Sequence([1, 2, 3])
        seq.state_slot = 0

        with self.assertRaisesRegex(
            RuntimeError,
            "state slot without KV blocks",
        ):
            build_prefill_batch_layout(
                (ScheduledChunk(seq, 0, 3),),
                block_size=4,
            )


if __name__ == "__main__":
    unittest.main()
