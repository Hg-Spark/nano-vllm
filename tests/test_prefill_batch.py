import unittest

from nanovllm.engine.model_runner import build_prefill_batch_layout
from nanovllm.engine.sequence import Sequence


class PrefillBatchLayoutTest(unittest.TestCase):

    def test_mixed_fresh_and_chunked_sequences_are_packed_by_request(self):
        first = Sequence(list(range(6)))
        first.block_table = [7, 9]
        first.state_slot = 4
        first.num_cached_tokens = 2
        first.num_state_tokens = 2
        first.num_scheduled_tokens = 3

        second = Sequence([10, 11])
        second.block_table = [3]
        second.state_slot = 1
        second.num_scheduled_tokens = 2

        layout = build_prefill_batch_layout(
            [first, second],
            block_size=4,
        )

        self.assertEqual(
            layout.input_ids,
            (2, 3, 4, 10, 11),
        )
        self.assertEqual(
            layout.positions,
            (2, 3, 4, 0, 1),
        )
        self.assertEqual(layout.q_offsets, (0, 3, 5))
        self.assertEqual(layout.k_offsets, (0, 5, 7))
        self.assertEqual(layout.max_seqlen_q, 3)
        self.assertEqual(layout.max_seqlen_k, 5)
        self.assertEqual(layout.state_slots, (4, 1))
        self.assertEqual(layout.state_prefix_lens, (2, 0))
        self.assertEqual(
            layout.slot_mapping,
            (30, 31, 36, 12, 13),
        )
        self.assertTrue(layout.use_block_tables)

    def test_fresh_variable_lengths_do_not_require_paged_readback(self):
        first = Sequence([1, 2])
        first.block_table = [1]
        first.state_slot = 0
        first.num_scheduled_tokens = 2

        second = Sequence([3, 4, 5])
        second.block_table = [2]
        second.state_slot = 1
        second.num_scheduled_tokens = 3

        layout = build_prefill_batch_layout(
            [first, second],
            block_size=4,
        )

        self.assertEqual(layout.q_offsets, (0, 2, 5))
        self.assertEqual(layout.k_offsets, (0, 2, 5))
        self.assertEqual(layout.state_prefix_lens, (0, 0))
        self.assertFalse(layout.use_block_tables)
        self.assertEqual(len(layout.slot_mapping), 5)

    def test_warmup_without_persistent_caches_is_supported(self):
        seq = Sequence([0, 0, 0])
        seq.num_scheduled_tokens = 3

        layout = build_prefill_batch_layout(
            [seq],
            block_size=4,
        )

        self.assertEqual(layout.input_ids, (0, 0, 0))
        self.assertEqual(layout.state_slots, (-1,))
        self.assertEqual(layout.state_prefix_lens, (0,))
        self.assertEqual(layout.slot_mapping, ())
        self.assertFalse(layout.use_block_tables)

    def test_duplicate_state_slots_are_rejected(self):
        first = Sequence([1])
        first.block_table = [0]
        first.state_slot = 2
        first.num_scheduled_tokens = 1

        second = Sequence([2])
        second.block_table = [1]
        second.state_slot = 2
        second.num_scheduled_tokens = 1

        with self.assertRaisesRegex(
            RuntimeError,
            "duplicate state slot",
        ):
            build_prefill_batch_layout(
                [first, second],
                block_size=4,
            )

    def test_kv_state_prefix_mismatch_is_rejected(self):
        seq = Sequence([1, 2, 3])
        seq.block_table = [0]
        seq.state_slot = 0
        seq.num_cached_tokens = 1
        seq.num_state_tokens = 0
        seq.num_scheduled_tokens = 2

        with self.assertRaisesRegex(
            RuntimeError,
            "KV/state prefix mismatch",
        ):
            build_prefill_batch_layout(
                [seq],
                block_size=4,
            )


if __name__ == "__main__":
    unittest.main()
