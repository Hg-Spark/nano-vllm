import unittest
from unittest.mock import Mock, patch

from nanovllm.engine.decode_batch import build_decode_batch_layout, prepare_decode
from nanovllm.engine.prefill_batch import build_prefill_batch_layout, prepare_prefill
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
        self.assertEqual(
            layout.rope_positions,
            (
                (2, 3, 4, 0, 1),
                (2, 3, 4, 0, 1),
                (2, 3, 4, 0, 1),
            ),
        )
        self.assertEqual(layout.q_offsets, (0, 3, 5))
        self.assertEqual(layout.state_slots, (4, 1))
        self.assertEqual(layout.state_prefix_lens, (2, 0))
        self.assertEqual(layout.slot_mapping, (30, 31, 36, 12, 13))
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
        self.assertEqual(layout.state_prefix_lens, (0, 0))
        self.assertEqual(layout.paged_kv_indptr, (0, 1, 2))
        self.assertEqual(layout.paged_kv_indices, (1, 2))
        self.assertEqual(layout.paged_kv_last_page_len, (2, 3))
        self.assertEqual(len(layout.slot_mapping), 5)

    def test_prefill_rejects_missing_kv_blocks_for_both_slot_states(self):
        for slot, error in (
            (-1, "prefill requires allocated KV blocks"),
            (0, "state slot without KV blocks"),
        ):
            with self.subTest(state_slot=slot):
                seq = Sequence([0, 0, 0])
                seq.state_slot = slot
                with self.assertRaisesRegex(RuntimeError, error):
                    build_prefill_batch_layout((ScheduledChunk(seq, 0, 3),), block_size=4)

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

    def test_prefill_and_decode_prepare_use_shared_int64_transfer(self):
        seq = Sequence([7, 8])
        seq.block_table = [1]
        seq.state_slot = 0

        for module, layout_builder, prepare, start, end in (
            (
                "nanovllm.engine.prefill_batch",
                build_prefill_batch_layout,
                prepare_prefill,
                0,
                2,
            ),
            (
                "nanovllm.engine.decode_batch",
                build_decode_batch_layout,
                prepare_decode,
                1,
                2,
            ),
        ):
            with self.subTest(module=module):
                seq.committed_tokens = start
                chunks = (ScheduledChunk(seq, start, end),)
                layout = layout_builder(chunks, block_size=4)
                with (
                    patch(f"{module}._cuda_int32", return_value=Mock()),
                    patch(f"{module}._cuda_int64", side_effect=("ids", "positions")) as transfer,
                ):
                    batch = prepare(chunks, block_size=4)

                self.assertEqual(batch.input_ids, "ids")
                self.assertEqual(batch.positions, "positions")
                self.assertEqual(
                    [call.args[0] for call in transfer.call_args_list],
                    [layout.input_ids, layout.rope_positions],
                )


if __name__ == "__main__":
    unittest.main()
