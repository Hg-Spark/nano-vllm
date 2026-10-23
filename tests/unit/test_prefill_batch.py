import unittest
from types import SimpleNamespace

from nanovllm.engine.batch_common import (
    build_decode_batch_layout,
    build_prefill_batch_layout,
    stage_pinned,
)
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence


class PrefillBatchLayoutTest(unittest.TestCase):
    def test_staging_supports_padded_graph_metadata(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")

        matrix = torch.empty((3, 4), dtype=torch.int64)
        staged = stage_pinned(matrix, ((1, 2), (3, 4), (5, 6)))
        self.assertEqual(tuple(staged.shape), (3, 2))
        torch.testing.assert_close(staged, torch.tensor([[1, 2], [3, 4], [5, 6]]))
        pages = torch.empty(8, dtype=torch.int32)
        active = stage_pinned(pages, [2, 4, 6])
        self.assertEqual(tuple(active.shape), (3,))
        torch.testing.assert_close(active, torch.tensor([2, 4, 6], dtype=torch.int32))

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
        self.assertEqual(layout.rope_positions, ((2, 3, 4, 0, 1),))
        self.assertEqual(layout.q_offsets, (0, 3, 5))
        self.assertEqual(layout.state_slots, (4, 1))
        self.assertEqual(layout.state_prefix_lens, (2, 0))
        self.assertEqual(layout.kv_batch_indices, (0, 0, 0, 1, 1))
        self.assertEqual(layout.kv_positions, (2, 3, 4, 0, 1))
        self.assertEqual(layout.paged_kv_indptr, (0, 2, 3))
        self.assertEqual(layout.paged_kv_indices, (7, 9, 3))
        self.assertEqual(layout.paged_kv_last_page_len, (1, 2))

    def test_mixed_image_batch_keeps_all_three_axes(self):
        text = Sequence([1, 2, 3, 4])
        text.block_table = [2]
        image = SimpleNamespace(
            image_start=1,
            image_end=3,
            rope_delta=0,
            rope_positions=(
                (0, 1, 2, 3),
                (0, 10, 11, 3),
                (0, 20, 21, 3),
            ),
        )
        visual = Sequence([5, 6, 7, 8], image_state=image)
        visual.block_table = [4]
        layout = build_prefill_batch_layout(
            (
                ScheduledChunk(text, 0, 4),
                ScheduledChunk(visual, 0, 4),
            ),
            block_size=16,
        )
        self.assertEqual(
            layout.rope_positions,
            (
                (0, 1, 2, 3, 0, 1, 2, 3),
                (0, 1, 2, 3, 0, 10, 11, 3),
                (0, 1, 2, 3, 0, 20, 21, 3),
            ),
        )
        self.assertEqual(len(layout.image_copy_spans), 1)
        self.assertEqual(layout.image_copy_spans[0].length, 2)

    def test_decode_append_metadata_uses_request_order_and_logical_positions(self):
        first = Sequence([1, 2, 3, 4, 5])
        first.block_table = [7, 2]
        first.state_slot = 3
        first.committed_tokens = 4
        second = Sequence([6, 7])
        second.block_table = [9]
        second.state_slot = 1
        second.committed_tokens = 1

        layout = build_decode_batch_layout(
            (ScheduledChunk(first, 4, 5), ScheduledChunk(second, 1, 2)), block_size=4
        )

        self.assertEqual(layout.kv_batch_indices, (0, 1))
        self.assertEqual(layout.kv_positions, (4, 1))
        self.assertEqual(layout.paged_kv_indptr, (0, 2, 3))
        self.assertEqual(layout.paged_kv_indices, (7, 2, 9))


if __name__ == "__main__":
    unittest.main()
