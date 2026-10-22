"""Prefill RoPE metadata only builds the axes needed by a batch."""

import unittest
from types import SimpleNamespace

from nanovllm.engine.batch_common import build_prefill_batch_layout
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence


class PrefillRopeLayoutTest(unittest.TestCase):
    def test_text_only_batch_uses_single_axis(self):
        first = Sequence([1, 2, 3, 4])
        second = Sequence([5, 6, 7])
        first.block_table = [7]
        second.block_table = [3]
        chunks = (
            ScheduledChunk(first, 0, 4),
            ScheduledChunk(second, 0, 3),
        )
        layout = build_prefill_batch_layout(chunks, block_size=16)
        self.assertEqual(layout.rope_positions, ((0, 1, 2, 3, 0, 1, 2),))
        self.assertEqual(layout.paged_kv_indices, (7, 3))

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


if __name__ == "__main__":
    unittest.main()
