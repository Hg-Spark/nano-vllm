from types import SimpleNamespace
import unittest

import torch

from nanovllm.engine.batch import (
    build_decode_batch_layout,
    build_prefill_batch_layout,
)
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.multimodal.request import (
    ImageState,
    image_state_from_processor_output,
)


def make_state(
    fingerprint="image-a",
    *,
    image_start=2,
    image_end=6,
    num_tokens=9,
):
    positions = tuple(
        tuple(range(num_tokens))
        for _ in range(3)
    )
    return ImageState(
        pixel_values=torch.zeros(4, 3),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        rope_positions=positions,
        rope_delta=-2,
        image_start=image_start,
        image_end=image_end,
        fingerprint=fingerprint,
        grid_signature=(1, 4, 4),
        visual_features=torch.ones(image_end - image_start, 8),
    )


def make_scheduler():
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos_token_ids=(99,),
        kvcache_block_size=4,
        max_prefix_cache_entries=8,
    )
    return Scheduler(config, num_kvcache_blocks=8)


class MultimodalRequestTest(unittest.TestCase):

    def test_processor_output_builds_expanded_positions(self):
        output = {
            "input_ids": torch.tensor(
                [[10, 11, 99, 99, 99, 99, 12]]
            ),
            "mm_token_type_ids": torch.tensor(
                [[0, 0, 1, 1, 1, 1, 0]]
            ),
            "pixel_values": torch.arange(
                12, dtype=torch.float32
            ).reshape(4, 3),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }
        root = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2)
        )

        token_ids, state = image_state_from_processor_output(
            output,
            root,
        )

        self.assertEqual(len(token_ids), 7)
        self.assertEqual((state.image_start, state.image_end), (2, 6))
        self.assertEqual(
            state.rope_positions,
            (
                (0, 1, 2, 2, 2, 2, 4),
                (0, 1, 2, 2, 3, 3, 4),
                (0, 1, 2, 3, 2, 3, 4),
            ),
        )
        self.assertEqual(state.rope_delta, -2)

    def test_processed_image_fingerprint_is_prompt_independent(self):
        root = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2)
        )
        pixel_values = torch.arange(
            12, dtype=torch.float32
        ).reshape(4, 3)
        grid = torch.tensor([[1, 4, 4]])
        short = {
            "input_ids": torch.tensor(
                [[10, 11, 99, 99, 99, 99, 12]]
            ),
            "mm_token_type_ids": torch.tensor(
                [[0, 0, 1, 1, 1, 1, 0]]
            ),
            "pixel_values": pixel_values,
            "image_grid_thw": grid,
        }
        longer = {
            "input_ids": torch.tensor(
                [[10, 11, 99, 99, 99, 99, 12, 13, 14]]
            ),
            "mm_token_type_ids": torch.tensor(
                [[0, 0, 1, 1, 1, 1, 0, 0, 0]]
            ),
            "pixel_values": pixel_values.clone(),
            "image_grid_thw": grid.clone(),
        }

        _, short_state = image_state_from_processor_output(short, root)
        _, long_state = image_state_from_processor_output(longer, root)

        self.assertEqual(short_state.fingerprint, long_state.fingerprint)

    def test_preempt_releases_visual_features(self):
        scheduler = make_scheduler()
        state = make_state(num_tokens=9)
        seq = Sequence(list(range(9)), image_state=state)
        scheduler.add(seq)

        scheduled = scheduler.schedule()
        self.assertEqual(scheduled.prefill_chunks[0].end, 4)
        self.assertIsNotNone(state.visual_features)

        scheduler.preempt(seq)

        self.assertIsNone(state.visual_features)
        self.assertEqual(seq.committed_tokens, 0)

    def test_prefill_image_span_is_clipped_to_chunk(self):
        state = make_state(num_tokens=9)
        seq = Sequence(list(range(9)), image_state=state)
        seq.committed_tokens = 3

        layout = build_prefill_batch_layout(
            (ScheduledChunk(seq, 3, 5),),
            block_size=4,
        )

        self.assertEqual(len(layout.image_copy_spans), 1)
        span = layout.image_copy_spans[0]
        self.assertIs(span.seq, seq)
        self.assertEqual(span.packed_dst_start, 0)
        self.assertEqual(span.feature_src_start, 1)
        self.assertEqual(span.length, 2)

    def test_decode_layout_uses_rope_delta(self):
        state = make_state(num_tokens=7, image_end=6)
        seq = Sequence(list(range(7)), image_state=state)
        seq.append_token(42)
        seq.block_table = [3, 4]
        seq.state_slot = 1
        seq.committed_tokens = 7

        layout = build_decode_batch_layout(
            (ScheduledChunk(seq, 7, 8),),
            block_size=4,
        )

        self.assertEqual(layout.input_ids, (42,))
        self.assertEqual(
            layout.rope_positions,
            ((5,), (5,), (5,)),
        )

    def test_prefix_after_image_requires_same_fingerprint(self):
        scheduler = make_scheduler()
        original_state = make_state("image-a")
        original = Sequence(
            list(range(9)),
            image_state=original_state,
        )
        scheduler.add(original)

        first = scheduler.schedule()
        self.assertFalse(first.prefill_chunks[0].capture_snapshot)
        scheduler.postprocess(first.prefill_chunks, [None], True)

        second = scheduler.schedule()
        self.assertEqual(second.prefill_chunks[0].end, 8)
        self.assertTrue(second.prefill_chunks[0].capture_snapshot)
        snapshot = GDNStateSnapshot(num_tokens=8, layers=())
        scheduler.postprocess(
            second.prefill_chunks,
            [None],
            True,
            {original.seq_id: snapshot},
        )

        final = scheduler.schedule()
        scheduler.postprocess(final.prefill_chunks, [99], True)
        self.assertIsNone(original_state.visual_features)

        newcomer = Sequence(
            list(range(9)),
            image_state=make_state("image-b"),
        )
        scheduler.add(newcomer)
        scheduled = scheduler.schedule()

        self.assertEqual(scheduled.prefill_chunks[0].start, 0)
        self.assertEqual(newcomer.committed_tokens, 0)


if __name__ == "__main__":
    unittest.main()
