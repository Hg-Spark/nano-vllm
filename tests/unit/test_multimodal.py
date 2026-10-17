import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from nanovllm.engine.batch_common import build_decode_batch_layout, build_prefill_batch_layout
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.multimodal.request import (
    ImagePrompt,
    ImageState,
    image_state_from_processor_output,
)
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import make_scheduler


def make_state(*, image_start=2, image_end=6, num_tokens=9):
    positions = tuple(tuple(range(num_tokens)) for _ in range(3))
    return ImageState(
        pixel_values=None,
        image_grid_thw=None,
        rope_positions=positions,
        rope_delta=-2,
        image_start=image_start,
        image_end=image_end,
        visual_features=object(),
    )


class MultimodalRequestTest(unittest.TestCase):
    def test_disabled_vision_rejects_image_before_processor(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(max_visual_tokens_per_request=0)
        with self.assertRaisesRegex(ValueError, "image requests require"):
            engine._prepare_request(ImagePrompt("image", object()), SamplingParams(max_tokens=1))

    def test_image_admission_checks_same_limit(self):
        engine = object.__new__(LLMEngine)
        engine.config = SimpleNamespace(
            root_config=object(),
            max_visual_tokens_per_request=3,
        )
        engine.max_model_len = 64
        engine.max_kv_tokens = 64
        engine._get_processor = lambda: lambda **_: {}
        visual_state = SimpleNamespace(image_start=2, image_end=6)
        with patch(
            "nanovllm.engine.llm_engine.image_state_from_processor_output",
            return_value=([1, 2, 3, 4, 5, 6, 7], visual_state),
        ):
            with self.assertRaisesRegex(ValueError, "max_visual_tokens_per_request"):
                engine._prepare_request(
                    ImagePrompt("image", object()), SamplingParams(max_tokens=1)
                )
            engine.config.max_visual_tokens_per_request = 4
            sequence = engine._prepare_request(
                ImagePrompt("image", object()), SamplingParams(max_tokens=1)
            )
        self.assertIs(sequence.image_state, visual_state)

    @unittest.skipIf(torch is None, "requires torch")
    def test_processor_output_builds_expanded_positions(self):
        output = {
            "input_ids": torch.tensor([[10, 11, 99, 99, 99, 99, 12]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0]]),
            "pixel_values": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }
        root = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        token_ids, state = image_state_from_processor_output(output, root)

        self.assertEqual(len(token_ids), 7)
        self.assertEqual((state.image_start, state.image_end), (2, 6))
        self.assertEqual(
            state.rope_positions,
            ((0, 1, 2, 2, 2, 2, 4), (0, 1, 2, 2, 3, 3, 4), (0, 1, 2, 3, 2, 3, 4)),
        )
        self.assertEqual(state.rope_delta, -2)

    def test_completed_request_releases_visual_features(self):
        scheduler = make_scheduler(
            max_num_seqs=2,
            num_blocks=8,
            block_size=4,
            max_prefix_cache_entries=8,
            eos_token_ids=(99,),
        )
        state = make_state(num_tokens=9)
        seq = Sequence(list(range(9)), SamplingParams(max_tokens=1), image_state=state)
        scheduler.add(seq)

        for _ in range(4):
            chunks = scheduler.schedule().prefill_chunks
            self.assertEqual(len(chunks), 1)
            chunk = chunks[0]
            scheduler.postprocess(
                chunks,
                [99 if chunk.end == len(seq) else None],
                True,
                {
                    seq.seq_id: GDNStateSnapshot(chunk.end, ())
                    for chunk in chunks
                    if chunk.capture_snapshot
                },
            )
            if seq.is_finished:
                break

        self.assertTrue(seq.is_finished)
        self.assertIsNone(state.visual_features)
        self.assertEqual(seq.committed_tokens, 0)

    def test_prefill_image_span_is_clipped_to_chunk(self):
        state = make_state(num_tokens=9)
        seq = Sequence(list(range(9)), image_state=state)
        seq.block_table = [0, 1]
        seq.state_slot = 0
        seq.committed_tokens = 3

        layout = build_prefill_batch_layout((ScheduledChunk(seq, 3, 5),), block_size=4)
        self.assertEqual(len(layout.image_copy_spans), 1)
        span = layout.image_copy_spans[0]
        self.assertEqual((span.packed_dst_start, span.feature_src_start, span.length), (0, 1, 2))

    def test_decode_layout_uses_rope_delta(self):
        state = make_state(num_tokens=7, image_end=6)
        seq = Sequence(list(range(7)), image_state=state)
        seq.append_token(42)
        seq.block_table = [3, 4]
        seq.state_slot = 1
        seq.committed_tokens = 7

        layout = build_decode_batch_layout((ScheduledChunk(seq, 7, 8),), block_size=4)
        self.assertEqual(layout.input_ids, (42,))
        self.assertEqual(layout.rope_positions, ((5,), (5,), (5,)))
        self.assertEqual(layout.kv_positions, (7,))
        self.assertEqual(layout.kv_batch_indices, (0,))


if __name__ == "__main__":
    unittest.main()
