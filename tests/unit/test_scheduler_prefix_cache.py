import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine import cache_runtime
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from scripts.verify_prefix_resume import run_prefix_resume_probe
from tests.scheduler_helpers import (
    make_scheduler,
    scheduled_sequences,
)


class SchedulerPrefixCacheTest(unittest.TestCase):
    def test_prefix_probe_reads_cache_metadata(self):
        snapshot = GDNStateSnapshot(16, ())
        entry = SimpleNamespace(num_tokens=16, block_ids=(0,), state_snapshot=snapshot)
        cache = SimpleNamespace(snapshot_nbytes=32, find=lambda *_a, **_kw: entry)

        def fake_llm(_path, **options):
            def generate(prompts, _params, **_kwargs):
                if options["max_prefix_cache_entries"] and prompts[0][-1] == 24:
                    cache_runtime.restore_gdn_state(
                        None, SimpleNamespace(committed_tokens=16, block_table=[0]), snapshot
                    )
                return [{"token_ids": [7, 7]}]

            return SimpleNamespace(
                generate=generate,
                exit=lambda: None,
                scheduler=SimpleNamespace(block_manager=SimpleNamespace(prefix_cache=cache)),
            )

        tokenizer = SimpleNamespace(
            encode=lambda text, **_kw: (
                [21, 22] if "branch A." in text else [23, 24] if "branch B." in text else [1]
            )
        )
        with (
            patch("nanovllm.LLM", side_effect=fake_llm),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("nanovllm.engine.cache_runtime.restore_gdn_state"),
            patch("torch.cuda.empty_cache"),
        ):
            result = run_prefix_resume_probe("stub", 2, 128, 16, "float16")
        self.assertEqual(result["prefix"], {"num_tokens": 16, "snapshot_bytes": 32})
        self.assertEqual(result["restore_boundary"], 16)

    def test_joint_prefix_hit_restores_kv_and_gdn_boundary_together(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
            num_blocks=8,
            block_size=4,
        )
        original = Sequence([0, 1, 2, 3, 4, 5], SamplingParams(max_tokens=1))
        scheduler.add(original)

        first = scheduler.schedule()
        self.assertEqual(first.prefill_tokens, 4)
        self.assertTrue(first.prefill_chunks[0].capture_snapshot)
        cached_block = original.block_table[0]
        snapshot = GDNStateSnapshot(num_tokens=4, layers=())
        scheduler.postprocess(
            first.prefill_chunks,
            [None],
            True,
            {original.seq_id: snapshot},
        )
        self.assertEqual(len(scheduler.block_manager.prefix_cache), 1)
        self.assertEqual(
            scheduler.block_manager.block_refcounts[cached_block],
            2,
        )

        final = scheduler.schedule()
        scheduler.postprocess(
            final.prefill_chunks,
            [99],
            True,
        )
        self.assertTrue(original.is_finished)
        self.assertEqual(
            scheduler.block_manager.block_refcounts[cached_block],
            1,
        )

        newcomer = Sequence([0, 1, 2, 3, 7, 8], SamplingParams(max_tokens=1))
        scheduler.add(newcomer)
        resumed = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(resumed.prefill_chunks),
            [newcomer],
        )
        self.assertEqual(newcomer.committed_tokens, 4)
        self.assertIs(newcomer.pending_state_snapshot, snapshot)
        self.assertEqual(newcomer.block_table[0], cached_block)
        self.assertEqual(resumed.prefill_chunks[0].num_tokens, 2)
        self.assertEqual(
            scheduler.block_manager.block_refcounts[cached_block],
            2,
        )

    def test_aligned_prompt_restores_reusable_partial_prefix(self):
        scheduler = make_scheduler(max_num_batched_tokens=16, max_num_seqs=2, block_size=4)
        tokens = list(range(8))
        original = Sequence(tokens, SamplingParams(max_tokens=1))
        scheduler.add(original)

        first = scheduler.schedule()
        self.assertEqual(
            [(chunk.end, chunk.capture_snapshot) for chunk in first.prefill_chunks],
            [(4, True)],
        )
        snapshot = GDNStateSnapshot(num_tokens=4, layers=())
        scheduler.postprocess(first.prefill_chunks, [None], True, {original.seq_id: snapshot})
        final = scheduler.schedule()
        self.assertTrue(final.prefill_chunks[0].capture_snapshot)
        full_snapshot = GDNStateSnapshot(num_tokens=8, layers=())
        scheduler.postprocess(final.prefill_chunks, [99], True, {original.seq_id: full_snapshot})

        newcomer = Sequence(tokens, SamplingParams(max_tokens=1))
        scheduler.add(newcomer)
        completed = scheduler.schedule()
        self.assertEqual(newcomer.committed_tokens, 4)
        self.assertIs(newcomer.pending_state_snapshot, snapshot)
        scheduler.postprocess(completed.prefill_chunks, [99], True)

        extended = Sequence(tokens + [8], SamplingParams(max_tokens=1))
        scheduler.add(extended)
        scheduler.schedule()
        self.assertEqual(extended.committed_tokens, 8)
        self.assertIs(extended.pending_state_snapshot, full_snapshot)

    def test_identical_prefills_capture_only_one_snapshot(self):
        scheduler = make_scheduler(max_num_batched_tokens=16, max_num_seqs=2, block_size=4)
        for _ in range(2):
            scheduler.add(Sequence([0, 1, 2, 3, 4, 5], SamplingParams(max_tokens=1)))
        chunks = scheduler.schedule().prefill_chunks
        self.assertEqual(
            [(chunk.end, chunk.capture_snapshot) for chunk in chunks],
            [(4, True), (6, False)],
        )


if __name__ == "__main__":
    unittest.main()
