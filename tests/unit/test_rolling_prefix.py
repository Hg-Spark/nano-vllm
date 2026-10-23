"""CPU-only coverage for one rolling GDN snapshot and deferred publication."""

import unittest
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.schedule import BatchResult
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams


class FakeRunner:
    def __init__(self):
        self.captures = []

    def run(self, chunks, is_prefill):
        return BatchResult(
            [None if is_prefill and chunk.end < len(chunk.seq) else 999 for chunk in chunks]
        )

    def capture_live_states(self, candidates):
        self.captures.extend(pos for _, pos in candidates)
        return {seq.seq_id: (GDNStateSnapshot(pos, ()),) for seq, pos in candidates}


class RollingPrefixTest(unittest.TestCase):
    def _run(self, prompt_len, max_tokens):
        config = SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            max_mixed_prefill_tokens=None,
            eos_token_ids=(),
            kv_cache_block_size=16,
            max_prefix_cache_entries=4,
            max_prefix_cache_snapshot_bytes=None,
        )
        scheduler = Scheduler(config, num_kv_cache_blocks=64)
        runner = FakeRunner()
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = runner
        seq = Sequence(
            list(range(prompt_len)),
            SamplingParams(max_tokens=max_tokens),
        )
        scheduler.add(seq)
        while not scheduler.is_finished():
            batch = scheduler.schedule()
            if batch.decode_chunks:
                engine._run_batch(batch.decode_chunks, False)
            if batch.prefill_chunks:
                engine._run_batch(batch.prefill_chunks, True)
        return scheduler, runner, seq

    def test_unaligned_prompt_captures_last_page_boundary(self):
        scheduler, runner, seq = self._run(65, 1)
        self.assertEqual(runner.captures, [64])
        self.assertEqual(scheduler.cache_manager.prefix_cache._resident_lengths, [64])
        self.assertIsNone(seq.cache_snapshot)

    def test_aligned_prompt_keeps_previous_page_for_sampling(self):
        scheduler, runner, seq = self._run(64, 1)
        self.assertEqual(runner.captures, [48])
        entry = scheduler.cache_manager.find_prefix(Sequence(list(range(64))))
        self.assertIsNotNone(entry)
        self.assertEqual(entry.num_tokens, 48)
        self.assertIsNone(seq.cache_snapshot)

    def test_decode_overwrites_rolling_checkpoint_without_early_publication(self):
        scheduler, runner, seq = self._run(33, 18)
        self.assertEqual(runner.captures, [32, 48])
        cache = scheduler.cache_manager.prefix_cache
        self.assertEqual(cache.published_count, 1)
        self.assertEqual(cache._resident_lengths, [48])
        self.assertIsNone(seq.cache_snapshot)

    def test_admitted_cached_snapshot_is_not_a_writable_staging_slot(self):
        scheduler, _, _ = self._run(65, 1)
        cache_manager = scheduler.cache_manager
        original = cache_manager.find_prefix(Sequence(list(range(65))))
        self.assertIsNotNone(original)
        consumer = Sequence(list(range(65)))
        cache_manager.admit(consumer, original)
        self.assertIsNone(consumer.cache_snapshot)
        self.assertIs(consumer.pending_state_snapshot, original.state_snapshot)
        cache_manager.release(consumer)


if __name__ == "__main__":
    unittest.main()
