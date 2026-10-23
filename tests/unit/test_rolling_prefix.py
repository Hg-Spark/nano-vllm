"""CPU-only regressions for prompt and rolling completion checkpoint lifetimes."""

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
        return {seq.seq_id: GDNStateSnapshot(pos, ()) for seq, pos in candidates}


class RollingPrefixTest(unittest.TestCase):
    @staticmethod
    def _engine(entries=4, snapshot_bytes=0, budget=None):
        config = SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            max_mixed_prefill_tokens=None,
            eos_token_ids=(),
            kv_cache_block_size=16,
            max_prefix_cache_entries=entries,
            max_prefix_cache_snapshot_bytes=budget,
        )
        scheduler = Scheduler(config, num_kv_cache_blocks=64, snapshot_nbytes=snapshot_bytes)
        runner = FakeRunner()
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = runner
        return engine, scheduler, runner

    @staticmethod
    def _complete(engine, scheduler, seq):
        scheduler.add(seq)
        for _ in range(256):
            if scheduler.is_finished():
                break
            batch = scheduler.schedule()
            if batch.decode_chunks:
                engine._run_batch(batch.decode_chunks, False)
            if batch.prefill_chunks:
                engine._run_batch(batch.prefill_chunks, True)
        else:
            raise AssertionError("scheduler did not finish")
        return scheduler.cache_manager.prefix_cache

    def test_unaligned_prompt_captures_last_page_boundary(self):
        engine, scheduler, runner = self._engine()
        seq = Sequence(list(range(65)), SamplingParams(max_tokens=1))
        cache = self._complete(engine, scheduler, seq)
        self.assertEqual(runner.captures, [64])
        self.assertEqual(cache._resident_lengths, [64])
        self.assertIsNone(seq.prompt_snapshot)
        self.assertIsNone(seq.completion_snapshot)

    def test_aligned_prompt_keeps_previous_page_for_sampling(self):
        engine, scheduler, runner = self._engine()
        seq = Sequence(list(range(64)), SamplingParams(max_tokens=1))
        cache = self._complete(engine, scheduler, seq)
        self.assertEqual(runner.captures, [48])
        entry = scheduler.cache_manager.find_prefix(Sequence(list(range(64))))
        self.assertEqual(entry.num_tokens, 48)
        self.assertEqual(cache._resident_lengths, [48])

    def test_prompt_checkpoint_survives_followup_completion_eviction(self):
        engine, scheduler, runner = self._engine(entries=2)
        prompt = list(range(64))
        cache = self._complete(engine, scheduler, Sequence(prompt, SamplingParams(max_tokens=2)))
        self.assertEqual(runner.captures, [48, 64])
        self.assertEqual(cache._resident_lengths, [48, 64])
        self.assertEqual(
            [entry.kind for entry in cache._entries.values()], ["prompt", "conversation"]
        )

        # Reuse S48, extend the answer and evict S64, not the only prompt
        # checkpoint eligible for the next identical 64-token request.
        cache = self._complete(engine, scheduler, Sequence(prompt, SamplingParams(max_tokens=20)))
        self.assertEqual(cache._resident_lengths, [48, 80])
        self.assertEqual(
            [entry.kind for entry in cache._entries.values()], ["prompt", "conversation"]
        )
        self.assertEqual(scheduler.cache_manager.find_prefix(Sequence(prompt)).num_tokens, 48)

    def test_final_decode_aligned_boundary_is_captured(self):
        for prompt_len, outputs, expected in (
            (15, 2, [16]),
            (15, 3, [16]),
            (31, 2, [16, 32]),
            (31, 3, [16, 32]),
        ):
            with self.subTest(prompt_len=prompt_len, outputs=outputs):
                engine, scheduler, runner = self._engine()
                seq = Sequence(list(range(prompt_len)), SamplingParams(max_tokens=outputs))
                cache = self._complete(engine, scheduler, seq)
                self.assertEqual(runner.captures, expected)
                self.assertEqual(cache._resident_lengths, expected)

    def test_decode_updates_one_completion_staging_slot(self):
        engine, scheduler, runner = self._engine(snapshot_bytes=68)
        seq = Sequence(list(range(33)), SamplingParams(max_tokens=18))
        cache = self._complete(engine, scheduler, seq)
        self.assertEqual(runner.captures, [32, 48])
        self.assertEqual(cache._resident_lengths, [32, 48])
        self.assertEqual(cache.peak_staging_snapshot_bytes, 136)
        self.assertEqual(cache.staging_snapshot_bytes, 0)
        self.assertEqual(cache.snapshot_bytes, 136)
        self.assertEqual(cache.peak_total_snapshot_bytes, 136)

    def test_capacity_one_can_replace_its_own_active_prefix(self):
        engine, scheduler, runner = self._engine(entries=1)
        prompt = list(range(65))
        source = Sequence(prompt, SamplingParams(max_tokens=1))
        cache = self._complete(engine, scheduler, source)
        self.assertEqual(cache._resident_lengths, [64])
        continuation = Sequence(prompt, SamplingParams(max_tokens=20))
        cache = self._complete(engine, scheduler, continuation)
        self.assertIn(80, runner.captures)
        self.assertEqual(cache._resident_lengths, [80])
        self.assertEqual(cache.published_count, 2)
        self.assertFalse(cache._active_reuses)

    def test_one_global_entry_does_not_limit_two_private_staging_slots(self):
        engine, scheduler, _ = self._engine(entries=1, snapshot_bytes=68, budget=68)
        seq = Sequence(list(range(64)), SamplingParams(max_tokens=2))
        cache = self._complete(engine, scheduler, seq)
        self.assertEqual(cache.snapshot_bytes, 68)
        self.assertEqual(cache._resident_lengths, [64])
        self.assertEqual(cache.peak_staging_snapshot_bytes, 136)
        self.assertEqual(cache.staging_snapshot_bytes, 0)

    def test_eos_on_final_aligned_decode_keeps_checkpoint(self):
        engine, scheduler, _ = self._engine()
        seq = Sequence(list(range(15)), SamplingParams(max_tokens=10))
        scheduler.add(seq)
        engine._run_batch(scheduler.schedule().prefill_chunks, True)
        # Only the Decode sample terminates this request.
        scheduler.eos_token_ids.add(999)
        engine._run_batch(scheduler.schedule().decode_chunks, False)
        self.assertTrue(seq.is_finished)
        self.assertEqual(scheduler.cache_manager.prefix_cache._resident_lengths, [16])

    def test_separate_turns_keep_independent_global_checkpoints(self):
        engine, scheduler, _ = self._engine(entries=8)
        first = Sequence(list(range(64)), SamplingParams(max_tokens=2))
        cache = self._complete(engine, scheduler, first)
        self.assertEqual(cache._resident_lengths, [48, 64])
        second_prompt = list(range(64)) + [999, 999] + list(range(1000, 1031))
        self.assertEqual(len(second_prompt), 97)
        second = Sequence(second_prompt, SamplingParams(max_tokens=17))
        cache = self._complete(engine, scheduler, second)
        self.assertEqual(cache._resident_lengths, [48, 64, 96, 112])
        self.assertEqual(cache.published_count, 4)
        self.assertIsNotNone(cache.find(Sequence(list(range(64))), 63))
        self.assertIsNotNone(cache.find(Sequence(second_prompt), 96))

    def test_cache_hit_does_not_mutate_published_snapshot(self):
        engine, scheduler, _ = self._engine()
        prompt = list(range(65))
        self._complete(engine, scheduler, Sequence(prompt, SamplingParams(max_tokens=1)))
        manager = scheduler.cache_manager
        original = manager.find_prefix(Sequence(prompt))
        self.assertIsNotNone(original)
        consumer = Sequence(prompt)
        manager.admit(consumer, original)
        self.assertIsNone(consumer.prompt_snapshot)
        self.assertIsNone(consumer.completion_snapshot)
        self.assertIs(consumer.pending_state_snapshot, original.state_snapshot)
        manager.release(consumer)


if __name__ == "__main__":
    unittest.main()
