"""CPU checkpoint lifecycle tests with real GDN tensors and snapshot code.

Only the expensive model forward is replaced; scheduler, KV ownership,
GDN state capture/rolling, publication, and restore use production code.
"""

import unittest
from types import SimpleNamespace

import torch

from nanovllm.engine.cache_runtime import (
    GDNCompletionBuffer,
    GDNRestoreStaging,
    restore_pending_states,
)
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.schedule import BatchResult
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class CPUStateLayer:
    """Small real tensors; each slot's checkpoint is 68 bytes."""

    def __init__(self, slots):
        self.conv_state = torch.zeros((slots, 3, 4), dtype=torch.bfloat16)
        self.recurrent_state = torch.zeros((slots, 11), dtype=torch.float32)

    def slot(self, slot_id):
        return self.conv_state[slot_id], self.recurrent_state[slot_id]

    def restore_state_slot(self, slot_id, snapshot):
        conv, recurrent = self.slot(slot_id)
        conv.copy_(snapshot[0])
        recurrent.copy_(snapshot[1])


class CPUForwardRunner(ModelRunner):
    """Run real snapshot methods with a deterministic CPU replacement for model forward."""

    def __init__(self, slots):
        self.layer = CPUStateLayer(slots)
        self.model = SimpleNamespace(state_cache_modules=lambda: [self.layer])
        self.gdn_completion_buffer = GDNCompletionBuffer(self.model.state_cache_modules())
        self.gdn_restore_staging = GDNRestoreStaging()

    @torch.inference_mode()
    def run(self, chunks, is_prefill):
        if is_prefill:
            restore_pending_states(
                self.model, [chunk.seq for chunk in chunks], self.gdn_restore_staging
            )
        for chunk in chunks:
            conv, recurrent = self.layer.slot(chunk.seq.state_slot)
            conv.fill_(chunk.end)
            recurrent.fill_(chunk.end + 1)
        return BatchResult(
            [None if is_prefill and chunk.end < len(chunk.seq) else 999 for chunk in chunks]
        )


class RollingPrefixTest(unittest.TestCase):
    @staticmethod
    def _engine(entries=4, budget=None):
        config = SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            max_mixed_prefill_tokens=None,
            eos_token_ids=(),
            kv_cache_block_size=16,
            max_prefix_cache_entries=entries,
            max_prefix_cache_snapshot_bytes=budget,
        )
        runner = CPUForwardRunner(config.max_num_seqs)
        snapshot_bytes = (
            runner.layer.conv_state[0].numel() * runner.layer.conv_state.element_size()
            + runner.layer.recurrent_state[0].numel() * runner.layer.recurrent_state.element_size()
        )
        scheduler = Scheduler(config, num_kv_cache_blocks=64, snapshot_nbytes=snapshot_bytes)
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

    def test_prompt_checkpoint_is_immutable_and_restorable(self):
        for prompt_len, boundary in ((64, 48), (65, 64)):
            with self.subTest(prompt_len=prompt_len):
                engine, scheduler, runner = self._engine()
                seq = Sequence(list(range(prompt_len)), SamplingParams(max_tokens=1))
                scheduler.add(seq)
                engine._run_batch(scheduler.schedule().prefill_chunks, True)
                snapshot = seq.prompt_snapshot
                self.assertEqual(snapshot.num_tokens, boundary)
                self.assertTrue(torch.all(snapshot.layers[0][0] == boundary))
                self.assertTrue(torch.all(snapshot.layers[0][1] == boundary + 1))
                self.assertEqual(snapshot.layers[0][0].device.type, "cpu")
                self.assertEqual(len(scheduler.cache_manager.prefix_cache), 0)

                engine._run_batch(scheduler.schedule().prefill_chunks, True)
                self.assertTrue(seq.is_finished)
                self.assertIsNone(seq.prompt_snapshot)
                # The next forward has modified the live slot, not the CPU checkpoint.
                self.assertTrue(torch.all(snapshot.layers[0][0] == boundary))
                continuation = Sequence(list(range(prompt_len)) + [80])
                entry = scheduler.cache_manager.find_prefix(continuation)
                self.assertEqual(entry.num_tokens, boundary)
                self.assertIs(entry.state_snapshot, snapshot)
                scheduler.cache_manager.admit(continuation, entry)
                self.assertIs(continuation.pending_state_snapshot, snapshot)
                self.assertEqual(continuation.block_table[0], entry.block_ids[0])
                restore_pending_states(runner.model, [continuation], runner.gdn_restore_staging)
                conv, recurrent = runner.layer.slot(continuation.state_slot)
                self.assertTrue(torch.all(conv == boundary))
                self.assertTrue(torch.all(recurrent == boundary + 1))
                scheduler.cache_manager.release(continuation)

    def test_prompt_checkpoint_survives_completion_eviction(self):
        engine, scheduler, _ = self._engine(entries=2)
        prompt = list(range(64))
        cache = self._complete(engine, scheduler, Sequence(prompt, SamplingParams(max_tokens=2)))
        self.assertEqual(cache._resident_lengths, [48, 64])
        cache = self._complete(engine, scheduler, Sequence(prompt, SamplingParams(max_tokens=20)))
        self.assertEqual(cache._resident_lengths, [48, 80])
        self.assertEqual(scheduler.cache_manager.find_prefix(Sequence(prompt)).num_tokens, 48)

    def test_decode_final_boundary_and_unaligned_end(self):
        for prompt_len, outputs, expected in (
            (15, 2, [16]),
            (15, 3, [16]),
            (31, 2, [16, 32]),
            (31, 3, [16, 32]),
        ):
            with self.subTest(prompt_len=prompt_len, outputs=outputs):
                engine, scheduler, _ = self._engine()
                cache = self._complete(
                    engine,
                    scheduler,
                    Sequence(list(range(prompt_len)), SamplingParams(max_tokens=outputs)),
                )
                self.assertEqual(cache._resident_lengths, expected)
                completion = cache.find(
                    Sequence(list(range(prompt_len)) + [999] * outputs),
                    max_tokens=expected[-1],
                )
                self.assertEqual(completion.num_tokens, expected[-1])
                self.assertTrue(torch.all(completion.state_snapshot.layers[0][0] == expected[-1]))

    def test_completion_rolling_exports_only_final_aligned_state(self):
        engine, scheduler, runner = self._engine()
        seq = Sequence(list(range(33)), SamplingParams(max_tokens=33))
        cache = self._complete(engine, scheduler, seq)

        # Decode crossed 48 and 64, but only the final aligned state is
        # published. Per-boundary buffer replacement is tested with real
        # tensors in test_cache_runtime.py.
        self.assertEqual(cache._resident_lengths, [32, 64])
        self.assertEqual(runner.gdn_completion_buffer.take([seq]), {})
        completion = cache.find(Sequence(list(range(33)) + [999] * 33), max_tokens=64)
        self.assertEqual(completion.num_tokens, 64)
        self.assertTrue(torch.all(completion.state_snapshot.layers[0][0] == 64))
        self.assertTrue(torch.all(completion.state_snapshot.layers[0][1] == 65))
        # The live slot advanced to 65; the published checkpoint remains at 64.
        self.assertTrue(torch.all(runner.layer.conv_state[0] == 65))
        self.assertEqual(cache.peak_staging_snapshot_bytes, 136)
        self.assertEqual(cache.staging_snapshot_bytes, 0)
        self.assertEqual(cache.snapshot_bytes, 136)

    def test_capacity_one_replaces_active_prefix_on_completion(self):
        engine, scheduler, _ = self._engine(entries=1)
        prompt = list(range(65))
        cache = self._complete(engine, scheduler, Sequence(prompt, SamplingParams(max_tokens=1)))
        self.assertEqual(cache._resident_lengths, [64])
        cache = self._complete(engine, scheduler, Sequence(prompt, SamplingParams(max_tokens=20)))
        self.assertEqual(cache._resident_lengths, [80])
        self.assertEqual(cache.published_count, 2)

    def test_eos_on_final_aligned_decode_keeps_checkpoint(self):
        engine, scheduler, _ = self._engine()
        seq = Sequence(list(range(15)), SamplingParams(max_tokens=10))
        scheduler.add(seq)
        engine._run_batch(scheduler.schedule().prefill_chunks, True)
        scheduler.eos_token_ids.add(999)
        engine._run_batch(scheduler.schedule().decode_chunks, False)
        self.assertTrue(seq.is_finished)
        self.assertEqual(scheduler.cache_manager.prefix_cache._resident_lengths, [16])

    def test_separate_turns_keep_both_checkpoint_generations(self):
        engine, scheduler, _ = self._engine(entries=8)
        first = Sequence(list(range(64)), SamplingParams(max_tokens=2))
        cache = self._complete(engine, scheduler, first)
        self.assertEqual(cache._resident_lengths, [48, 64])
        second_prompt = list(range(64)) + [999, 999] + list(range(1000, 1031))
        cache = self._complete(
            engine,
            scheduler,
            Sequence(second_prompt, SamplingParams(max_tokens=17)),
        )
        self.assertEqual(cache._resident_lengths, [48, 64, 96, 112])
        self.assertEqual(cache.published_count, 4)
        self.assertIsNotNone(cache.find(Sequence(list(range(64))), 63))
        self.assertIsNotNone(cache.find(Sequence(second_prompt), 96))


if __name__ == "__main__":
    unittest.main()
