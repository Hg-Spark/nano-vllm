import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import make_scheduler


class SchedulerPrefixCacheTest(unittest.TestCase):
    def test_completed_aligned_prompt_publishes_prior_joint_state_before_release(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=128, max_num_seqs=2, num_blocks=32, block_size=16
        )
        original = Sequence(list(range(64)), SamplingParams(max_tokens=1))
        scheduler.add(original)
        first = scheduler.schedule()
        self.assertEqual((first.prefill_chunks[0].start, first.prefill_chunks[0].end), (0, 48))
        snapshot = GDNStateSnapshot(48, ())
        original.cache_snapshot = snapshot
        block = original.block_table[0]
        scheduler.postprocess(first.prefill_chunks, [None], True)

        final = scheduler.schedule()
        self.assertEqual((final.prefill_chunks[0].start, final.prefill_chunks[0].end), (48, 64))
        scheduler.cache_manager.publish_batch([(original, snapshot)])
        scheduler.postprocess(final.prefill_chunks, [99], True)
        self.assertTrue(original.is_finished)
        self.assertEqual(scheduler.cache_manager.block_manager.block_refcounts[block], 1)

        next_turn = Sequence(list(range(64)) + [80], SamplingParams(max_tokens=1))
        scheduler.add(next_turn)
        resumed = scheduler.schedule()
        self.assertEqual(resumed.prefill_chunks[0].start, 48)
        self.assertIs(next_turn.pending_state_snapshot, snapshot)
        self.assertEqual(next_turn.block_table[0], block)

    def test_completed_decode_snapshot_reuses_generated_history(self):
        scheduler = make_scheduler(num_blocks=32, block_size=16, max_num_batched_tokens=128)
        seq = Sequence(list(range(30)), SamplingParams(max_tokens=4))
        scheduler.add(seq)
        first = scheduler.schedule()
        scheduler.postprocess(first.prefill_chunks, [None], True)
        self.assertEqual(len(scheduler.cache_manager.prefix_cache), 0)

        second = scheduler.schedule()
        scheduler.postprocess(second.prefill_chunks, [30], True)
        for token in (31, 32):
            step = scheduler.schedule()
            scheduler.postprocess(step.decode_chunks, [token], False)
        snapshot = GDNStateSnapshot(32, ())
        seq.cache_snapshot = snapshot
        final = scheduler.schedule()
        scheduler.cache_manager.publish_batch([(seq, snapshot)])
        scheduler.postprocess(final.decode_chunks, [33], False)
        self.assertTrue(seq.is_finished)

        continuation = Sequence(list(range(34)) + [99], SamplingParams(max_tokens=1))
        scheduler.add(continuation)
        resumed = scheduler.schedule()
        self.assertEqual(resumed.prefill_chunks[0].start, 32)
        self.assertIs(continuation.pending_state_snapshot, snapshot)

    def test_unaligned_prompt_splits_at_last_full_page(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4096, max_num_seqs=1, num_blocks=300, block_size=16
        )
        seq = Sequence(list(range(4097)), SamplingParams(max_tokens=1))
        scheduler.cache_manager.block_manager.ensure_capacity(seq, 1100)
        scheduler.cache_manager.state_manager.allocate(seq)
        seq.committed_tokens = 1100
        scheduler.prefilling.append(seq)
        chunk = scheduler.schedule().prefill_chunks[0]
        self.assertEqual((chunk.start, chunk.end), (1100, 4096))
        self.assertFalse(scheduler.cache_manager.prefix_cache.can_publish(seq, 4097))
        self.assertTrue(scheduler.cache_manager.prefix_cache.can_publish(seq, 4096))


if __name__ == "__main__":
    unittest.main()
