import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import (
    make_scheduler,
    scheduled_sequences,
)


class SchedulerPrefixCacheTest(unittest.TestCase):
    def test_joint_prefix_hit_restores_kv_and_gdn_boundary_together(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=64, max_num_seqs=2, num_blocks=64, block_size=4
        )
        original = Sequence(list(range(65)), SamplingParams(max_tokens=1))
        scheduler.add(original)

        first = scheduler.schedule()
        self.assertEqual(first.prefill_tokens, 64)
        self.assertEqual(first.prefill_chunks[0].checkpoint_positions, (64,))
        cached_block = original.block_table[0]
        snapshot = GDNStateSnapshot(num_tokens=64, layers=())
        scheduler.postprocess(first.prefill_chunks, [None], True, {original.seq_id: (snapshot,)})
        self.assertEqual(len(scheduler.block_manager.prefix_cache), 1)
        self.assertEqual(scheduler.block_manager.block_refcounts[cached_block], 2)

        final = scheduler.schedule()
        scheduler.postprocess(final.prefill_chunks, [99], True)
        self.assertTrue(original.is_finished)
        self.assertEqual(scheduler.block_manager.block_refcounts[cached_block], 1)

        newcomer = Sequence(list(range(64)) + [80], SamplingParams(max_tokens=1))
        scheduler.add(newcomer)
        resumed = scheduler.schedule()
        self.assertEqual(scheduled_sequences(resumed.prefill_chunks), [newcomer])
        self.assertEqual(newcomer.committed_tokens, 64)
        self.assertIs(newcomer.pending_state_snapshot, snapshot)
        self.assertEqual(newcomer.block_table[0], cached_block)
        self.assertEqual(resumed.prefill_chunks[0].num_tokens, 1)
        self.assertEqual(scheduler.block_manager.block_refcounts[cached_block], 2)

    def test_aligned_prompt_restores_reusable_partial_prefix(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=129, max_num_seqs=2, num_blocks=100, block_size=4
        )
        tokens = list(range(128))
        original = Sequence(tokens, SamplingParams(max_tokens=1))
        scheduler.add(original)

        first = scheduler.schedule()
        self.assertEqual(first.prefill_chunks[0].checkpoint_positions, (64, 128))
        snapshots = (GDNStateSnapshot(64, ()), GDNStateSnapshot(128, ()))
        scheduler.postprocess(first.prefill_chunks, [99], True, {original.seq_id: snapshots})

        newcomer = Sequence(tokens + [129], SamplingParams(max_tokens=1))
        scheduler.add(newcomer)
        scheduler.schedule()
        self.assertEqual(newcomer.committed_tokens, 128)
        self.assertIs(newcomer.pending_state_snapshot, snapshots[1])

    def test_identical_prefills_capture_only_one_snapshot(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=130, max_num_seqs=2, num_blocks=64, block_size=4
        )
        for _ in range(2):
            scheduler.add(Sequence(list(range(65)), SamplingParams(max_tokens=1)))
        chunks = scheduler.schedule().prefill_chunks
        self.assertEqual(
            [(chunk.end, chunk.checkpoint_positions) for chunk in chunks],
            [(65, (64,)), (65, ())],
        )

    def test_unaligned_prefill_does_not_force_tiny_checkpoint_chunk(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4096,
            max_num_seqs=1,
            num_blocks=300,
            block_size=16,
        )
        seq = Sequence(list(range(4097)), SamplingParams(max_tokens=1))
        scheduler.block_manager.ensure_capacity(seq, 1100)
        scheduler.state_manager.allocate(seq)
        seq.committed_tokens = 1100
        scheduler.prefilling.append(seq)

        first = scheduler.schedule().prefill_chunks
        self.assertEqual(
            (first[0].start, first[0].end, first[0].checkpoint_positions),
            (1100, 4097, ()),
        )
        self.assertEqual(first[0].checkpoint_interval, 0)


if __name__ == "__main__":
    unittest.main()
