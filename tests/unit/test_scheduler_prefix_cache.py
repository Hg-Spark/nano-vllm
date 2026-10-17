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

    def test_joint_prefix_rejects_invalid_snapshot_before_commit(self):
        for snapshot, error in (
            (object(), "invalid GDN state snapshot"),
            (GDNStateSnapshot(num_tokens=3, layers=()), "does not match KV prefix boundary"),
        ):
            with self.subTest(error=error):
                scheduler = make_scheduler(
                    max_num_batched_tokens=4,
                    max_num_seqs=2,
                    num_blocks=8,
                    block_size=4,
                )
                seq = Sequence([0, 1, 2, 3, 4, 5], SamplingParams(max_tokens=1))
                scheduler.add(seq)
                scheduled = scheduler.schedule()
                self.assertEqual(scheduled.prefill_tokens, 4)

                with self.assertRaisesRegex(RuntimeError, error):
                    scheduler.postprocess(
                        scheduled.prefill_chunks,
                        [None],
                        True,
                        {seq.seq_id: snapshot},
                    )

                self.assertEqual(seq.committed_tokens, 0)
                self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 4)
                self.assertEqual(len(scheduler.block_manager.prefix_cache), 0)


if __name__ == "__main__":
    unittest.main()
