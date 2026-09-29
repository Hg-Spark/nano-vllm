import unittest
from unittest.mock import patch

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
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
        original = Sequence([0, 1, 2, 3, 4, 5])
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
        self.assertEqual(scheduler.block_manager.num_cached_prefixes, 1)
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

        newcomer = Sequence([0, 1, 2, 3, 7, 8])
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

    def test_joint_prefix_rejects_snapshot_without_boundary_metadata(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
            num_blocks=8,
            block_size=4,
        )
        seq = Sequence([0, 1, 2, 3, 4, 5])
        scheduler.add(seq)

        scheduled = scheduler.schedule()
        self.assertEqual(scheduled.prefill_tokens, 4)

        with self.assertRaisesRegex(
            RuntimeError,
            "invalid GDN state snapshot",
        ):
            scheduler.postprocess(
                scheduled.prefill_chunks,
                [None],
                True,
                {seq.seq_id: object()},
            )

        self.assertEqual(seq.committed_tokens, 0)
        self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 4)
        self.assertEqual(scheduler.block_manager.num_cached_prefixes, 0)

    def test_joint_prefix_rejects_mismatched_snapshot_boundary(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
            num_blocks=8,
            block_size=4,
        )
        seq = Sequence([0, 1, 2, 3, 4, 5])
        scheduler.add(seq)

        scheduled = scheduler.schedule()

        with self.assertRaisesRegex(
            RuntimeError,
            "does not match scheduled prefix",
        ):
            scheduler.postprocess(
                scheduled.prefill_chunks,
                [None],
                True,
                {
                    seq.seq_id: GDNStateSnapshot(
                        num_tokens=3,
                        layers=(),
                    )
                },
            )

        self.assertEqual(seq.committed_tokens, 0)
        self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 4)
        self.assertEqual(scheduler.block_manager.num_cached_prefixes, 0)

    def test_postprocess_rejects_result_count_mismatch_before_commit(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
        )
        first = Sequence([1, 2])
        second = Sequence([3, 4])
        scheduler.add(first)
        scheduler.add(second)

        scheduled = scheduler.schedule()
        before = [
            chunk.seq.committed_tokens
            for chunk in scheduled.prefill_chunks
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "model result count does not match scheduled sequence count",
        ):
            scheduler.postprocess(
                scheduled.prefill_chunks,
                [10],
                True,
            )

        self.assertEqual(
            [
                chunk.seq.committed_tokens
                for chunk in scheduled.prefill_chunks
            ],
            before,
        )

    def test_postprocess_validates_all_samples_before_commit(self):
        scheduler = make_scheduler(max_num_batched_tokens=4)
        first = Sequence([1, 2])
        second = Sequence([3, 4])
        scheduler.add(first)
        scheduler.add(second)

        scheduled = scheduler.schedule()
        with self.assertRaisesRegex(
            RuntimeError,
            "completed model step did not produce a token",
        ):
            scheduler.postprocess(
                scheduled.prefill_chunks,
                [10, None],
                True,
            )

        self.assertEqual(first.committed_tokens, 0)
        self.assertEqual(second.committed_tokens, 0)

    def test_prefix_publish_failure_keeps_batch_uncommitted(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
            num_blocks=8,
            block_size=4,
        )
        seq = Sequence([0, 1, 2, 3, 4, 5])
        scheduler.add(seq)
        scheduled = scheduler.schedule()
        snapshot = GDNStateSnapshot(num_tokens=4, layers=())

        with patch.object(
            scheduler.block_manager,
            "publish_prefix",
            side_effect=RuntimeError("injected publish failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected publish failure",
            ):
                scheduler.postprocess(
                    scheduled.prefill_chunks,
                    [None],
                    True,
                    {seq.seq_id: snapshot},
                )

        self.assertEqual(seq.committed_tokens, 0)


if __name__ == "__main__":
    unittest.main()
