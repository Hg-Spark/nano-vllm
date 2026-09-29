import unittest

from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from tests.scheduler_helpers import (
    make_running_sequence,
    make_scheduler,
    scheduled_sequences,
)


class SchedulerCoreTest(unittest.TestCase):

    def test_decode_uses_budget_before_chunked_prefill(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=4,
        )
        decode_seq = make_running_sequence(
            scheduler,
            [1, 2],
        )
        prefill_seq = Sequence(list(range(10)))
        scheduler.add(prefill_seq)

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.decode_chunks),
            [decode_seq],
        )
        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            [prefill_seq],
        )
        self.assertEqual(scheduled.decode_tokens, 1)
        self.assertEqual(scheduled.prefill_tokens, 3)
        self.assertEqual(
            scheduled.prefill_chunks[0].num_tokens,
            3,
        )

    def test_sequence_limit_is_shared(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=1,
        )
        decode_seq = make_running_sequence(
            scheduler,
            [1, 2],
        )
        scheduler.add(Sequence([3, 4, 5]))

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.decode_chunks),
            [decode_seq],
        )
        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            [],
        )

    def test_partial_prefill_keeps_state_slot(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=3,
            max_num_seqs=2,
        )
        seq = Sequence(list(range(8)))
        scheduler.add(seq)

        first = scheduler.schedule()
        first_slot = seq.state_slot
        self.assertEqual(first.prefill_tokens, 3)
        self.assertGreaterEqual(first_slot, 0)

        scheduler.postprocess(
            first.prefill_chunks,
            [None],
            True,
        )
        self.assertEqual(seq.committed_tokens, 3)
        self.assertEqual(
            scheduler.state_manager.slot_owners[first_slot],
            seq.seq_id,
        )

        second = scheduler.schedule()

        self.assertEqual(second.prefill_tokens, 3)
        self.assertEqual(seq.state_slot, first_slot)

    def test_chunked_prefill_allocates_only_scheduled_kv_range(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=3,
            max_num_seqs=2,
            num_blocks=4,
            block_size=4,
        )
        seq = Sequence(list(range(10)))
        scheduler.add(seq)

        first = scheduler.schedule()

        self.assertEqual(first.prefill_tokens, 3)
        self.assertEqual(len(seq.block_table), 1)

        scheduler.postprocess(
            first.prefill_chunks,
            [None],
            True,
        )
        second = scheduler.schedule()

        self.assertEqual(second.prefill_tokens, 3)
        self.assertEqual(len(seq.block_table), 2)

    def test_any_configured_eos_finishes_request(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
        )
        seq = make_running_sequence(
            scheduler,
            [1, 2],
        )
        scheduler.postprocess(
            (ScheduledChunk(seq, len(seq) - 1, len(seq)),),
            [100],
            False,
        )

        self.assertTrue(seq.is_finished)
        self.assertEqual(seq.state_slot, -1)
        self.assertFalse(seq.block_table)

    def test_variable_length_prefills_share_one_batch(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=7,
            max_num_seqs=3,
        )
        seqs = [
            Sequence([1, 2]),
            Sequence([3, 4, 5]),
            Sequence([6, 7]),
        ]
        for seq in seqs:
            scheduler.add(seq)

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            seqs,
        )
        self.assertEqual(
            [chunk.num_tokens for chunk in scheduled.prefill_chunks],
            [2, 3, 2],
        )
        self.assertEqual(scheduled.prefill_tokens, 7)
        self.assertEqual(
            len({seq.state_slot for seq in seqs}),
            3,
        )

    def test_continuous_batching_reuses_released_state_slot(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
        )
        finished = make_running_sequence(
            scheduler,
            [1, 2],
        )
        survivor = make_running_sequence(
            scheduler,
            [3, 4],
        )
        released_slot = finished.state_slot
        survivor_slot = survivor.state_slot

        scheduler.postprocess(
            (ScheduledChunk(finished, len(finished) - 1, len(finished)),),
            [99],
            False,
        )

        newcomer = Sequence([5, 6, 7])
        scheduler.add(newcomer)
        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.decode_chunks),
            [survivor],
        )
        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            [newcomer],
        )
        self.assertEqual(newcomer.state_slot, released_slot)
        self.assertEqual(survivor.state_slot, survivor_slot)
        self.assertEqual(
            scheduler.state_manager.slot_owners[released_slot],
            newcomer.seq_id,
        )
        self.assertEqual(
            scheduler.state_manager.slot_owners[survivor_slot],
            survivor.seq_id,
        )

    def test_chunked_prefill_batches_other_waiters_work_conservingly(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=3,
            max_num_seqs=2,
        )
        long_seq = Sequence(list(range(8)))
        short_seq = Sequence([20])
        scheduler.add(long_seq)
        scheduler.add(short_seq)

        first = scheduler.schedule()
        long_slot = long_seq.state_slot
        scheduler.postprocess(
            first.prefill_chunks,
            [None],
            True,
        )

        second = scheduler.schedule()
        self.assertEqual(
            scheduled_sequences(second.prefill_chunks),
            [short_seq, long_seq],
        )
        self.assertEqual(
            [chunk.num_tokens for chunk in second.prefill_chunks],
            [1, 2],
        )
        scheduler.postprocess(
            second.prefill_chunks,
            [99, None],
            True,
        )

        self.assertTrue(short_seq.is_finished)
        self.assertEqual(long_seq.committed_tokens, 5)
        self.assertEqual(long_seq.state_slot, long_slot)
        self.assertEqual(
            scheduler.state_manager.slot_owners[long_slot],
            long_seq.seq_id,
        )

        third = scheduler.schedule()
        self.assertEqual(
            scheduled_sequences(third.prefill_chunks),
            [long_seq],
        )
        self.assertEqual(third.prefill_chunks[0].num_tokens, 3)

    def test_failed_prefill_step_releases_uncertain_hybrid_state(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=3,
            max_num_seqs=2,
        )
        seq = Sequence(list(range(8)))
        scheduler.add(seq)

        scheduled = scheduler.schedule()
        slot = seq.state_slot
        blocks = set(seq.block_table)

        scheduler.recover_failed_step(
            scheduled.prefill_chunks
        )

        self.assertEqual(list(scheduler.waiting), [seq])
        self.assertNotIn(seq, scheduler.running)
        self.assertEqual(seq.committed_tokens, 0)
        self.assertEqual(seq.state_slot, -1)
        self.assertFalse(seq.block_table)
        self.assertIsNone(
            scheduler.state_manager.slot_owners[slot]
        )
        self.assertTrue(all(
            scheduler.block_manager.block_refcounts[block_id] == 0
            for block_id in blocks
        ))

    def test_decode_budget_rotates_running_requests(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=3,
        )
        first = make_running_sequence(scheduler, [1, 2])
        second = make_running_sequence(scheduler, [3, 4])
        third = make_running_sequence(scheduler, [5, 6])

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.decode_chunks),
            [first, second],
        )
        scheduler.postprocess(
            scheduled.decode_chunks,
            [10, 11],
            False,
        )

        next_step = scheduler.schedule()

        self.assertEqual(next_step.decode_chunks[0].seq, third)
        self.assertIn(first, scheduled_sequences(next_step.decode_chunks))

    def test_invalid_decode_boundary_keeps_request_in_queue(self):
        scheduler = make_scheduler()
        seq = make_running_sequence(scheduler, [1, 2])
        seq.committed_tokens = len(seq)

        with self.assertRaisesRegex(RuntimeError, "decode prefix mismatch"):
            scheduler.schedule()

        self.assertEqual(list(scheduler.running), [seq])

if __name__ == "__main__":
    unittest.main()
