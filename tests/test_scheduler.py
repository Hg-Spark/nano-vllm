import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(
    max_num_batched_tokens=4,
    max_num_seqs=4,
    num_blocks=16,
):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos_token_ids=(99, 100),
        kvcache_block_size=256,
        num_kvcache_blocks=num_blocks,
        max_num_state_slots=max_num_seqs,
    )
    Sequence.block_size = config.kvcache_block_size
    return Scheduler(config)


def make_running_sequence(scheduler, token_ids):
    seq = Sequence(token_ids)
    scheduler.block_manager.allocate(seq)
    scheduler.state_manager.allocate(seq)
    seq.num_cached_tokens = len(seq)
    seq.num_state_tokens = len(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return seq


class SchedulerTest(unittest.TestCase):

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
            scheduled.decode_seqs,
            [decode_seq],
        )
        self.assertEqual(
            scheduled.prefill_seqs,
            [prefill_seq],
        )
        self.assertEqual(scheduled.decode_tokens, 1)
        self.assertEqual(scheduled.prefill_tokens, 3)
        self.assertEqual(
            prefill_seq.num_scheduled_tokens,
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
            scheduled.decode_seqs,
            [decode_seq],
        )
        self.assertEqual(
            scheduled.prefill_seqs,
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
            first.prefill_seqs,
            [None],
            True,
        )
        self.assertEqual(seq.num_cached_tokens, 3)
        self.assertEqual(seq.num_state_tokens, 3)
        self.assertEqual(
            scheduler.state_manager.owner_of(first_slot),
            seq.seq_id,
        )

        second = scheduler.schedule()

        self.assertEqual(second.prefill_tokens, 3)
        self.assertEqual(seq.state_slot, first_slot)

    def test_preempt_releases_kv_and_state(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
        )
        seq = make_running_sequence(
            scheduler,
            [1, 2],
        )
        slot = seq.state_slot
        blocks = set(seq.block_table)

        scheduler.preempt(seq)

        self.assertEqual(seq.state_slot, -1)
        self.assertEqual(seq.num_state_tokens, 0)
        self.assertFalse(seq.block_table)
        self.assertNotIn(
            slot,
            scheduler.state_manager.used_slot_ids,
        )
        self.assertTrue(
            blocks.isdisjoint(
                scheduler.block_manager.used_block_ids
            )
        )

    def test_any_configured_eos_finishes_request(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
        )
        seq = make_running_sequence(
            scheduler,
            [1, 2],
        )
        seq.num_scheduled_tokens = 1

        scheduler.postprocess(
            [seq],
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

        self.assertEqual(scheduled.prefill_seqs, seqs)
        self.assertEqual(
            [seq.num_scheduled_tokens for seq in seqs],
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

        finished.num_scheduled_tokens = 1
        scheduler.postprocess(
            [finished],
            [99],
            False,
        )

        newcomer = Sequence([5, 6, 7])
        scheduler.add(newcomer)
        scheduled = scheduler.schedule()

        self.assertEqual(scheduled.decode_seqs, [survivor])
        self.assertEqual(scheduled.prefill_seqs, [newcomer])
        self.assertEqual(newcomer.state_slot, released_slot)
        self.assertEqual(survivor.state_slot, survivor_slot)
        self.assertEqual(
            scheduler.state_manager.owner_of(released_slot),
            newcomer.seq_id,
        )
        self.assertEqual(
            scheduler.state_manager.owner_of(survivor_slot),
            survivor.seq_id,
        )


    def test_chunked_prefill_can_join_new_request_on_final_chunk(self):
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
            first.prefill_seqs,
            [None],
            True,
        )

        second = scheduler.schedule()
        scheduler.postprocess(
            second.prefill_seqs,
            [None],
            True,
        )

        third = scheduler.schedule()

        self.assertEqual(
            third.prefill_seqs,
            [long_seq, short_seq],
        )
        self.assertEqual(
            [seq.num_scheduled_tokens for seq in third.prefill_seqs],
            [2, 1],
        )
        self.assertEqual(long_seq.num_state_tokens, 6)
        self.assertEqual(long_seq.state_slot, long_slot)
        self.assertNotEqual(short_seq.state_slot, long_slot)
        self.assertEqual(
            scheduler.state_manager.owner_of(long_slot),
            long_seq.seq_id,
        )

    def test_scheduler_rejects_kv_state_progress_divergence(self):
        scheduler = make_scheduler()
        seq = make_running_sequence(
            scheduler,
            [1, 2],
        )
        seq.num_state_tokens -= 1

        with self.assertRaisesRegex(
            RuntimeError,
            "KV/state prefix mismatch",
        ):
            scheduler.schedule()

if __name__ == "__main__":
    unittest.main()
