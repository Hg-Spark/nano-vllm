import unittest
from types import SimpleNamespace

from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import GDNStateSnapshot


def make_scheduler(
    max_num_batched_tokens=4,
    max_num_seqs=4,
    num_blocks=16,
    block_size=256,
):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos_token_ids=(99, 100),
        kvcache_block_size=block_size,
        max_prefix_cache_entries=16,
    )
    return Scheduler(config, num_blocks)


def make_running_sequence(scheduler, token_ids):
    seq = Sequence(token_ids)
    scheduler.block_manager.ensure_capacity(seq, len(seq))
    scheduler.state_manager.allocate(seq)
    seq.committed_tokens = len(seq) - 1
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return seq


def scheduled_sequences(chunks):
    return [chunk.seq for chunk in chunks]


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
            scheduler.state_manager.owner_of(first_slot),
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

    def test_long_prompt_admits_without_full_prompt_reservation(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
            num_blocks=4,
            block_size=4,
        )
        running = make_running_sequence(
            scheduler,
            [1, 2, 3, 4, 5],
        )
        long_prompt = Sequence(list(range(10)))
        scheduler.add(long_prompt)

        # The running request owns two of four blocks. The long prompt needs
        # three blocks in total, but its first scheduled range needs only one.
        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.decode_chunks),
            [running],
        )
        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            [long_prompt],
        )
        self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 3)
        self.assertEqual(len(long_prompt.block_table), 1)

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

        self.assertNotIn(seq, scheduler.running)
        self.assertEqual(list(scheduler.waiting), [seq])
        self.assertEqual(seq.state_slot, -1)
        self.assertEqual(seq.committed_tokens, 0)
        self.assertFalse(seq.block_table)
        self.assertIsNone(
            scheduler.state_manager.owner_of(slot)
        )
        self.assertTrue(all(
            scheduler.block_manager.block_refcounts[block_id] == 0
            for block_id in blocks
        ))

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
            scheduler.state_manager.owner_of(released_slot),
            newcomer.seq_id,
        )
        self.assertEqual(
            scheduler.state_manager.owner_of(survivor_slot),
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
            scheduler.state_manager.owner_of(long_slot),
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
            scheduler.state_manager.owner_of(slot)
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
        self.assertEqual(len(scheduler.prefix_runtime), 1)
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
        self.assertEqual(len(scheduler.prefix_runtime), 0)

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
        self.assertEqual(len(scheduler.prefix_runtime), 0)

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

    def test_prefill_reservation_rolls_back_kv_and_state_together(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
            num_blocks=4,
            block_size=4,
        )
        seq = Sequence([1, 2, 3, 4])
        scheduler.add(seq)

        original_ensure_capacity = (
            scheduler.block_manager.ensure_capacity
        )

        def fail_after_kv_growth(target_seq, target_tokens):
            original_ensure_capacity(target_seq, target_tokens)
            raise RuntimeError("injected reservation failure")

        scheduler.block_manager.ensure_capacity = fail_after_kv_growth

        with self.assertRaisesRegex(
            RuntimeError,
            "injected reservation failure",
        ):
            scheduler.schedule()

        self.assertEqual(list(scheduler.waiting), [seq])
        self.assertEqual(seq.state_slot, -1)
        self.assertFalse(seq.block_table)
        self.assertEqual(seq.committed_tokens, 0)
        self.assertEqual(
            sum(scheduler.block_manager.block_refcounts),
            0,
        )
        self.assertEqual(
            len(scheduler.state_manager.free_slot_ids),
            2,
        )

    def test_blocked_waiter_does_not_block_schedulable_waiter(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=2,
            num_blocks=2,
            block_size=4,
        )

        blocked = Sequence([1, 2, 3, 4, 5, 6])
        scheduler.block_manager.ensure_capacity(blocked, 4)
        scheduler.state_manager.allocate(blocked)
        blocked.committed_tokens = 4
        scheduler.add(blocked)

        schedulable = Sequence([7, 8, 9])
        scheduler.block_manager.ensure_capacity(schedulable, 2)
        scheduler.state_manager.allocate(schedulable)
        schedulable.committed_tokens = 2
        scheduler.add(schedulable)

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            [schedulable],
        )
        self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 1)
        self.assertEqual(list(scheduler.waiting), [blocked])
        self.assertIn(schedulable, scheduler.running)

    def test_decode_preempts_resource_holding_partial_prefill(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=2,
            num_blocks=2,
            block_size=4,
        )

        decode_seq = Sequence([1, 2, 3, 4, 5])
        scheduler.block_manager.ensure_capacity(decode_seq, 4)
        scheduler.state_manager.allocate(decode_seq)
        decode_seq.committed_tokens = 4
        decode_seq.status = SequenceStatus.RUNNING
        scheduler.running.append(decode_seq)

        partial = Sequence([6, 7, 8, 9, 10, 11])
        scheduler.block_manager.ensure_capacity(partial, 4)
        scheduler.state_manager.allocate(partial)
        partial.committed_tokens = 4
        scheduler.add(partial)

        self.assertFalse(scheduler.block_manager.free_block_ids)

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.decode_chunks),
            [decode_seq],
        )
        self.assertEqual(partial.committed_tokens, 0)
        self.assertEqual(partial.state_slot, -1)
        self.assertFalse(partial.block_table)
        self.assertEqual(list(scheduler.waiting), [partial])
        self.assertEqual(len(decode_seq.block_table), 2)

    def test_waiting_only_deadlock_preempts_one_holder(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=2,
            num_blocks=2,
            block_size=4,
        )

        first = Sequence([1, 2, 3, 4, 5, 6])
        scheduler.block_manager.ensure_capacity(first, 4)
        scheduler.state_manager.allocate(first)
        first.committed_tokens = 4
        scheduler.add(first)

        second = Sequence([7, 8, 9, 10, 11, 12])
        scheduler.block_manager.ensure_capacity(second, 4)
        scheduler.state_manager.allocate(second)
        second.committed_tokens = 4
        scheduler.add(second)

        self.assertFalse(scheduler.block_manager.free_block_ids)

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            [first],
        )
        self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 2)
        self.assertIn(first, scheduler.running)
        self.assertEqual(second.committed_tokens, 0)
        self.assertEqual(second.state_slot, -1)
        self.assertFalse(second.block_table)
        self.assertEqual(list(scheduler.waiting), [second])

    def test_preempted_waiter_moves_behind_other_waiters(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=3,
            num_blocks=3,
            block_size=4,
        )

        victim = Sequence([1, 2, 3, 4, 5])
        scheduler.block_manager.ensure_capacity(victim, 4)
        scheduler.state_manager.allocate(victim)
        victim.committed_tokens = 4
        scheduler.add(victim)

        peer = Sequence([6, 7])
        scheduler.add(peer)

        scheduler.preempt(victim)

        self.assertEqual(list(scheduler.waiting), [peer, victim])
        self.assertEqual(victim.committed_tokens, 0)
        self.assertEqual(victim.state_slot, -1)
        self.assertFalse(victim.block_table)

    def test_decode_headroom_preserves_next_growth_block(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=2,
            num_blocks=2,
            block_size=4,
        )
        running = make_running_sequence(
            scheduler,
            [1, 2, 3, 4],
        )
        waiter = Sequence([5])
        scheduler.add(waiter)

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.decode_chunks),
            [running],
        )
        self.assertEqual(scheduled.prefill_chunks, ())
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 1)
        self.assertEqual(list(scheduler.waiting), [waiter])

    def test_deadlock_recovery_admits_only_one_request(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=3,
            num_blocks=4,
            block_size=4,
        )

        first = Sequence([1, 2, 3, 4, 5])
        scheduler.block_manager.ensure_capacity(first, 4)
        scheduler.state_manager.allocate(first)
        first.committed_tokens = 4
        scheduler.add(first)

        second = Sequence([6, 7, 8, 9, 10])
        scheduler.block_manager.ensure_capacity(second, 4)
        scheduler.state_manager.allocate(second)
        second.committed_tokens = 4
        scheduler.add(second)

        victim = Sequence(list(range(20, 29)))
        scheduler.block_manager.ensure_capacity(victim, 8)
        scheduler.state_manager.allocate(victim)
        victim.committed_tokens = 8
        scheduler.add(victim)

        scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            [first],
        )
        self.assertEqual(victim.committed_tokens, 0)
        self.assertEqual(victim.state_slot, -1)
        self.assertFalse(victim.block_table)
        self.assertIn(second, scheduler.waiting)
        self.assertIn(victim, scheduler.waiting)

    def test_resource_accounting_tracks_joint_prefix_refs(self):
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
        scheduler.postprocess(
            scheduled.prefill_chunks,
            [None],
            True,
            {seq.seq_id: snapshot},
        )

        scheduler.validate_resource_accounting()

    def test_resource_accounting_detects_kv_refcount_corruption(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4,
            max_num_seqs=2,
            num_blocks=4,
            block_size=4,
        )
        seq = Sequence([1, 2, 3])
        scheduler.block_manager.ensure_capacity(seq, 3)
        scheduler.state_manager.allocate(seq)
        seq.committed_tokens = 3
        scheduler.add(seq)

        scheduler.validate_resource_accounting()
        block_id = seq.block_table[0]
        scheduler.block_manager.block_refcounts[block_id] += 1

        with self.assertRaisesRegex(
            RuntimeError,
            "KV cache refcount accounting mismatch",
        ):
            scheduler.validate_resource_accounting()

    def test_scheduler_rejects_committed_history_without_resources(self):
        scheduler = make_scheduler()
        seq = Sequence([1, 2])
        seq.committed_tokens = 1
        scheduler.add(seq)

        with self.assertRaisesRegex(
            RuntimeError,
            "committed history without hybrid resources",
        ):
            scheduler.schedule()


if __name__ == "__main__":
    unittest.main()
