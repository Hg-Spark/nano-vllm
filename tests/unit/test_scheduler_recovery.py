import unittest
from unittest.mock import patch

from nanovllm.engine.sequence import Sequence, SequenceStatus
from tests.scheduler_helpers import (
    make_running_sequence,
    make_scheduler,
    scheduled_sequences,
)


class SchedulerRecoveryTest(unittest.TestCase):

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

    def test_waiting_deadlock_retries_shared_prefix_holders(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=3,
            num_blocks=2,
            block_size=4,
        )

        first_shared = Sequence([1, 2, 3, 4, 5])
        scheduler.block_manager.ensure_capacity(first_shared, 4)
        scheduler.state_manager.allocate(first_shared)
        first_shared.committed_tokens = 4
        shared_block = first_shared.block_table[0]

        second_shared = Sequence([1, 2, 3, 4, 6])
        scheduler.state_manager.allocate(second_shared)
        scheduler.block_manager.attach_shared_prefix(
            second_shared,
            (shared_block,),
            4,
        )
        second_shared.committed_tokens = 4

        independent = Sequence([7, 8, 9, 10, 11])
        scheduler.block_manager.ensure_capacity(independent, 4)
        scheduler.state_manager.allocate(independent)
        independent.committed_tokens = 4

        # Reverse victim selection first sees first_shared. Releasing it only
        # drops the shared block's refcount and does not free a physical page.
        scheduler.add(independent)
        scheduler.add(second_shared)
        scheduler.add(first_shared)
        self.assertFalse(scheduler.block_manager.free_block_ids)

        with patch.object(
            scheduler,
            "_try_restore_prefix",
            wraps=scheduler._try_restore_prefix,
        ) as restore:
            scheduled = scheduler.schedule()

        self.assertEqual(
            scheduled_sequences(scheduled.prefill_chunks),
            [independent],
        )
        self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 1)
        self.assertEqual(first_shared.committed_tokens, 0)
        self.assertEqual(second_shared.committed_tokens, 0)
        self.assertFalse(first_shared.block_table)
        self.assertFalse(second_shared.block_table)
        self.assertEqual(restore.call_count, 0)

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
        slot = victim.state_slot
        blocks = tuple(victim.block_table)
        scheduler.add(victim)

        peer = Sequence([6, 7])
        scheduler.add(peer)

        scheduler.preempt(victim)

        self.assertEqual(list(scheduler.waiting), [peer, victim])
        self.assertEqual(victim.committed_tokens, 0)
        self.assertEqual(victim.state_slot, -1)
        self.assertFalse(victim.block_table)
        self.assertIsNone(scheduler.state_manager.slot_owners[slot])
        self.assertTrue(all(
            scheduler.block_manager.block_refcounts[block_id] == 0
            for block_id in blocks
        ))

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


if __name__ == "__main__":
    unittest.main()
