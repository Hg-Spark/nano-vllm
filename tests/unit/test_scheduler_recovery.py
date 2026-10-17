import unittest

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import (
    make_scheduler,
    scheduled_sequences,
)


class SchedulerRecoveryTest(unittest.TestCase):
    def test_blocked_waiter_does_not_block_schedulable_waiter(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2, max_num_seqs=2, num_blocks=2, block_size=4
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
        self.assertEqual(scheduled_sequences(scheduled.prefill_chunks), [schedulable])
        self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 1)
        self.assertEqual(list(scheduler.waiting), [blocked])
        self.assertIn(schedulable, scheduler.running)

    def test_decode_preempts_resource_holding_partial_prefill(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2, max_num_seqs=2, num_blocks=2, block_size=4
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

        scheduled = scheduler.schedule()
        self.assertEqual(scheduled_sequences(scheduled.decode_chunks), [decode_seq])
        self.assertEqual(partial.committed_tokens, 0)
        self.assertEqual(partial.state_slot, -1)
        self.assertFalse(partial.block_table)

    def test_waiting_deadlock_preempts_holders_until_progress(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2, max_num_seqs=3, num_blocks=2, block_size=4
        )
        first_shared = Sequence([1, 2, 3, 4, 5])
        scheduler.block_manager.ensure_capacity(first_shared, 4)
        scheduler.state_manager.allocate(first_shared)
        first_shared.committed_tokens = 4
        shared_block = first_shared.block_table[0]

        second_shared = Sequence([1, 2, 3, 4, 6])
        scheduler.state_manager.allocate(second_shared)
        scheduler.block_manager.attach_shared_prefix(second_shared, (shared_block,), 4)
        second_shared.committed_tokens = 4

        independent = Sequence([7, 8, 9, 10, 11])
        scheduler.block_manager.ensure_capacity(independent, 4)
        scheduler.state_manager.allocate(independent)
        independent.committed_tokens = 4

        scheduler.add(independent)
        scheduler.add(second_shared)
        scheduler.add(first_shared)

        scheduled = scheduler.schedule()
        self.assertEqual(scheduled_sequences(scheduled.prefill_chunks), [independent])
        self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 1)
        self.assertFalse(first_shared.block_table)
        self.assertFalse(second_shared.block_table)

    def test_multiple_partial_prefills_can_finish_in_one_batch(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4, max_num_seqs=2, num_blocks=8, block_size=4
        )
        first = Sequence(list(range(6)))
        second = Sequence(list(range(10, 16)))
        scheduler.add(first)
        scheduler.add(second)

        step1 = scheduler.schedule()
        self.assertEqual(scheduled_sequences(step1.prefill_chunks), [first])
        scheduler.postprocess(step1.prefill_chunks, [None], True)

        step2 = scheduler.schedule()
        self.assertEqual(scheduled_sequences(step2.prefill_chunks), [second])
        scheduler.postprocess(step2.prefill_chunks, [None], True)
        self.assertEqual([seq.committed_tokens for seq in scheduler.waiting], [4, 4])
        self.assertNotEqual(first.state_slot, second.state_slot)

        step3 = scheduler.schedule()
        self.assertEqual(scheduled_sequences(step3.prefill_chunks), [first, second])
        self.assertEqual([chunk.num_tokens for chunk in step3.prefill_chunks], [2, 2])
        scheduler.postprocess(step3.prefill_chunks, [11, 12], True)
        self.assertEqual((first.committed_tokens, second.committed_tokens), (6, 6))
        self.assertEqual(list(scheduler.running), [first, second])

    def test_prefill_recovers_without_repeated_preemption_cycle(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=512,
            max_num_seqs=2,
            num_blocks=10,
            block_size=16,
            max_prefix_cache_entries=1,
            prefix_cache_eager_snapshot=True,
        )
        specs = ((113, 32), (110, 39), (93, 4))
        seqs = [
            Sequence(
                [1] * prompt_len,
                SamplingParams(max_tokens=max_tokens, ignore_eos=True),
            )
            for prompt_len, max_tokens in specs
        ]
        for seq in seqs:
            scheduler.add(seq)

        for _ in range(250):
            output = scheduler.schedule()
            if output.decode_chunks:
                scheduler.postprocess(
                    output.decode_chunks, [2] * len(output.decode_chunks), False
                )
            if output.prefill_chunks:
                snapshots = {
                    chunk.seq.seq_id: GDNStateSnapshot(chunk.end, ())
                    for chunk in output.prefill_chunks
                    if chunk.capture_snapshot
                }
                sampled = [
                    2 if chunk.end == len(chunk.seq) else None
                    for chunk in output.prefill_chunks
                ]
                scheduler.postprocess(output.prefill_chunks, sampled, True, snapshots)
            if scheduler.is_finished():
                break

        self.assertTrue(scheduler.is_finished(), "resource pressure stalled prefill completion")
        self.assertEqual([seq.completion_token_ids for seq in seqs], [[2] * n for _, n in specs])


if __name__ == "__main__":
    unittest.main()
