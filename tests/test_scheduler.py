import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(max_num_batched_tokens=4, max_num_seqs=4, num_blocks=16):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=num_blocks,
        enable_prefix_cache=True,
        max_num_state_slots=max_num_seqs,
    )
    Sequence.block_size = config.kvcache_block_size
    return Scheduler(config)


def make_running_sequence(scheduler, token_ids):
    seq = Sequence(token_ids)
    scheduler.block_manager.allocate(seq, 0)
    scheduler.state_manager.allocate(seq)
    seq.num_cached_tokens = len(seq)
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    scheduler.running.append(seq)
    return seq


class SchedulerTest(unittest.TestCase):

    def test_decode_uses_budget_before_chunked_prefill(self):
        scheduler = make_scheduler(max_num_batched_tokens=4, max_num_seqs=4)
        decode_seq = make_running_sequence(scheduler, [1, 2])
        prefill_seq = Sequence(list(range(10)))
        scheduler.add(prefill_seq)

        scheduled = scheduler.schedule()

        self.assertEqual(scheduled.decode_seqs, [decode_seq])
        self.assertEqual(scheduled.prefill_seqs, [prefill_seq])
        self.assertEqual(scheduled.decode_tokens, 1)
        self.assertEqual(scheduled.prefill_tokens, 3)
        self.assertEqual(prefill_seq.num_scheduled_tokens, 3)
        self.assertGreaterEqual(prefill_seq.state_slot, 0)

    def test_sequence_limit_is_shared_by_decode_and_prefill(self):
        scheduler = make_scheduler(max_num_batched_tokens=8, max_num_seqs=1)
        decode_seq = make_running_sequence(scheduler, [1, 2])
        scheduler.add(Sequence([3, 4, 5]))

        scheduled = scheduler.schedule()

        self.assertEqual(scheduled.decode_seqs, [decode_seq])
        self.assertEqual(scheduled.prefill_seqs, [])

    def test_partial_prefill_keeps_state_slot_across_chunks(self):
        scheduler = make_scheduler(max_num_batched_tokens=3, max_num_seqs=2)
        seq = Sequence(list(range(8)))
        scheduler.add(seq)

        first = scheduler.schedule()
        first_slot = seq.state_slot
        self.assertEqual(first.prefill_tokens, 3)
        self.assertGreaterEqual(first_slot, 0)
        scheduler.postprocess(first.prefill_seqs, [123], True)

        second = scheduler.schedule()
        self.assertEqual(second.prefill_tokens, 3)
        self.assertEqual(seq.state_slot, first_slot)

    def test_prefix_cache_can_be_disabled_for_hybrid_state(self):
        scheduler = make_scheduler(max_num_batched_tokens=512, max_num_seqs=2)
        scheduler.block_manager.enable_prefix_cache = False

        first = Sequence([7] * 256 + [8])
        cached = scheduler.block_manager.can_allocate(first)
        self.assertEqual(cached, 0)
        scheduler.block_manager.allocate(first, cached)
        first.num_scheduled_tokens = first.num_tokens
        scheduler.block_manager.hash_blocks(first)
        scheduler.block_manager.deallocate(first)

        second = Sequence([7] * 256 + [9])
        cached = scheduler.block_manager.can_allocate(second)
        self.assertEqual(cached, 0)

    def test_preempt_releases_state_slot(self):
        scheduler = make_scheduler(max_num_batched_tokens=4, max_num_seqs=2)
        seq = make_running_sequence(scheduler, [1, 2])
        slot = seq.state_slot

        scheduler.preempt(seq)

        self.assertEqual(seq.state_slot, -1)
        self.assertNotIn(slot, scheduler.state_manager.used_slot_ids)
        self.assertIn(slot, scheduler.state_manager.free_slot_ids)


if __name__ == "__main__":
    unittest.main()
