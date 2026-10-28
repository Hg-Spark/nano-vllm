import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import make_running_sequence, make_scheduler, scheduled_sequences


class SchedulerAdmissionTest(unittest.TestCase):
    def test_three_requests_finish_with_ten_kv_pages(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=512,
            max_num_seqs=2,
            num_blocks=10,
            block_size=16,
            max_prefix_cache_entries=1,
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
                scheduler.postprocess(output.decode_chunks, [2] * len(output.decode_chunks), False)
            if output.prefill_chunks:
                snapshots = {
                    chunk.seq.seq_id: tuple(
                        GDNStateSnapshot(position, ()) for position in chunk.checkpoint_positions
                    )
                    for chunk in output.prefill_chunks
                    if chunk.checkpoint_positions
                }
                sampled = [
                    2 if chunk.end == len(chunk.seq) else None for chunk in output.prefill_chunks
                ]
                scheduler.postprocess(output.prefill_chunks, sampled, True, snapshots)
            if scheduler.is_finished():
                break

        self.assertTrue(scheduler.is_finished())
        self.assertEqual([seq.completion_token_ids for seq in seqs], [[2] * n for _, n in specs])

    def test_admission_without_prefix_cache_uses_request_reservations(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=2,
            num_blocks=4,
            block_size=4,
            max_prefix_cache_entries=0,
        )
        running = make_running_sequence(scheduler, [10, 11, 12, 13, 14])
        waiting = Sequence([21, 22, 23], SamplingParams(max_tokens=1))
        scheduler.add(waiting)

        output = scheduler.schedule()
        self.assertEqual(scheduled_sequences(output.decode_chunks), [running])
        self.assertEqual(scheduled_sequences(output.prefill_chunks), [waiting])
        self.assertEqual(output.prefill_tokens, 3)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 1)

    def test_admission_counts_shared_kv_blocks_only_once(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=128,
            max_num_seqs=2,
            num_blocks=13,
            block_size=16,
        )
        source = Sequence(list(range(64)) + [200])
        scheduler.block_manager.ensure_capacity(source, 64)
        scheduler.block_manager.prefix_cache.publish_many(source, (GDNStateSnapshot(64, ()),))
        scheduler.block_manager.deallocate(source)

        for suffix in (list(range(64, 128)), list(range(64, 127)) + [200]):
            scheduler.add(Sequence(list(range(64)) + suffix, SamplingParams(max_tokens=1)))
        scheduled = scheduler.schedule()
        self.assertEqual(len(scheduled.prefill_chunks), 2)
        self.assertEqual([chunk.num_tokens for chunk in scheduled.prefill_chunks], [64, 64])
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 1)

    def test_partial_prefills_resume_before_new_requests(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4, max_num_seqs=2, num_blocks=8, block_size=4
        )
        first = Sequence(list(range(6)), SamplingParams(max_tokens=1))
        second = Sequence(list(range(10, 16)), SamplingParams(max_tokens=1))
        scheduler.add(first)
        scheduler.add(second)

        step1 = scheduler.schedule()
        scheduler.postprocess(step1.prefill_chunks, [None], True)
        step2 = scheduler.schedule()
        self.assertEqual(scheduled_sequences(step2.prefill_chunks), [first, second])
        self.assertEqual([chunk.num_tokens for chunk in step2.prefill_chunks], [2, 2])
        scheduler.postprocess(step2.prefill_chunks, [11, None], True)
        self.assertEqual(second.committed_tokens, 2)
        self.assertTrue(first.is_finished)
        self.assertEqual(list(scheduler.running), [])


if __name__ == "__main__":
    unittest.main()
