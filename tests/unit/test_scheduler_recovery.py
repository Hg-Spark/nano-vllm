import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import make_scheduler, scheduled_sequences


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
                    chunk.seq.seq_id: GDNStateSnapshot(chunk.end, ())
                    for chunk in output.prefill_chunks
                    if chunk.capture_snapshot
                }
                sampled = [
                    2 if chunk.end == len(chunk.seq) else None for chunk in output.prefill_chunks
                ]
                scheduler.postprocess(output.prefill_chunks, sampled, True, snapshots)
            if scheduler.is_finished():
                break

        self.assertTrue(scheduler.is_finished())
        self.assertEqual([seq.completion_token_ids for seq in seqs], [[2] * n for _, n in specs])

    def test_eager_snapshot_preserves_parallel_prefill(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=512, max_num_seqs=2, num_blocks=16, block_size=16
        )
        seqs = [
            Sequence([token] * 20, SamplingParams(max_tokens=2)) for token in (1, 2)
        ]
        for seq in seqs:
            scheduler.add(seq)

        chunks = scheduler.schedule().prefill_chunks
        self.assertEqual(scheduled_sequences(chunks), seqs)
        self.assertEqual([chunk.end for chunk in chunks], [16, 16])

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
        scheduler.postprocess(step2.prefill_chunks, [None], True)
        self.assertEqual([seq.committed_tokens for seq in scheduler.waiting], [4, 4])
        self.assertNotEqual(first.state_slot, second.state_slot)

        step3 = scheduler.schedule()
        self.assertEqual(scheduled_sequences(step3.prefill_chunks), [first, second])
        self.assertEqual([chunk.num_tokens for chunk in step3.prefill_chunks], [2, 2])
        scheduler.postprocess(step3.prefill_chunks, [11, 12], True)
        self.assertEqual(list(scheduler.running), [first, second])


if __name__ == "__main__":
    unittest.main()
