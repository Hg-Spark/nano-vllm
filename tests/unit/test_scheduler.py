import unittest

from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
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

    def test_unaligned_prompt_schedules_complete_prefix_block(self):
        for prompt_len, boundary in ((20, 16), (100, 96)):
            with self.subTest(prompt_len=prompt_len):
                scheduler = make_scheduler(
                    max_num_batched_tokens=512,
                    max_num_seqs=2,
                    num_blocks=16,
                    block_size=16,
                )
                scheduler.add(Sequence(list(range(prompt_len))))
                chunks = scheduler.schedule().prefill_chunks
                self.assertEqual(
                    [(chunk.start, chunk.end, chunk.capture_snapshot) for chunk in chunks],
                    [(0, boundary, True)],
                )

    def test_disabled_prefix_publication_does_not_split_prefill(self):
        cases = (
            ("non_eager", {"max_prefix_cache_entries": 16, "prefix_cache_eager_snapshot": False}),
            ("disabled", {"max_prefix_cache_entries": 0}),
            ("zero_snapshot_budget", {"max_prefix_cache_snapshot_bytes": 0}),
            ("undersized_snapshot_budget", {"max_prefix_cache_snapshot_bytes": 1, "snapshot_nbytes": 16}),
        )
        for label, options in cases:
            with self.subTest(mode=label):
                scheduler = make_scheduler(max_num_batched_tokens=512, block_size=16, **options)
                scheduler.add(Sequence(list(range(20))))
                chunks = scheduler.schedule().prefill_chunks
                self.assertEqual(
                    [(c.start, c.end, c.capture_snapshot) for c in chunks],
                    [(0, 20, False)],
                )

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
        self.assertEqual(second.prefill_tokens, 1)
        self.assertEqual(len(seq.block_table), 1)
        self.assertTrue(second.prefill_chunks[0].capture_snapshot)

        scheduler.postprocess(
            second.prefill_chunks,
            [None],
            True,
            {seq.seq_id: GDNStateSnapshot(num_tokens=4, layers=())},
        )
        third = scheduler.schedule()
        self.assertEqual(third.prefill_tokens, 3)
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

    def test_ignore_eos_completes_only_at_max_tokens(self):
        scheduler = make_scheduler(max_num_batched_tokens=4, max_num_seqs=1)
        seq = Sequence(
            [1, 2],
            SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True),
        )
        scheduler.block_manager.ensure_capacity(seq, len(seq))
        scheduler.state_manager.allocate(seq)
        seq.committed_tokens = len(seq) - 1
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        for index, token_id in enumerate((99, 7, 100), start=1):
            scheduled = scheduler.schedule()
            self.assertEqual(len(scheduled.decode_chunks), 1)
            scheduler.postprocess(scheduled.decode_chunks, [token_id], False)
            self.assertEqual(seq.num_completion_tokens, index)
            self.assertEqual(seq.is_finished, index == 3)

        self.assertEqual(seq.completion_token_ids, [99, 7, 100])
        self.assertTrue(scheduler.is_finished())

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

        third = scheduler.schedule()
        self.assertEqual(
            scheduled_sequences(third.prefill_chunks),
            [long_seq],
        )
        self.assertEqual(third.prefill_chunks[0].num_tokens, 3)

    def test_page_growth_does_not_starve_ready_decode(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=2,
            num_blocks=2,
            block_size=4,
        )
        blocked = Sequence([1, 2, 3, 4, 5])
        scheduler.block_manager.ensure_capacity(blocked, 4)
        scheduler.state_manager.allocate(blocked)
        blocked.committed_tokens = 4
        blocked.status = SequenceStatus.RUNNING
        scheduler.running.append(blocked)
        ready = make_running_sequence(scheduler, [6, 7, 8])
        scheduled = scheduler.schedule()
        self.assertEqual(scheduled_sequences(scheduled.decode_chunks), [ready])
        self.assertIn(blocked, scheduler.waiting)
        self.assertEqual(list(scheduler.running), [ready])

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


if __name__ == "__main__":
    unittest.main()
