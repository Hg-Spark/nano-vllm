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

    def test_chunked_prefill_keeps_new_request_fifo(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=4, max_num_seqs=2, num_blocks=16, block_size=4
        )
        admitted = Sequence(list(range(12)), SamplingParams(max_tokens=1))
        first = Sequence([21, 22], SamplingParams(max_tokens=1))
        second = Sequence([31, 32], SamplingParams(max_tokens=1))
        scheduler.add(admitted)
        initial = scheduler.schedule().prefill_chunks
        scheduler.postprocess(initial, [None], True)
        scheduler.add(first)
        middle = scheduler.schedule().prefill_chunks
        scheduler.postprocess(middle, [None], True)
        scheduler.add(second)

        final = scheduler.schedule().prefill_chunks
        self.assertEqual(list(scheduler.waiting), [first, second])
        scheduler.postprocess(final, [99], True)
        following = scheduler.schedule().prefill_chunks
        self.assertIs(following[0].seq, first)

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

    def test_power_of_two_checkpoint_grid(self):
        cases = (
            (20, ()),
            (100, (64,)),
            (320, (64, 128, 192, 256)),
            (321, (128, 256)),
            (400, (128, 256, 384)),
            (1024, (256, 512, 768)),
            (2049, (512, 1024, 1536, 2048)),
        )
        for prompt_len, positions in cases:
            with self.subTest(prompt_len=prompt_len):
                scheduler = make_scheduler(
                    max_num_batched_tokens=4096,
                    max_num_seqs=1,
                    num_blocks=256,
                    block_size=16,
                )
                seq = Sequence(list(range(prompt_len)), SamplingParams(max_tokens=1))
                scheduler.add(seq)
                chunks = scheduler.schedule().prefill_chunks
                self.assertEqual(len(chunks), 1)
                self.assertEqual((chunks[0].start, chunks[0].end), (0, prompt_len))
                self.assertEqual(chunks[0].checkpoint_positions, positions)

    def test_disabled_prefix_publication_does_not_split_prefill(self):
        cases = (
            ("non_eager", {"max_prefix_cache_entries": 16, "prefix_cache_eager_snapshot": False}),
            ("disabled", {"max_prefix_cache_entries": 0}),
            ("zero_snapshot_budget", {"max_prefix_cache_snapshot_bytes": 0}),
            (
                "undersized_snapshot_budget",
                {"max_prefix_cache_snapshot_bytes": 1, "snapshot_nbytes": 16},
            ),
        )
        for label, options in cases:
            with self.subTest(mode=label):
                scheduler = make_scheduler(max_num_batched_tokens=512, block_size=16, **options)
                scheduler.add(Sequence(list(range(20))))
                chunks = scheduler.schedule().prefill_chunks
                self.assertEqual(
                    [(c.start, c.end, bool(c.checkpoint_positions)) for c in chunks],
                    [(0, 20, False)],
                )

    def test_non_eager_captures_natural_checkpoint_boundaries(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=64,
            max_num_seqs=1,
            num_blocks=16,
            block_size=16,
            prefix_cache_eager_snapshot=False,
        )
        seq = Sequence(list(range(130)), SamplingParams(max_tokens=1))
        scheduler.add(seq)
        first = scheduler.schedule().prefill_chunks
        self.assertEqual((first[0].end, first[0].checkpoint_positions), (64, (64,)))
        scheduler.postprocess(first, [None], True, {seq.seq_id: (GDNStateSnapshot(64, ()),)})
        second = scheduler.schedule().prefill_chunks
        self.assertEqual((second[0].end, second[0].checkpoint_positions), (128, (128,)))

    def test_chunked_prefill_allocates_only_scheduled_kv_range(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=3,
            max_num_seqs=2,
            num_blocks=4,
            block_size=4,
        )
        seq = Sequence(list(range(10)), SamplingParams(max_tokens=1))
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
        self.assertFalse(second.prefill_chunks[0].checkpoint_positions)

        scheduler.postprocess(
            second.prefill_chunks,
            [None],
            True,
        )
        third = scheduler.schedule()
        self.assertEqual(third.prefill_tokens, 3)
        self.assertEqual(len(seq.block_table), 3)

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

        newcomer = Sequence([5, 6, 7], SamplingParams(max_tokens=2))
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

    def test_admitted_decode_does_not_need_preemption(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=2,
            max_num_seqs=2,
            num_blocks=3,
            block_size=4,
        )
        first = make_running_sequence(scheduler, [1, 2, 3, 4, 5])
        second = make_running_sequence(scheduler, [6, 7, 8])
        scheduled = scheduler.schedule()
        self.assertEqual(scheduled_sequences(scheduled.decode_chunks), [first, second])
        self.assertEqual(list(scheduler.running), [first, second])

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
