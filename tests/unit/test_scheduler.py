import unittest
from unittest.mock import patch

from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import (
    make_running_sequence,
    make_scheduler,
    scheduled_sequences,
)


class SchedulerCoreTest(unittest.TestCase):
    def test_decode_budget_and_sequence_limit(self):
        cases = ((4, 4, 3), (8, 1, 0))
        for budget, max_seqs, expected_prefill in cases:
            with self.subTest(budget=budget, max_seqs=max_seqs):
                scheduler = make_scheduler(max_num_batched_tokens=budget, max_num_seqs=max_seqs)
                decode_seq = make_running_sequence(scheduler, [1, 2])
                prefill_seq = Sequence(list(range(10)))
                scheduler.add(prefill_seq)

                scheduled = scheduler.schedule()
                self.assertEqual(scheduled_sequences(scheduled.decode_chunks), [decode_seq])
                self.assertEqual(scheduled.decode_tokens, 1)
                self.assertEqual(scheduled.prefill_tokens, expected_prefill)
                self.assertEqual(
                    scheduled_sequences(scheduled.prefill_chunks),
                    [prefill_seq] if expected_prefill else [],
                )
                if expected_prefill:
                    self.assertEqual(scheduled.prefill_chunks[0].num_tokens, 3)

    def test_mixed_prefill_cap_preserves_pure_prefill_budget(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=512,
            max_num_seqs=2,
            num_blocks=256,
            max_mixed_prefill_tokens=64,
        )
        running = make_running_sequence(scheduler, [1, 2])
        seq = Sequence(list(range(1024)), SamplingParams(max_tokens=1))
        scheduler.add(seq)

        mixed = scheduler.schedule()
        self.assertEqual(mixed.decode_tokens, 1)
        self.assertEqual(mixed.prefill_tokens, 64)
        self.assertEqual(mixed.prefill_chunks[0].end, 64)
        scheduler.postprocess(mixed.decode_chunks, [7], False)
        scheduler.postprocess(mixed.prefill_chunks, [None], True)

        # Removing the decode request restores the normal prefill budget.
        scheduler.running.remove(running)
        scheduler._release_runtime_state(running)
        prefill = scheduler.schedule()
        self.assertEqual(prefill.decode_tokens, 0)
        self.assertGreater(prefill.prefill_tokens, 64)

    def test_zero_mixed_prefill_cap_skips_prefill_until_decode_stops(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=64,
            max_num_seqs=2,
            num_blocks=64,
            max_mixed_prefill_tokens=0,
        )
        running = make_running_sequence(scheduler, [1, 2])
        waiting = Sequence(list(range(100)), SamplingParams(max_tokens=1))
        scheduler.add(waiting)
        with patch.object(
            scheduler.cache_manager,
            "reservation_usage",
            side_effect=AssertionError("no KV reservation scan when prefill cap is zero"),
        ):
            scheduled = scheduler.schedule()
        self.assertEqual(scheduled.decode_tokens, 1)
        self.assertFalse(scheduled.prefill_chunks)
        self.assertIn(waiting, scheduler.waiting)
        self.assertIn(running, scheduler.running)

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

    def test_trimmed_prefill_does_not_evict_checkpoint_for_unused_kv_page(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=65,
            max_num_seqs=1,
            num_blocks=8,
            block_size=16,
            max_prefix_cache_entries=4,
        )
        cache = scheduler.cache_manager
        old = Sequence([900] * 80)
        cache.block_manager.ensure_capacity(old, 64)
        cache.prefix_cache.publish_many(old, (GDNStateSnapshot(64, ()),))
        cache.release(old)
        self.assertEqual(len(cache.prefix_cache), 1)
        self.assertEqual(len(cache.block_manager.free_block_ids), 4)

        incoming = Sequence(list(range(80)), SamplingParams(max_tokens=1))
        scheduler.add(incoming)
        scheduled = scheduler.schedule()
        self.assertEqual(len(scheduled.prefill_chunks), 1)
        chunk = scheduled.prefill_chunks[0]
        self.assertEqual((chunk.start, chunk.end), (0, 64))
        self.assertIsNotNone(cache.prefix_cache.find(Sequence([900] * 80), 64))
        self.assertEqual(len(cache.prefix_cache), 1)

    def test_disabled_prefix_publication_does_not_split_prefill(self):
        cases = (
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
                    [(c.start, c.end) for c in chunks],
                    [(0, 20)],
                )

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

        scheduler.postprocess(
            second.prefill_chunks,
            [None],
            True,
        )
        third = scheduler.schedule()
        self.assertEqual(third.prefill_tokens, 2)
        self.assertEqual(len(seq.block_table), 2)
        scheduler.postprocess(third.prefill_chunks, [None], True)
        fourth = scheduler.schedule()
        self.assertEqual(fourth.prefill_tokens, 2)
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
        scheduler.cache_manager.block_manager.ensure_capacity(seq, len(seq))
        scheduler.cache_manager.state_manager.allocate(seq)
        seq.committed_tokens = len(seq) - 1
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

    def test_prefill_never_trims_chunks_to_checkpoint_grid(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=64, max_num_seqs=2, num_blocks=128, block_size=16
        )
        running = make_running_sequence(scheduler, [1, 2])
        running.max_tokens = 10
        seq = Sequence(list(range(1024)), SamplingParams(max_tokens=1))
        scheduler.add(seq)
        ends = []
        for _ in range(4):
            batch = scheduler.schedule()
            ends.append(batch.prefill_chunks[0].end)
            scheduler.postprocess(batch.decode_chunks, [7], False)
            scheduler.postprocess(batch.prefill_chunks, [None], True)
        self.assertEqual(ends, [63, 126, 189, 252])
        self.assertIn(running, scheduler.running)

    def test_prefill_skips_eviction_when_smaller_chunk_can_progress(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=65,
            num_blocks=8,
            block_size=16,
            max_prefix_cache_entries=4,
        )
        cache = scheduler.cache_manager
        old = Sequence([900] * 65)
        cache.block_manager.ensure_capacity(old, 64)
        cache.prefix_cache.publish_many(old, (GDNStateSnapshot(64, ()),))
        cache.release(old)
        incoming = Sequence(list(range(80)), SamplingParams(max_tokens=1))
        scheduler.add(incoming)
        chunk = scheduler.schedule().prefill_chunks[0]
        self.assertEqual((chunk.start, chunk.end), (0, 64))
        self.assertEqual(len(cache.prefix_cache), 1)


if __name__ == "__main__":
    unittest.main()
