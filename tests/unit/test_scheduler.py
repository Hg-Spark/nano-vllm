import unittest

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

    def test_fixed_64_token_checkpoint_grid(self):
        for prompt_len in (20, 100, 320, 321, 400, 1024, 2049):
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
                candidates = list(range(64, prompt_len + 1, 64))
                expected = candidates[:1] + candidates[-1:] if len(candidates) > 2 else candidates
                self.assertEqual(chunks[0].checkpoint_positions, tuple(expected))

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
                    [(c.start, c.end, bool(c.checkpoint_positions)) for c in chunks],
                    [(0, 20, False)],
                )

    def test_decode_budget_bounds_checkpoint_alignment_cost(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=512,
            max_num_seqs=2,
            num_blocks=256,
            block_size=16,
        )
        running = make_running_sequence(scheduler, [1, 2])
        seq = Sequence(list(range(2048)), SamplingParams(max_tokens=1))
        scheduler.add(seq)

        first = scheduler.schedule()
        self.assertEqual(first.prefill_chunks[0].num_tokens, 448)
        scheduler.postprocess(first.decode_chunks, [7], False)
        scheduler.postprocess(first.prefill_chunks, [None], True)

        second = scheduler.schedule()
        self.assertEqual(second.prefill_chunks[0].num_tokens, 448)
        self.assertEqual(second.prefill_chunks[0].start, 448)
        self.assertIn(running, [chunk.seq for chunk in second.decode_chunks])

        # With 128-token pages, spending 127 of 511 tokens to end at 384
        # is too expensive. The 128/384 checkpoints remain capturable
        # internally because this chunk starts on the joint grid.
        scheduler = make_scheduler(
            max_num_batched_tokens=512,
            max_num_seqs=2,
            num_blocks=20,
            block_size=128,
        )
        make_running_sequence(scheduler, [1, 2])
        candidate = Sequence(list(range(512)), SamplingParams(max_tokens=1))
        scheduler.add(candidate)
        scheduled = scheduler.schedule()
        self.assertEqual(
            (scheduled.prefill_chunks[0].start, scheduled.prefill_chunks[0].end),
            (0, 511),
        )
        self.assertEqual(scheduled.prefill_chunks[0].checkpoint_positions, (128, 384))
        self.assertEqual(scheduled.prefill_chunks[0].checkpoint_interval, 64)
        scheduler.postprocess(scheduled.decode_chunks, [7], False)
        scheduler.postprocess(scheduled.prefill_chunks, [None], True)
        continued = scheduler.schedule().prefill_chunks[0]
        self.assertEqual((continued.start, continued.end), (511, 512))
        self.assertEqual(continued.checkpoint_positions, (512,))
        self.assertEqual(continued.checkpoint_interval, 0)

    def test_off_grid_prefill_recovers_early_without_repeated_tiny_chunks(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=64,
            max_num_seqs=2,
            num_blocks=256,
            block_size=16,
        )
        running = Sequence([1, 2], SamplingParams(max_tokens=100, ignore_eos=True))
        scheduler.block_manager.ensure_capacity(running, len(running))
        scheduler.state_manager.allocate(running)
        running.committed_tokens = len(running) - 1
        scheduler.running.append(running)

        seq = Sequence(list(range(3500)), SamplingParams(max_tokens=1))
        scheduler.add(seq)
        ends = []
        captured = []
        for _ in range(8):
            batch = scheduler.schedule()
            self.assertEqual(batch.decode_tokens, 1)
            self.assertEqual(len(batch.prefill_chunks), 1)
            chunk = batch.prefill_chunks[0]
            self.assertEqual(chunk.checkpoint_interval, 0)
            ends.append(chunk.end)
            captured.extend(chunk.checkpoint_positions)
            scheduler.postprocess(batch.decode_chunks, [7], False)
            snapshots = {
                seq.seq_id: tuple(
                    GDNStateSnapshot(position, ()) for position in chunk.checkpoint_positions
                )
            }
            scheduler.postprocess(batch.prefill_chunks, [None], True, snapshots)

        # Capture at 64, then avoid a tiny chunk at every grid boundary.
        self.assertEqual(ends, [63, 64, 127, 190, 253, 316, 379, 384])
        self.assertEqual(captured, [64, 384])
        self.assertIsNotNone(scheduler.block_manager.prefix_cache.find(seq, 384))
        self.assertEqual(seq.committed_tokens, 384)

    def test_off_grid_prefill_recovers_on_joint_128_token_boundary(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=64,
            max_num_seqs=2,
            num_blocks=256,
            block_size=128,
        )
        running = Sequence([1, 2], SamplingParams(max_tokens=100, ignore_eos=True))
        scheduler.block_manager.ensure_capacity(running, len(running))
        scheduler.state_manager.allocate(running)
        running.committed_tokens = len(running) - 1
        scheduler.running.append(running)

        seq = Sequence(list(range(3500)), SamplingParams(max_tokens=1))
        scheduler.add(seq)
        ends = []
        positions = []
        for _ in range(4):
            scheduled = scheduler.schedule()
            self.assertEqual(scheduled.decode_tokens, 1)
            chunk = scheduled.prefill_chunks[0]
            ends.append(chunk.end)
            positions.extend(chunk.checkpoint_positions)
            self.assertEqual(chunk.checkpoint_interval, 0)
            scheduler.postprocess(scheduled.decode_chunks, [7], False)
            snapshots = {
                seq.seq_id: tuple(
                    GDNStateSnapshot(position, ()) for position in chunk.checkpoint_positions
                )
            }
            scheduler.postprocess(scheduled.prefill_chunks, [None], True, snapshots)

        self.assertEqual(ends, [63, 126, 128, 191])
        self.assertEqual(positions, [128])
        self.assertIsNotNone(scheduler.block_manager.prefix_cache.find(seq, 128))

    def test_off_grid_terminal_chunk_never_splits_for_checkpoint(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=64,
            max_num_seqs=2,
            num_blocks=64,
            block_size=16,
        )
        running = make_running_sequence(scheduler, [1, 2])
        seq = Sequence(list(range(100)), SamplingParams(max_tokens=1))
        scheduler.add(seq)
        first = scheduler.schedule()
        self.assertEqual(first.prefill_chunks[0].end, 63)
        scheduler.postprocess(first.decode_chunks, [7], False)
        scheduler.postprocess(first.prefill_chunks, [None], True)

        final = scheduler.schedule()
        self.assertEqual(
            (final.prefill_chunks[0].start, final.prefill_chunks[0].end),
            (63, 100),
        )
        self.assertFalse(final.prefill_chunks[0].checkpoint_positions)

    def test_short_remaining_prompt_does_not_split_for_cache(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=128,
            max_num_seqs=1,
            num_blocks=64,
            block_size=16,
        )
        seq = Sequence(list(range(100)), SamplingParams(max_tokens=1))
        scheduler.add(seq)
        chunk = scheduler.schedule().prefill_chunks[0]
        self.assertEqual((chunk.start, chunk.end), (0, 100))
        self.assertEqual(chunk.checkpoint_positions, (64,))

    def test_natural_64_checkpoint_after_small_prefill_chunks(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=16, max_num_seqs=1, num_blocks=16, block_size=16
        )
        seq = Sequence(list(range(69)), SamplingParams(max_tokens=1))
        scheduler.add(seq)
        for end in (16, 32, 48, 64):
            chunks = scheduler.schedule().prefill_chunks
            self.assertEqual(chunks[0].end, end)
            self.assertEqual(chunks[0].checkpoint_positions, (64,) if end == 64 else ())
            snapshots = {seq.seq_id: (GDNStateSnapshot(64, ()),)} if end == 64 else None
            scheduler.postprocess(chunks, [None], True, snapshots)

        entry = scheduler.block_manager.prefix_cache.find(seq, max_tokens=64)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.num_tokens, 64)

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
