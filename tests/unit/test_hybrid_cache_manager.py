import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import make_scheduler


class HybridCacheManagerTest(unittest.TestCase):
    def test_joint_prefix_admission_and_release(self):
        scheduler = make_scheduler(max_num_seqs=2, num_blocks=16, block_size=16)
        cache = scheduler.cache_manager
        self.assertTrue(cache.prefix_cache.enabled)
        source = Sequence(list(range(65)))
        cache.block_manager.ensure_capacity(source, 64)
        snapshot = GDNStateSnapshot(64, ())
        cache.publish_batch([(source, snapshot)])
        cache.block_manager.deallocate(source)

        seq = Sequence(list(range(64)) + [101], SamplingParams(max_tokens=1))
        entry = cache.find_prefix(seq)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.num_tokens, 64)
        self.assertEqual(cache.admission_blocks(seq, entry, set()), 5)

        cache.admit(seq, entry)
        self.assertEqual(seq.committed_tokens, 64)
        self.assertIs(seq.pending_state_snapshot, snapshot)
        self.assertEqual(len(seq.block_table), 4)
        self.assertGreaterEqual(seq.state_slot, 0)
        self.assertTrue(
            all(cache.block_manager.block_refcounts[block] == 2 for block in seq.block_table)
        )

        cache.release(seq)
        self.assertEqual(seq.committed_tokens, 0)
        self.assertIsNone(seq.pending_state_snapshot)
        self.assertEqual(seq.state_slot, -1)
        self.assertEqual(seq.block_table, [])
        cache.clear_prefixes()
        self.assertEqual(len(cache.block_manager.free_block_ids), cache.total_blocks)

    def test_reservation_counts_shared_pages_only_once(self):
        scheduler = make_scheduler(num_blocks=16, block_size=16, max_num_seqs=3)
        cache = scheduler.cache_manager
        one = Sequence([1] * 65, SamplingParams(max_tokens=1))
        two = Sequence([2] * 65, SamplingParams(max_tokens=1))
        cache.block_manager.ensure_capacity(one, 64)
        cache.block_manager.attach_shared_prefix(two, tuple(one.block_table), 64)
        occupied, reserved = cache.reservation_usage((one, two))
        self.assertEqual(len(occupied), 4)
        self.assertEqual(reserved, 6)

        cache.block_manager.deallocate(one)
        cache.block_manager.deallocate(two)

    def test_disabled_prefix_uses_request_reservations(self):
        scheduler = make_scheduler(num_blocks=16, block_size=16, max_prefix_cache_entries=0)
        cache = scheduler.cache_manager
        one = Sequence([1] * 65, SamplingParams(max_tokens=1))
        two = Sequence([2] * 33, SamplingParams(max_tokens=1))
        occupied, reserved = cache.reservation_usage((one, two))
        self.assertEqual(occupied, set())
        self.assertEqual(reserved, 8)

    def test_failed_gdn_slot_admission_releases_shared_kv_refs(self):
        scheduler = make_scheduler(max_num_seqs=1, num_blocks=16, block_size=16)
        cache = scheduler.cache_manager
        source = Sequence(list(range(65)))
        cache.block_manager.ensure_capacity(source, 64)
        cache.publish_batch([(source, GDNStateSnapshot(64, ()))])
        cache.block_manager.deallocate(source)

        first = Sequence([500], SamplingParams(max_tokens=1))
        cache.admit(first, None)
        seq = Sequence(list(range(64)) + [101], SamplingParams(max_tokens=1))
        entry = cache.find_prefix(seq)
        self.assertIsNotNone(entry)
        before = [cache.block_manager.block_refcounts[block] for block in entry.block_ids]
        with self.assertRaisesRegex(RuntimeError, "no free hybrid state slots"):
            cache.admit(seq, entry)
        self.assertEqual(seq.block_table, [])
        self.assertEqual(seq.committed_tokens, 0)
        self.assertIsNone(seq.pending_state_snapshot)
        self.assertEqual(
            [cache.block_manager.block_refcounts[block] for block in entry.block_ids], before
        )
        cache.release(first)

    def test_cache_admission_checks_aligned_processed_state(self):
        scheduler = make_scheduler(num_blocks=32, block_size=16)
        cache = scheduler.cache_manager
        seq = Sequence(list(range(30)), SamplingParams(max_tokens=3))
        cache.block_manager.ensure_capacity(seq, 32)
        self.assertFalse(cache.prefix_cache.can_publish(seq, 30))
        seq.append_token(30)
        seq.append_token(31)
        self.assertTrue(cache.prefix_cache.can_publish(seq, 32))
        cache.publish_batch([(seq, GDNStateSnapshot(32, ()))])
        self.assertFalse(cache.prefix_cache.can_publish(seq, 32))

    def test_prefill_evicts_only_enough_checkpoints_to_make_progress(self):
        scheduler = make_scheduler(num_blocks=16, block_size=16, max_prefix_cache_entries=4)
        manager = scheduler.cache_manager
        for token in (11, 12, 13, 14):
            source = Sequence([token] * 65)
            manager.block_manager.ensure_capacity(source, 64)
            manager.publish_batch([(source, GDNStateSnapshot(64, ()))])
            manager.release(source)
        self.assertEqual(len(manager.block_manager.free_block_ids), 0)
        self.assertEqual(len(manager.prefix_cache), 4)

        seq = Sequence(list(range(129)), SamplingParams(max_tokens=1))
        end = manager.prepare_prefill(seq, 0, 128)
        self.assertEqual(end, 64)
        self.assertEqual(manager.prefix_cache.evicted_count, 1)
        self.assertEqual(len(manager.prefix_cache), 3)


if __name__ == "__main__":
    unittest.main()
