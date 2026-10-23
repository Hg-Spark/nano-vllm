import unittest
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine.hybrid_cache_manager import HybridCacheManager
from nanovllm.engine.prefix_cache import _sequence_block_hashes
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


def make_cache_manager(
    *,
    num_blocks: int,
    block_size: int,
    max_prefix_entries: int,
    max_prefix_cache_snapshot_bytes: int | None = None,
    snapshot_nbytes: int = 0,
) -> HybridCacheManager:
    """Exercise the production cache owner instead of the removed legacy mode."""
    config = SimpleNamespace(
        kv_cache_block_size=block_size,
        max_num_seqs=8,
        max_prefix_cache_entries=max_prefix_entries,
        max_prefix_cache_snapshot_bytes=max_prefix_cache_snapshot_bytes,
    )
    return HybridCacheManager(config, num_blocks, snapshot_nbytes)


def publish_snapshot(
    manager: HybridCacheManager,
    seq: Sequence,
    prefix_tokens: int,
    snapshot: GDNStateSnapshot | None = None,
) -> None:
    snapshot = snapshot or GDNStateSnapshot(prefix_tokens, ())
    manager.publish_batch([(seq, snapshot)])


def publish_prefix(
    manager: HybridCacheManager,
    tokens: list[int],
    prefix_tokens: int,
    snapshot: GDNStateSnapshot | None = None,
) -> tuple[Sequence, int]:
    seq = Sequence(tokens)
    manager.block_manager.ensure_capacity(seq, prefix_tokens)
    seq.committed_tokens = 0
    manager.state_manager.allocate(seq)
    block_id = seq.block_table[0]
    publish_snapshot(manager, seq, prefix_tokens, snapshot)
    return seq, block_id


class PrefixCacheTest(unittest.TestCase):
    def test_complete_block_hashes_are_reused_after_append(self):
        seq = Sequence(list(range(65)))
        hashes, num_blocks = _sequence_block_hashes(seq, 16, 64)
        self.assertEqual(num_blocks, 4)
        before = tuple(hashes)
        seq.append_token(66)
        hashes, num_blocks = _sequence_block_hashes(seq, 16, 66)
        self.assertEqual(num_blocks, 4)
        self.assertEqual(tuple(hashes), before)

    def test_find_uses_resident_lengths_instead_of_scanning_missing_pages(self):
        manager = make_cache_manager(num_blocks=64, block_size=16, max_prefix_entries=4)
        prefix = [index % 100 for index in range(128)]
        publish_prefix(manager, prefix + [900], 64)
        cache = manager.prefix_cache
        self.assertEqual(cache._resident_lengths, [64])

        class CountingEntries(dict):
            def __init__(self, entries):
                super().__init__(entries)
                self.probes = 0

            def get(self, key, default=None):
                self.probes += 1
                return super().get(key, default)

        cache._entries = CountingEntries(cache._entries)
        long_seq = Sequence(prefix + list(range(128, 4096)))
        self.assertEqual(cache.find(long_seq, 4095).num_tokens, 64)
        self.assertEqual(cache._entries.probes, 1)
        self.assertEqual(len(long_seq.prefix_block_hashes), 4)

        other = Sequence([42] * 4096)
        self.assertIsNone(cache.find(other, 4095))
        self.assertEqual(cache._entries.probes, 2)

        self.assertTrue(cache.evict())
        self.assertEqual(cache._resident_lengths, [])
        with patch("nanovllm.engine.prefix_cache.sha256", wraps=sha256) as hash_fn:
            self.assertIsNone(cache.find(Sequence(list(range(4096))), 4095))
            hash_fn.assert_not_called()

    def test_resident_length_index_keeps_sibling_checkpoints(self):
        manager = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=4)
        left, _ = publish_prefix(manager, [1] * 64 + [9], 64)
        publish_prefix(manager, [2] * 64 + [9], 64)
        cache = manager.prefix_cache
        self.assertEqual(cache._resident_lengths, [64])
        self.assertEqual(cache._resident_length_counts[64], 2)
        self.assertTrue(cache.evict())
        self.assertEqual(cache._resident_lengths, [64])
        self.assertEqual(cache._resident_length_counts[64], 1)
        self.assertTrue(cache.evict())
        self.assertEqual(cache._resident_lengths, [])

    def test_longest_exact_prefix_and_max_tokens_cap(self):
        manager = make_cache_manager(num_blocks=8, block_size=4, max_prefix_entries=4)
        publish_prefix(manager, [1, 2, 3, 4, 9], 4)
        publish_prefix(manager, [1, 2, 3, 4, 5, 6, 7, 8, 9], 8)

        hit = manager.prefix_cache.find(Sequence([1, 2, 3, 4, 5, 6, 7, 8, 10]), max_tokens=8)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.num_tokens, 8)
        self.assertIsNone(manager.prefix_cache.find(Sequence([1, 2, 3, 4]), max_tokens=3))

    def test_eviction_releases_blocks_under_entry_and_byte_limits(self):
        for byte_limit in (None, 12):
            with self.subTest(byte_limit=byte_limit):
                manager = make_cache_manager(
                    num_blocks=8,
                    block_size=4,
                    max_prefix_entries=1 if byte_limit is None else 4,
                    max_prefix_cache_snapshot_bytes=byte_limit,
                    snapshot_nbytes=9 if byte_limit is not None else 0,
                )
                first, first_block = publish_prefix(manager, [1, 2, 3, 4, 9], 4)
                self.assertEqual(manager.block_manager.block_refcounts[first_block], 2)
                second, second_block = publish_prefix(manager, [5, 6, 7, 8, 9], 4)
                cache = manager.prefix_cache

                self.assertEqual(len(cache), 1)
                self.assertEqual(cache.snapshot_bytes, 9 if byte_limit is not None else 0)
                self.assertEqual(manager.block_manager.block_refcounts[first_block], 1)
                self.assertEqual(manager.block_manager.block_refcounts[second_block], 2)
                hit = cache.find(Sequence([5, 6, 7, 8, 10]), max_tokens=4)
                self.assertEqual(hit.block_ids, (second_block,))

                manager.release(first)
                manager.release(second)
                self.assertEqual(manager.block_manager.block_refcounts[first_block], 0)
                self.assertTrue(cache.evict())
                self.assertEqual(cache.snapshot_bytes, 0)
                self.assertEqual(manager.block_manager.block_refcounts[second_block], 0)

    def test_checkpoint_publish_eviction_and_peak_snapshot_counters(self):
        manager = make_cache_manager(
            num_blocks=12,
            block_size=16,
            max_prefix_entries=1,
            snapshot_nbytes=256,
        )
        cache = manager.prefix_cache
        self.assertEqual(
            (cache.published_count, cache.evicted_count, cache.peak_snapshot_bytes), (0, 0, 0)
        )

        publish_prefix(manager, [1] * 64 + [9], 64)
        self.assertEqual((cache.published_count, cache.evicted_count), (1, 0))
        publish_prefix(manager, [2] * 64 + [9], 64)
        self.assertEqual((cache.published_count, cache.evicted_count), (2, 1))
        self.assertEqual(cache.peak_snapshot_bytes, 256)
        self.assertEqual(cache.snapshot_bytes, 256)
        self.assertEqual(len(cache), 1)
        self.assertTrue(cache.evict())
        self.assertEqual((cache.evicted_count, cache.snapshot_bytes, len(cache)), (2, 0, 0))

    def test_global_eviction_admits_new_branch(self):
        manager = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=2)
        shared = list(range(128))
        original, _ = publish_prefix(manager, shared + [200], 64)
        manager.block_manager.ensure_capacity(original, 128)
        publish_snapshot(manager, original, 128)
        publish_prefix(manager, [300] * 64 + [301], 64)

        self.assertEqual(len(manager.prefix_cache), 2)
        self.assertEqual(
            manager.prefix_cache.find(Sequence(shared + [202]), max_tokens=128).num_tokens,
            128,
        )
        self.assertIsNotNone(manager.prefix_cache.find(Sequence([300] * 64 + [302]), max_tokens=64))

    def test_batch_publish_rejects_misaligned_boundary(self):
        manager = make_cache_manager(num_blocks=10, block_size=16, max_prefix_entries=2)
        seq = Sequence(list(range(128)))
        with self.assertRaisesRegex(RuntimeError, "invalid GDN snapshot"):
            manager.prefix_cache.publish_batch([(seq, GDNStateSnapshot(63, ()))])

    def test_full_budget_refreshes_deepest_checkpoint_on_same_path(self):
        manager = make_cache_manager(num_blocks=24, block_size=16, max_prefix_entries=2)
        seq = Sequence(list(range(192)) + [9])
        manager.block_manager.ensure_capacity(seq, 192)
        cache = manager.prefix_cache
        cache.publish_batch([(seq, GDNStateSnapshot(64, ())), (seq, GDNStateSnapshot(128, ()))])
        cache.publish_batch([(seq, GDNStateSnapshot(192, ()))])
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.find(seq, 192).num_tokens, 192)
        self.assertIsNone(cache.find(seq, 64))
        self.assertEqual(cache.find(seq, 128).num_tokens, 128)

    def test_cache_maintenance_does_not_evict_active_shared_blocks(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=2)
        first, _ = publish_prefix(manager, [1] * 65, 64)
        second, _ = publish_prefix(manager, [2] * 65, 64)
        cache = manager.prefix_cache
        self.assertFalse(cache.evict(manager.block_manager.block_refcounts))
        self.assertEqual(len(cache), 2)
        manager.release(first)
        self.assertTrue(cache.evict(manager.block_manager.block_refcounts))
        self.assertEqual(len(cache), 1)
        manager.release(second)

    def test_find_is_read_only_until_prefix_is_attached(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=2)
        original, _ = publish_prefix(manager, [1] * 65, 64)
        cache = manager.prefix_cache
        newcomer = Sequence([1] * 64 + [2])
        hit = cache.find(newcomer, max_tokens=64)
        self.assertEqual(cache._active_reuses, {})
        manager.block_manager.attach_shared_prefix(newcomer, hit.block_ids, hit.num_tokens)
        cache.mark_reused(newcomer, hit)
        self.assertEqual(sum(cache._active_reuses.values()), 1)
        manager.release(original)
        manager.release(newcomer)
        self.assertEqual(cache._active_reuses, {})
        self.assertEqual(cache._seq_reuses, {})

    def test_reuse_reference_counts_and_eviction_survives_live_requests(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=4)
        original, _ = publish_prefix(manager, [1] * 65, 64)
        cache = manager.prefix_cache
        entry = cache.find(original, max_tokens=64)
        consumers = [Sequence([1] * 64 + [token]) for token in (2, 3)]
        for seq in consumers:
            manager.block_manager.attach_shared_prefix(seq, entry.block_ids, entry.num_tokens)
            cache.mark_reused(seq, entry)
        digest = entry.block_hashes[-1]
        self.assertEqual(cache._active_reuses[digest], 2)

        manager.release(consumers[0])
        self.assertEqual(cache._active_reuses[digest], 1)
        # Eviction releases only the cache owner; live request KV refs remain valid.
        self.assertTrue(cache.evict())
        self.assertNotIn(digest, cache._entries)
        self.assertEqual(cache._active_reuses[digest], 1)
        manager.release(consumers[1])
        self.assertEqual(cache._active_reuses, {})
        self.assertEqual(cache._seq_reuses, {})
        manager.release(original)
        self.assertEqual(
            len(manager.block_manager.free_block_ids), manager.block_manager.num_blocks
        )

    def test_unmark_reused_is_idempotent(self):
        manager = make_cache_manager(num_blocks=8, block_size=16, max_prefix_entries=2)
        seq = Sequence([1] * 65)
        manager.release(seq)
        manager.release(seq)
        self.assertEqual(manager.prefix_cache._active_reuses, {})

    def test_zero_snapshot_budget_skips_publication(self):
        manager = make_cache_manager(
            num_blocks=4,
            block_size=4,
            max_prefix_entries=4,
            max_prefix_cache_snapshot_bytes=0,
        )
        seq, block_id = publish_prefix(manager, [1, 2, 3, 4, 5], 4)
        self.assertEqual(len(manager.prefix_cache), 0)
        self.assertEqual(manager.prefix_cache.snapshot_bytes, 0)
        self.assertEqual(manager.block_manager.block_refcounts[block_id], 1)
        self.assertFalse(manager.prefix_cache._can_capture())

    def test_publish_rejects_image_region_even_if_called_directly(self):
        manager = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(128)))
        seq.image_state = SimpleNamespace(image_start=64, image_end=96)
        manager.block_manager.ensure_capacity(seq, 128)
        cache = manager.prefix_cache
        with self.assertRaisesRegex(RuntimeError, "invalid GDN snapshot"):
            cache.publish_batch([(seq, GDNStateSnapshot(128, ()))])
        self.assertEqual(len(cache), 0)

    def test_global_eviction_prefers_unshared_checkpoint(self):
        manager = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=2)
        first, _ = publish_prefix(manager, [1] * 65, 64)
        second, _ = publish_prefix(manager, [2] * 65, 64)
        cache = manager.prefix_cache
        shared = cache.find(Sequence([1] * 65), max_tokens=64)
        consumer = Sequence([1] * 64 + [5])
        manager.block_manager.attach_shared_prefix(consumer, shared.block_ids, shared.num_tokens)
        cache.mark_reused(consumer, shared)
        self.assertTrue(cache.evict())
        self.assertIn(shared.block_hashes[-1], cache._entries)
        self.assertIsNone(cache.find(Sequence([2] * 65), max_tokens=64))
        manager.release(first)
        manager.release(second)
        manager.release(consumer)

    def test_global_capacity_not_per_path_quota(self):
        manager = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(257)))
        manager.block_manager.ensure_capacity(seq, 256)
        for position in (64, 128, 192, 256):
            publish_snapshot(manager, seq, position)
        cache = manager.prefix_cache
        self.assertEqual(len(cache), 4)
        self.assertEqual(cache._resident_lengths, [64, 128, 192, 256])
        self.assertEqual(cache.find(seq, 64).num_tokens, 64)
        self.assertEqual(cache.find(seq, 256).num_tokens, 256)

    def test_out_of_order_batch_publication_preserves_resident_index(self):
        manager = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(257)))
        manager.block_manager.ensure_capacity(seq, 256)
        snapshots = (
            GDNStateSnapshot(256, ()),
            GDNStateSnapshot(64, ()),
            GDNStateSnapshot(128, ()),
        )
        manager.prefix_cache.publish_batch([(seq, snapshot) for snapshot in snapshots])
        cache = manager.prefix_cache
        self.assertEqual(cache._resident_lengths, [64, 128, 256])
        self.assertIs(cache.find(seq, 256).state_snapshot, snapshots[0])
        self.assertIs(cache.find(seq, 64).state_snapshot, snapshots[1])

    def test_completed_generation_can_cache_tokens_beyond_original_prompt(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=2)
        seq = Sequence(list(range(30)))
        manager.block_manager.ensure_capacity(seq, 32)
        seq.append_token(30)
        seq.append_token(31)
        seq.append_token(32)  # Last sampled token has not been forwarded.
        manager.prefix_cache.publish_batch([(seq, GDNStateSnapshot(32, ()))])
        hit = manager.prefix_cache.find(Sequence(list(range(33)) + [99]), 33)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.num_tokens, 32)
        self.assertIsNone(manager.prefix_cache.find(Sequence(list(range(32)) + [99]), 31))

    def test_active_cache_entry_blocks_eviction_during_publication(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=1)
        original, _ = publish_prefix(manager, [1] * 65, 64)
        cache = manager.prefix_cache
        hit = cache.find(Sequence([1] * 65), 64)
        consumer = Sequence([1] * 64 + [2])
        manager.block_manager.attach_shared_prefix(consumer, hit.block_ids, 64)
        cache.mark_reused(consumer, hit)
        other = Sequence([2] * 65)
        manager.block_manager.ensure_capacity(other, 64)
        self.assertIsNotNone(cache.find(consumer, 64))
        before = cache.published_count
        cache.publish_batch([(other, GDNStateSnapshot(64, ()))])
        self.assertEqual(cache.published_count, before)
        manager.release(original)
        manager.release(consumer)
        manager.release(other)

    def test_batch_publication_deduplicates_identical_hashes(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=4)
        first, _ = publish_prefix(manager, [1] * 17, 16)
        cache = manager.prefix_cache
        second = Sequence([2] * 17)
        third = Sequence([2] * 17)
        manager.block_manager.ensure_capacity(second, 16)
        manager.block_manager.ensure_capacity(third, 16)
        cache.publish_batch(
            [
                (second, GDNStateSnapshot(16, ())),
                (third, GDNStateSnapshot(16, ())),
                (first, GDNStateSnapshot(16, ())),
            ]
        )
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.published_count, 2)
        self.assertEqual(cache._resident_length_counts[16], 2)
        self.assertIsNotNone(cache.find(Sequence([2] * 17), 16))

    def test_batch_publication_limits_survivors_to_capacity(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=1)
        cache = manager.prefix_cache
        candidates = []
        for value in (10, 11, 12):
            seq = Sequence([value] * 16)
            manager.block_manager.ensure_capacity(seq, 16)
            candidates.append((seq, GDNStateSnapshot(16, ())))
        cache.publish_batch(candidates)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.published_count, 1)
        for seq, _ in candidates[:-1]:
            self.assertIsNone(cache.find(seq, 16))
        self.assertIsNotNone(cache.find(candidates[-1][0], 16))

    def test_conversation_completion_can_evict_idle_prompt(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=1)
        cache = manager.prefix_cache
        prompt = Sequence([5] * 16)
        manager.block_manager.ensure_capacity(prompt, 16)
        cache.publish_batch([(prompt, GDNStateSnapshot(16, ()))])
        convo = Sequence([7] * 15)
        manager.block_manager.ensure_capacity(convo, 16)
        convo.append_token(8)
        self.assertEqual(cache.published_count, 1)
        cache.publish_batch([(convo, GDNStateSnapshot(16, ()))])
        self.assertIsNone(cache.find(Sequence([5] * 16 + [3]), 16))
        self.assertIsNotNone(cache.find(Sequence([7] * 15 + [8, 3]), 16))
        self.assertEqual(len(cache), 1)

    def test_batch_publication_last_candidate_wins_when_capacity_one(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=1)
        cache = manager.prefix_cache
        convo = Sequence([7] * 15)
        manager.block_manager.ensure_capacity(convo, 16)
        convo.append_token(8)
        prompt = Sequence([5] * 16)
        manager.block_manager.ensure_capacity(prompt, 16)
        cache.publish_batch([(convo, GDNStateSnapshot(16, ())), (prompt, GDNStateSnapshot(16, ()))])
        self.assertEqual(len(cache), 1)
        self.assertIsNone(cache.find(convo, 16))
        self.assertIsNotNone(cache.find(prompt, 16))

    def test_batch_commit_matches_virtual_survivors_when_old_entry_reappears(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=2)
        cache = manager.prefix_cache
        old_a, _ = publish_prefix(manager, [1] * 17, 16)
        old_b, _ = publish_prefix(manager, [2] * 17, 16)
        new_c = Sequence([3] * 17)
        new_d = Sequence([4] * 17)
        for seq in (new_c, new_d):
            manager.block_manager.ensure_capacity(seq, 16)

        proposals = [(new_c, 16), (old_a, 16), (new_d, 16)]
        cache.publish_batch([(seq, GDNStateSnapshot(position, ())) for seq, position in proposals])

        self.assertIsNotNone(cache.find(Sequence([1] * 17), 16))
        self.assertIsNotNone(cache.find(Sequence([4] * 17), 16))
        self.assertIsNone(cache.find(Sequence([2] * 17), 16))
        self.assertIsNone(cache.find(Sequence([3] * 17), 16))
        self.assertEqual(len(cache), 2)
        for seq in (old_a, old_b, new_c, new_d):
            manager.release(seq)


if __name__ == "__main__":
    unittest.main()
