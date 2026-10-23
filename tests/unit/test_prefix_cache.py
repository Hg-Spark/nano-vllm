import unittest
from hashlib import sha256
from struct import pack
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine.hybrid_cache_manager import HybridCacheManager
from nanovllm.engine.prefix_cache import _sequence_block_hashes
from nanovllm.engine.schedule import GDN_CHECKPOINT_INTERVAL
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


def make_cache_manager(
    *,
    num_blocks: int,
    block_size: int,
    max_prefix_entries: int,
    max_prefix_cache_snapshot_bytes: int | None = None,
    snapshot_nbytes: int = 0,
    max_checkpoints_per_path: int = 4,
) -> HybridCacheManager:
    """Exercise the production cache owner instead of the removed legacy mode."""
    config = SimpleNamespace(
        kv_cache_block_size=block_size,
        max_num_seqs=8,
        max_prefix_cache_entries=max_prefix_entries,
        max_prefix_cache_snapshot_bytes=max_prefix_cache_snapshot_bytes,
        max_gdn_checkpoints=max_checkpoints_per_path,
    )
    return HybridCacheManager(config, num_blocks, snapshot_nbytes)


def _block_hashes(tokens: list[int] | tuple[int, ...], block_size: int, stop: int):
    """Independent chained SHA-256 oracle; production uses incremental Sequence hashes."""
    parent = bytes(32)
    for start in range(0, min(stop, len(tokens)) - block_size + 1, block_size):
        parent = sha256(
            parent + pack(f"<{block_size}I", *tokens[start : start + block_size])
        ).digest()
        yield start + block_size, parent


def publish_snapshot(
    manager: HybridCacheManager,
    seq: Sequence,
    prefix_tokens: int,
    snapshot: GDNStateSnapshot | None = None,
) -> None:
    cache = manager.prefix_cache
    if not cache._can_capture():
        return
    snapshot = snapshot or GDNStateSnapshot(prefix_tokens, ())
    if prefix_tokens % GDN_CHECKPOINT_INTERVAL == 0:
        cache.publish_many(seq, (snapshot,))
        return
    # Small-block eviction fixtures exercise the private policy below the GDN grid.
    path = []
    for position, digest in _block_hashes(seq.token_ids, cache.block_size, prefix_tokens):
        if digest in cache._entries:
            path.append(digest)
    if position != prefix_tokens:
        raise ValueError("test prefix must end on a KV block boundary")
    cache._publish(seq, snapshot, prefix_tokens, digest, path)


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
        manager = make_cache_manager(num_blocks=12, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(130)))
        cache = manager.prefix_cache

        # An empty cache must skip hashing entirely.
        cache.find(seq, max_tokens=64)
        self.assertEqual(seq.prefix_block_hashes, [])
        cache.candidates(seq, 0, 64)
        first = tuple(seq.prefix_block_hashes)
        self.assertEqual(len(first), 4)

        cache.candidates(seq, 64, 128)
        self.assertEqual(tuple(seq.prefix_block_hashes[:4]), first)
        self.assertEqual(len(seq.prefix_block_hashes), 8)

        for token in range(130, 144):
            seq.append_token(token)
        cache.candidates(seq, 128, 144)
        # Appended generated tokens are not part of the reusable prompt.
        self.assertEqual(len(seq.prefix_block_hashes), 8)
        _sequence_block_hashes(seq, 16, 144)
        self.assertEqual(len(seq.prefix_block_hashes), 9)
        self.assertEqual(
            seq.prefix_block_hashes,
            [digest for _, digest in _block_hashes(seq.token_ids, 16, 144)],
        )

    def test_incremental_hashing_only_processes_new_complete_pages(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(162)))
        cache = manager.prefix_cache

        with patch("nanovllm.engine.prefix_cache.sha256", wraps=sha256) as hash_fn:
            self.assertIsNone(cache.find(seq, max_tokens=128))
            self.assertEqual(hash_fn.call_count, 0)

            candidates, cached = cache.candidates(seq, 64, 160)
            self.assertEqual([position for position, _ in candidates], [128])
            self.assertEqual(cached, [])
            self.assertEqual(hash_fn.call_count, 10)

            # Previously calculated blocks are never hashed again.
            self.assertIsNone(cache.find(seq, max_tokens=64))
            self.assertIsNone(cache.find(seq, max_tokens=159))
            self.assertEqual(hash_fn.call_count, 10)

            # The incomplete final block is never hashed.
            self.assertIsNone(cache.find(seq, max_tokens=162))
            self.assertEqual(hash_fn.call_count, 10)

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

    def test_checkpoint_candidates_align_to_kv_page_and_gdn_grid(self):
        for block_size, expected_hashes in ((32, 12), (64, 6)):
            with self.subTest(block_size=block_size):
                manager = make_cache_manager(
                    num_blocks=16, block_size=block_size, max_prefix_entries=4
                )
                seq = Sequence(list(range(401)))
                candidates, cached = manager.prefix_cache.candidates(seq, 0, 400)
                self.assertEqual(
                    [position for position, _ in candidates], [64, 128, 192, 256, 320, 384]
                )
                self.assertEqual(cached, [])
                self.assertEqual(len(seq.prefix_block_hashes), expected_hashes)

    def test_candidate_scan_only_probes_new_boundaries(self):
        manager = make_cache_manager(num_blocks=8, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(8193)))
        cache = manager.prefix_cache
        cache.find(seq, max_tokens=8192)  # Warm the block hashes.

        class CountingEntries(dict):
            def __init__(self, entries):
                super().__init__(entries)
                self.probes = 0

            def __contains__(self, digest):
                self.probes += 1
                return super().__contains__(digest)

        cache._entries = CountingEntries(cache._entries)
        candidates, cached = cache.candidates(seq, 8128, 8192)
        self.assertEqual([position for position, _ in candidates], [8192])
        self.assertEqual(cached, [])
        self.assertEqual(cache._entries.probes, 1)

    def test_publish_probes_resident_ancestors_not_every_historical_page(self):
        manager = make_cache_manager(num_blocks=520, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(8193)))
        manager.block_manager.ensure_capacity(seq, 8192)
        cache = manager.prefix_cache
        cache.publish_many(seq, (GDNStateSnapshot(64, ()),))

        class CountingEntries(dict):
            def __init__(self, entries):
                super().__init__(entries)
                self.probes = 0

            def __contains__(self, key):
                self.probes += 1
                return super().__contains__(key)

        cache._entries = CountingEntries(cache._entries)
        cache.publish_many(seq, (GDNStateSnapshot(8192, ()),))
        self.assertEqual(cache.find(seq, 8192).num_tokens, 8192)
        self.assertLessEqual(cache._entries.probes, 8)

    def test_cached_checkpoints_include_ancestors_and_exclude_other_branches(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(257)))
        manager.block_manager.ensure_capacity(seq, 192)
        cache = manager.prefix_cache
        cache.publish_many(
            seq,
            (GDNStateSnapshot(64, ()), GDNStateSnapshot(192, ())),
        )

        candidates, cached = cache.candidates(seq, 128, 256)
        self.assertEqual(cached, [64, 192])
        self.assertEqual([position for position, _ in candidates], [256])

        diverged = Sequence(seq.token_ids[:80] + [999] + seq.token_ids[81:])
        candidates, cached = cache.candidates(diverged, 128, 256)
        self.assertEqual(cached, [64])
        self.assertEqual([position for position, _ in candidates], [192, 256])

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
        self.assertEqual(cache.peak_snapshot_bytes, 512)
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

    def test_global_capacity_replaces_oldest_entry(self):
        manager = make_cache_manager(num_blocks=16, block_size=16, max_prefix_entries=1)
        old, old_block = publish_prefix(manager, [1] * 64 + [9], 64)
        publish_prefix(manager, list(range(128)) + [9], 128)

        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(
            manager.prefix_cache.find(Sequence(list(range(128)) + [10]), 128).num_tokens,
            128,
        )
        self.assertEqual(manager.block_manager.block_refcounts[old_block], 1)
        manager.release(old)

    def test_global_one_entry_captures_only_deepest_checkpoint(self):
        manager = make_cache_manager(num_blocks=20, block_size=16, max_prefix_entries=1)
        seq = Sequence(list(range(256)))
        candidates, cached = manager.prefix_cache.candidates(seq, 0, 256)
        self.assertEqual(
            [
                position
                for position, _ in manager.prefix_cache.select_checkpoints(candidates, cached)
            ],
            [256],
        )

        manager = make_cache_manager(num_blocks=96, block_size=16, max_prefix_entries=1)
        seq = Sequence(list(range(1024)) + [9])
        manager.block_manager.ensure_capacity(seq, 1024)
        publish_snapshot(manager, seq, 512)
        candidates, cached = manager.prefix_cache.candidates(seq, 512, 1024)
        self.assertEqual(
            [p for p, _ in manager.prefix_cache.select_checkpoints(candidates, cached)],
            [1024],
        )
        publish_snapshot(manager, seq, 1024)
        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(manager.prefix_cache.find(seq, 1024).num_tokens, 1024)

        shorter = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=1)
        original = Sequence(list(range(128)) + [9])
        shorter.block_manager.ensure_capacity(original, 128)
        publish_snapshot(shorter, original, 128)
        other = Sequence(list(range(1, 193)) + [9])
        candidates, cached = shorter.prefix_cache.candidates(other, 0, 192)
        self.assertEqual(
            [p for p, _ in shorter.prefix_cache.select_checkpoints(candidates, cached)],
            [192],
        )

    def test_backfilled_shorter_checkpoint_does_not_delete_longer(self):
        manager = make_cache_manager(
            num_blocks=16, block_size=16, max_prefix_entries=2, max_checkpoints_per_path=1
        )
        seq = Sequence(list(range(128)) + [9])
        manager.block_manager.ensure_capacity(seq, 128)
        publish_snapshot(manager, seq, 128)
        publish_snapshot(manager, seq, 64)
        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(manager.prefix_cache.find(seq, 128).num_tokens, 128)

        manager = make_cache_manager(
            num_blocks=32, block_size=16, max_prefix_entries=4, max_checkpoints_per_path=2
        )
        seq = Sequence(list(range(256)) + [9])
        manager.block_manager.ensure_capacity(seq, 256)
        for position in (64, 256, 128):
            publish_snapshot(manager, seq, position)
        self.assertEqual(len(manager.prefix_cache), 2)
        self.assertEqual(manager.prefix_cache.find(seq, 256).num_tokens, 256)

    def test_future_descendant_rejects_wasted_checkpoint_capture(self):
        manager = make_cache_manager(
            num_blocks=48,
            block_size=16,
            max_prefix_entries=4,
            max_checkpoints_per_path=2,
        )
        full = Sequence(list(range(321)))
        manager.block_manager.ensure_capacity(full, 320)
        publish_snapshot(manager, full, 64)
        publish_snapshot(manager, full, 320)
        cache = manager.prefix_cache
        stats = (cache.published_count, cache.evicted_count, cache.snapshot_bytes)
        short = Sequence(list(range(193)))
        end, positions, interval = manager.plan_prefill_checkpoints(short, 64, 193, set())
        self.assertEqual((end, positions, interval), (193, (), 0))
        manager.block_manager.ensure_capacity(short, 192)
        publish_snapshot(manager, short, 192)
        self.assertEqual((cache.published_count, cache.evicted_count, cache.snapshot_bytes), stats)
        self.assertEqual(cache.find(full, 320).num_tokens, 320)

    def test_failed_backfill_is_atomic_across_sibling_branches(self):
        manager = make_cache_manager(
            num_blocks=64,
            block_size=16,
            max_prefix_entries=8,
            max_checkpoints_per_path=3,
        )
        shared = list(range(192))
        branch_a = Sequence(shared + list(range(1000, 1129)))
        branch_b = Sequence(shared + list(range(2000, 2129)))
        for seq in (branch_a, branch_b):
            manager.block_manager.ensure_capacity(seq, 320)
        cache = manager.prefix_cache
        for position in (64, 256, 320):
            publish_snapshot(manager, branch_a, position)
        for position in (256, 320):
            publish_snapshot(manager, branch_b, position)
        reused = cache.find(branch_b, 256)
        cache.mark_reused(branch_b, reused)
        original = (
            set(cache._entries),
            cache.published_count,
            cache.evicted_count,
            tuple(cache._resident_lengths),
        )
        publish_snapshot(manager, branch_a, 192)
        self.assertEqual(
            (
                set(cache._entries),
                cache.published_count,
                cache.evicted_count,
                tuple(cache._resident_lengths),
            ),
            original,
        )
        self.assertEqual(cache.find(branch_a, 256).num_tokens, 256)
        self.assertEqual(cache.find(branch_b, 256).num_tokens, 256)
        cache.unmark_reused(branch_b)

    def test_global_pressure_prefers_other_branch_over_own_ancestor(self):
        manager = make_cache_manager(
            num_blocks=24,
            block_size=16,
            max_prefix_entries=2,
        )
        branch_a = Sequence(list(range(129)))
        branch_b = Sequence([999] * 65)
        manager.block_manager.ensure_capacity(branch_a, 128)
        manager.block_manager.ensure_capacity(branch_b, 64)
        cache = manager.prefix_cache
        publish_snapshot(manager, branch_a, 64)
        publish_snapshot(manager, branch_b, 64)
        publish_snapshot(manager, branch_a, 128)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.find(branch_a, 64).num_tokens, 64)
        self.assertEqual(cache.find(branch_a, 128).num_tokens, 128)
        self.assertIsNone(cache.find(branch_b, 64))

    def test_batch_publish_rejects_misaligned_boundary(self):
        manager = make_cache_manager(num_blocks=10, block_size=16, max_prefix_entries=2)
        seq = Sequence(list(range(128)))
        with self.assertRaisesRegex(RuntimeError, "invalid GDN snapshot"):
            manager.prefix_cache.publish_many(seq, (GDNStateSnapshot(63, ()),))

    def test_full_budget_refreshes_deepest_checkpoint_on_same_path(self):
        manager = make_cache_manager(num_blocks=24, block_size=16, max_prefix_entries=2)
        seq = Sequence(list(range(192)) + [9])
        manager.block_manager.ensure_capacity(seq, 192)
        cache = manager.prefix_cache
        cache.publish_many(seq, (GDNStateSnapshot(64, ()), GDNStateSnapshot(128, ())))
        candidates, cached = cache.candidates(seq, 128, 192)
        self.assertEqual(
            [p for p, _ in cache.select_checkpoints(candidates, cached)],
            [192],
        )
        cache.publish_many(seq, (GDNStateSnapshot(192, ()),))
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.find(seq, 192).num_tokens, 192)
        self.assertEqual(cache.find(seq, 64).num_tokens, 64)

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
        self.assertFalse(manager.prefix_cache.candidates(seq, 0, 4)[0])

    def test_prefix_cache_only_caches_text_before_image_span(self):
        manager = make_cache_manager(num_blocks=32, block_size=4, max_prefix_entries=4)
        seq = Sequence(list(range(81)))
        seq.state_slot = 0
        seq.image_state = SimpleNamespace(image_start=64, image_end=72)
        manager.block_manager.ensure_capacity(seq, 72)

        self.assertEqual(
            [position for position, _ in manager.prefix_cache.candidates(seq, 0, 80)[0]],
            [64],
        )
        self.assertEqual(len(manager.prefix_cache), 0)

    def test_publish_rejects_image_region_even_if_called_directly(self):
        manager = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(128)))
        seq.image_state = SimpleNamespace(image_start=64, image_end=96)
        manager.block_manager.ensure_capacity(seq, 128)
        cache = manager.prefix_cache
        with self.assertRaisesRegex(RuntimeError, "invalid GDN snapshot"):
            cache.publish_many(seq, (GDNStateSnapshot(128, ()),))
        self.assertEqual(len(cache), 0)

    def test_single_slot_prefers_checkpoint_reusable_by_identical_prompt(self):
        manager = make_cache_manager(num_blocks=32, block_size=16, max_prefix_entries=1)
        original = Sequence(list(range(128)))
        manager.block_manager.ensure_capacity(original, 128)
        end, positions, _ = manager.plan_prefill_checkpoints(original, 0, 128, set())
        self.assertEqual(end, 128)
        self.assertEqual(positions, (64,))
        manager.prefix_cache.publish_many(original, (GDNStateSnapshot(64, ()),))
        repeated = Sequence(list(range(128)))
        self.assertEqual(manager.find_prefix(repeated).num_tokens, 64)

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

    def test_shared_history_reuses_snapshots_and_preserves_shortest(self):
        manager = make_cache_manager(
            num_blocks=64,
            block_size=16,
            max_prefix_entries=8,
            max_checkpoints_per_path=4,
        )
        original = Sequence(list(range(320)))
        # Both requests share 256 tokens, then diverge on separate branches.
        longer = Sequence(list(range(256)) + list(range(1000, 1272)))
        manager.block_manager.ensure_capacity(original, 256)
        manager.block_manager.ensure_capacity(longer, 512)

        history = {}
        for position in (64, 128, 192, 256):
            snapshot = GDNStateSnapshot(position, ())
            history[position] = snapshot
            publish_snapshot(manager, original, position, snapshot)

        for position in (128, 256):
            publish_snapshot(manager, longer, position)
        self.assertEqual(len(manager.prefix_cache), 4)
        reused = manager.prefix_cache.find(longer, max_tokens=256)
        self.assertIs(reused.state_snapshot, history[256])
        # An admitted prefix hit protects the reused checkpoint on this path.
        manager.prefix_cache.mark_reused(longer, reused)

        for position in (384, 448, 512):
            publish_snapshot(manager, longer, position)
        self.assertEqual(len(manager.prefix_cache), 4)
        self.assertEqual(manager.prefix_cache.find(longer, max_tokens=527).num_tokens, 512)
        self.assertEqual(manager.prefix_cache.find(original, max_tokens=319).num_tokens, 256)
        self.assertEqual(manager.prefix_cache.find(original, max_tokens=80).num_tokens, 64)
        self.assertEqual(manager.block_manager.block_refcounts[original.block_table[0]], 3)


if __name__ == "__main__":
    unittest.main()
