import unittest
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.prefix_cache import _block_hashes
from nanovllm.engine.schedule import GDN_CHECKPOINT_INTERVAL
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


def publish_snapshot(
    manager: BlockManager,
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
    manager: BlockManager,
    tokens: list[int],
    prefix_tokens: int,
    snapshot: GDNStateSnapshot | None = None,
) -> tuple[Sequence, int]:
    seq = Sequence(tokens)
    manager.ensure_capacity(seq, prefix_tokens)
    seq.committed_tokens = 0
    seq.state_slot = 0
    block_id = seq.block_table[0]
    publish_snapshot(manager, seq, prefix_tokens, snapshot)
    return seq, block_id


class BlockManagerPrefixCacheTest(unittest.TestCase):
    def test_complete_block_hashes_are_reused_after_append(self):
        manager = BlockManager(num_blocks=12, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(130)))
        cache = manager.prefix_cache

        cache.find(seq, max_tokens=64)
        first = tuple(seq.prefix_block_hashes)
        self.assertEqual(len(first), 4)

        cache.candidates(seq, 64, 128)
        self.assertEqual(tuple(seq.prefix_block_hashes[:4]), first)
        self.assertEqual(len(seq.prefix_block_hashes), 8)

        for token in range(130, 144):
            seq.append_token(token)
        cache.find(seq, max_tokens=144)
        self.assertEqual(len(seq.prefix_block_hashes), 9)
        self.assertEqual(
            seq.prefix_block_hashes,
            [digest for _, digest in _block_hashes(seq.token_ids, 16, 144)],
        )

    def test_incremental_hashing_only_processes_new_complete_pages(self):
        manager = BlockManager(num_blocks=16, block_size=16, max_prefix_entries=4)
        seq = Sequence(list(range(162)))
        cache = manager.prefix_cache

        with patch("nanovllm.engine.prefix_cache.sha256", wraps=sha256) as hash_fn:
            self.assertIsNone(cache.find(seq, max_tokens=128))
            self.assertEqual(hash_fn.call_count, 8)

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

    def test_checkpoint_candidates_align_to_kv_page_and_gdn_grid(self):
        manager = BlockManager(num_blocks=8, block_size=128, max_prefix_entries=4)
        seq = Sequence(list(range(401)))
        candidates, cached = manager.prefix_cache.candidates(seq, 0, 400)
        self.assertEqual([position for position, _ in candidates], [128, 256, 384])
        self.assertEqual(cached, [])
        self.assertEqual(len(seq.prefix_block_hashes), 3)

    def test_longest_exact_prefix_and_max_tokens_cap(self):
        manager = BlockManager(num_blocks=8, block_size=4, max_prefix_entries=4)
        publish_prefix(manager, [1, 2, 3, 4, 9], 4)
        publish_prefix(manager, [1, 2, 3, 4, 5, 6, 7, 8, 9], 8)

        hit = manager.prefix_cache.find(Sequence([1, 2, 3, 4, 5, 6, 7, 8, 10]), max_tokens=8)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.num_tokens, 8)
        self.assertIsNone(manager.prefix_cache.find(Sequence([1, 2, 3, 4]), max_tokens=3))

    def test_eviction_releases_blocks_under_entry_and_byte_limits(self):
        for byte_limit in (None, 12):
            with self.subTest(byte_limit=byte_limit):
                manager = BlockManager(
                    num_blocks=8,
                    block_size=4,
                    max_prefix_entries=1 if byte_limit is None else 4,
                    max_prefix_cache_snapshot_bytes=byte_limit,
                    snapshot_nbytes=9 if byte_limit is not None else 0,
                )
                first, first_block = publish_prefix(manager, [1, 2, 3, 4, 9], 4)
                self.assertEqual(manager.block_refcounts[first_block], 2)
                second, second_block = publish_prefix(manager, [5, 6, 7, 8, 9], 4)
                cache = manager.prefix_cache

                self.assertEqual(len(cache), 1)
                self.assertEqual(cache.snapshot_bytes, 9 if byte_limit is not None else 0)
                self.assertEqual(manager.block_refcounts[first_block], 1)
                self.assertEqual(manager.block_refcounts[second_block], 2)
                hit = cache.find(Sequence([5, 6, 7, 8, 10]), max_tokens=4)
                self.assertEqual(hit.block_ids, (second_block,))

                manager.deallocate(first)
                manager.deallocate(second)
                self.assertEqual(manager.block_refcounts[first_block], 0)
                self.assertTrue(cache.evict())
                self.assertEqual(cache.snapshot_bytes, 0)
                self.assertEqual(manager.block_refcounts[second_block], 0)

    def test_global_eviction_admits_new_branch(self):
        manager = BlockManager(num_blocks=32, block_size=16, max_prefix_entries=2)
        shared = list(range(128))
        original, _ = publish_prefix(manager, shared + [200], 64)
        manager.ensure_capacity(original, 128)
        publish_snapshot(manager, original, 128)
        publish_prefix(manager, [300] * 64 + [301], 64)

        self.assertEqual(len(manager.prefix_cache), 2)
        self.assertEqual(
            manager.prefix_cache.find(Sequence(shared + [202]), max_tokens=128).num_tokens,
            128,
        )
        self.assertIsNotNone(manager.prefix_cache.find(Sequence([300] * 64 + [302]), max_tokens=64))

    def test_global_capacity_replaces_oldest_entry(self):
        manager = BlockManager(num_blocks=16, block_size=16, max_prefix_entries=1)
        old, old_block = publish_prefix(manager, [1] * 64 + [9], 64)
        publish_prefix(manager, list(range(128)) + [9], 128)

        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(
            manager.prefix_cache.find(Sequence(list(range(128)) + [10]), 128).num_tokens,
            128,
        )
        self.assertEqual(manager.block_refcounts[old_block], 1)
        manager.deallocate(old)

    def test_global_one_entry_captures_only_deepest_checkpoint(self):
        manager = BlockManager(num_blocks=20, block_size=16, max_prefix_entries=1)
        seq = Sequence(list(range(256)))
        candidates, cached = manager.prefix_cache.candidates(seq, 0, 256)
        self.assertEqual(
            [
                position
                for position, _ in manager.prefix_cache.select_checkpoints(candidates, cached)
            ],
            [256],
        )

        manager = BlockManager(num_blocks=96, block_size=16, max_prefix_entries=1)
        seq = Sequence(list(range(1024)) + [9])
        manager.ensure_capacity(seq, 1024)
        publish_snapshot(manager, seq, 512)
        candidates, cached = manager.prefix_cache.candidates(seq, 512, 1024)
        self.assertEqual(
            [p for p, _ in manager.prefix_cache.select_checkpoints(candidates, cached)],
            [1024],
        )
        publish_snapshot(manager, seq, 1024)
        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(manager.prefix_cache.find(seq, 1024).num_tokens, 1024)

        shorter = BlockManager(num_blocks=32, block_size=16, max_prefix_entries=1)
        original = Sequence(list(range(128)) + [9])
        shorter.ensure_capacity(original, 128)
        publish_snapshot(shorter, original, 128)
        other = Sequence(list(range(1, 193)) + [9])
        candidates, cached = shorter.prefix_cache.candidates(other, 0, 192)
        self.assertEqual(
            [p for p, _ in shorter.prefix_cache.select_checkpoints(candidates, cached)],
            [192],
        )

    def test_backfilled_shorter_checkpoint_does_not_delete_longer(self):
        manager = BlockManager(
            num_blocks=16, block_size=16, max_prefix_entries=2, max_checkpoints_per_path=1
        )
        seq = Sequence(list(range(128)) + [9])
        manager.ensure_capacity(seq, 128)
        publish_snapshot(manager, seq, 128)
        publish_snapshot(manager, seq, 64)
        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(manager.prefix_cache.find(seq, 128).num_tokens, 128)

        manager = BlockManager(
            num_blocks=32, block_size=16, max_prefix_entries=4, max_checkpoints_per_path=2
        )
        seq = Sequence(list(range(256)) + [9])
        manager.ensure_capacity(seq, 256)
        for position in (64, 256, 128):
            publish_snapshot(manager, seq, position)
        self.assertEqual(len(manager.prefix_cache), 2)
        self.assertEqual(manager.prefix_cache.find(seq, 256).num_tokens, 256)

    def test_batch_publish_rejects_misaligned_boundary(self):
        manager = BlockManager(num_blocks=10, block_size=16, max_prefix_entries=2)
        seq = Sequence(list(range(128)))
        with self.assertRaisesRegex(RuntimeError, "invalid GDN snapshot"):
            manager.prefix_cache.publish_many(seq, (GDNStateSnapshot(63, ()),))

    def test_full_budget_refreshes_deepest_checkpoint_on_same_path(self):
        manager = BlockManager(num_blocks=24, block_size=16, max_prefix_entries=2)
        seq = Sequence(list(range(192)) + [9])
        manager.ensure_capacity(seq, 192)
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
        manager = BlockManager(num_blocks=16, block_size=16, max_prefix_entries=2)
        first, _ = publish_prefix(manager, [1] * 65, 64)
        second, _ = publish_prefix(manager, [2] * 65, 64)
        cache = manager.prefix_cache
        self.assertFalse(cache.evict(manager.block_refcounts))
        self.assertEqual(len(cache), 2)
        manager.deallocate(first)
        self.assertTrue(cache.evict(manager.block_refcounts))
        self.assertEqual(len(cache), 1)
        manager.deallocate(second)

    def test_find_is_read_only_until_prefix_is_attached(self):
        manager = BlockManager(num_blocks=16, block_size=16, max_prefix_entries=2)
        original, _ = publish_prefix(manager, [1] * 65, 64)
        cache = manager.prefix_cache
        newcomer = Sequence([1] * 64 + [2])
        hit = cache.find(newcomer, max_tokens=64)
        self.assertEqual(cache._reused, set())
        manager.attach_shared_prefix(newcomer, hit.block_ids, hit.num_tokens)
        cache.mark_reused(newcomer, hit)
        self.assertEqual(len(cache._reused), 1)
        manager.deallocate(original)
        manager.deallocate(newcomer)

    def test_zero_snapshot_budget_skips_publication(self):
        manager = BlockManager(
            num_blocks=4,
            block_size=4,
            max_prefix_entries=4,
            max_prefix_cache_snapshot_bytes=0,
        )
        seq, block_id = publish_prefix(manager, [1, 2, 3, 4, 5], 4)
        self.assertEqual(len(manager.prefix_cache), 0)
        self.assertEqual(manager.prefix_cache.snapshot_bytes, 0)
        self.assertEqual(manager.block_refcounts[block_id], 1)
        self.assertFalse(manager.prefix_cache.candidates(seq, 0, 4)[0])

    def test_prefix_cache_only_caches_text_before_image_span(self):
        manager = BlockManager(num_blocks=32, block_size=4, max_prefix_entries=4)
        seq = Sequence(list(range(81)))
        seq.state_slot = 0
        seq.image_state = SimpleNamespace(image_start=64, image_end=72)
        manager.ensure_capacity(seq, 72)

        self.assertEqual(
            [position for position, _ in manager.prefix_cache.candidates(seq, 0, 80)[0]],
            [64],
        )
        self.assertEqual(len(manager.prefix_cache), 0)

    def test_shared_history_reuses_snapshots_and_preserves_shortest(self):
        manager = BlockManager(
            num_blocks=64,
            block_size=16,
            max_prefix_entries=8,
            max_checkpoints_per_path=4,
        )
        original = Sequence(list(range(320)))
        # Both requests share 256 tokens, then diverge on separate branches.
        longer = Sequence(list(range(256)) + list(range(1000, 1272)))
        manager.ensure_capacity(original, 256)
        manager.ensure_capacity(longer, 512)

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
        self.assertEqual(manager.block_refcounts[original.block_table[0]], 3)


if __name__ == "__main__":
    unittest.main()
