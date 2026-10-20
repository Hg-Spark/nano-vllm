import unittest
from types import SimpleNamespace

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
    def test_longest_exact_prefix_and_max_tokens_cap(self):
        manager = BlockManager(num_blocks=8, block_size=4, max_prefix_entries=4)
        publish_prefix(manager, [1, 2, 3, 4, 9], 4)
        publish_prefix(manager, [1, 2, 3, 4, 5, 6, 7, 8, 9], 8)

        hit = manager.prefix_cache.find(Sequence([1, 2, 3, 4, 5, 6, 7, 8, 10]), max_tokens=8)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.num_tokens, 8)
        self.assertIsNone(manager.prefix_cache.find(Sequence([1, 2, 3, 4]), max_tokens=3))

    def test_eviction_releases_cache_block_reference(self):
        manager = BlockManager(num_blocks=4, block_size=4, max_prefix_entries=1)
        first_seq, first_block = publish_prefix(manager, [1, 2, 3, 4, 9], 4)
        self.assertEqual(manager.block_refcounts[first_block], 2)

        _, second_block = publish_prefix(manager, [5, 6, 7, 8, 9], 4)
        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(manager.block_refcounts[first_block], 1)

        hit = manager.prefix_cache.find(Sequence([5, 6, 7, 8, 10]), max_tokens=4)
        self.assertEqual(hit.block_ids, (second_block,))
        manager.deallocate(first_seq)
        self.assertEqual(manager.block_refcounts[first_block], 0)

    def test_snapshot_byte_limit_evicts_longest_and_releases_refs(self):
        manager = BlockManager(
            num_blocks=8,
            block_size=4,
            max_prefix_entries=4,
            max_prefix_cache_snapshot_bytes=12,
            snapshot_nbytes=9,
        )
        first, first_block = publish_prefix(manager, [1, 2, 3, 4, 5], 4)
        self.assertEqual(manager.prefix_cache.snapshot_bytes, 9)
        self.assertEqual(manager.block_refcounts[first_block], 2)

        second, second_block = publish_prefix(manager, [6, 7, 8, 9, 10], 4)
        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(manager.prefix_cache.snapshot_bytes, 9)
        self.assertEqual(manager.block_refcounts[first_block], 1)
        self.assertEqual(manager.block_refcounts[second_block], 2)

        for seq in (first, second):
            manager.deallocate(seq)
        self.assertTrue(manager.prefix_cache.evict())
        self.assertEqual(manager.prefix_cache.snapshot_bytes, 0)
        self.assertEqual(manager.block_refcounts[second_block], 0)

    def test_longest_leaf_eviction_keeps_shorter_shared_prefix(self):
        manager = BlockManager(num_blocks=32, block_size=16, max_prefix_entries=2)
        shared = list(range(128))
        original, _ = publish_prefix(manager, shared + [200], 64)
        manager.ensure_capacity(original, 128)
        publish_snapshot(manager, original, 128)
        publish_prefix(manager, [300] * 64 + [301], 64)

        self.assertEqual(len(manager.prefix_cache), 2)
        self.assertEqual(
            manager.prefix_cache.find(Sequence(shared + [202]), max_tokens=128).num_tokens,
            64,
        )
        self.assertIsNotNone(manager.prefix_cache.find(Sequence([300] * 64 + [302]), max_tokens=64))

    def test_global_capacity_evicts_longest_leaf_including_new_entry(self):
        manager = BlockManager(num_blocks=16, block_size=16, max_prefix_entries=1)
        old, old_block = publish_prefix(manager, [1] * 64 + [9], 64)
        publish_prefix(manager, list(range(128)) + [9], 128)

        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertIsNone(manager.prefix_cache.find(Sequence(list(range(128)) + [10]), 128))
        self.assertEqual(manager.block_refcounts[old_block], 2)
        manager.deallocate(old)

    def test_single_checkpoint_path_replaces_older_boundary(self):
        manager = BlockManager(
            num_blocks=16, block_size=16, max_prefix_entries=2, max_checkpoints_per_path=1
        )
        seq, _ = publish_prefix(manager, list(range(128)) + [9], 64)
        manager.ensure_capacity(seq, 128)
        publish_snapshot(manager, seq, 128)

        self.assertEqual(len(manager.prefix_cache), 1)
        self.assertEqual(
            manager.prefix_cache.find(Sequence(list(range(128)) + [10]), 128).num_tokens,
            128,
        )

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
        longer = Sequence(list(range(400)))
        manager.ensure_capacity(original, 256)
        manager.ensure_capacity(longer, 384)

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

        publish_snapshot(manager, longer, 384)
        self.assertEqual(len(manager.prefix_cache), 4)
        self.assertEqual(manager.prefix_cache.find(longer, max_tokens=399).num_tokens, 384)
        self.assertEqual(manager.prefix_cache.find(original, max_tokens=80).num_tokens, 64)
        self.assertEqual(manager.block_refcounts[original.block_table[0]], 4)


if __name__ == "__main__":
    unittest.main()
