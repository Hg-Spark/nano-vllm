import unittest
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


def publish_prefix(
    manager: BlockManager, tokens: list[int], prefix_tokens: int
) -> tuple[Sequence, int]:
    seq = Sequence(tokens)
    manager.ensure_capacity(seq, prefix_tokens)
    seq.committed_tokens = 0
    seq.state_slot = 0
    block_id = seq.block_table[0]
    manager.publish_prefix(
        seq, GDNStateSnapshot(num_tokens=prefix_tokens, layers=()), prefix_tokens
    )
    return seq, block_id


class BlockManagerPrefixCacheTest(unittest.TestCase):
    def test_longest_exact_prefix_and_max_tokens_cap(self):
        manager = BlockManager(num_blocks=8, block_size=4, max_prefix_entries=4)
        publish_prefix(manager, [1, 2, 3, 4, 9], 4)
        publish_prefix(manager, [1, 2, 3, 4, 5, 6, 7, 8, 9], 8)

        hit = manager.find_prefix(Sequence([1, 2, 3, 4, 5, 6, 7, 8, 10]), max_tokens=8)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.num_tokens, 8)
        self.assertIsNone(manager.find_prefix(Sequence([1, 2, 3, 4]), max_tokens=3))

    def test_lru_eviction_releases_cache_block_reference(self):
        manager = BlockManager(num_blocks=4, block_size=4, max_prefix_entries=1)
        first_seq, first_block = publish_prefix(manager, [1, 2, 3, 4, 9], 4)
        self.assertEqual(manager.block_refcounts[first_block], 2)

        _, second_block = publish_prefix(manager, [5, 6, 7, 8, 9], 4)
        self.assertEqual(manager.num_cached_prefixes, 1)
        self.assertEqual(manager.block_refcounts[first_block], 1)

        hit = manager.find_prefix(Sequence([5, 6, 7, 8, 10]), max_tokens=4)
        self.assertEqual(hit.block_ids, (second_block,))
        manager.deallocate(first_seq)
        self.assertEqual(manager.block_refcounts[first_block], 0)

    def test_prefix_cache_only_caches_text_before_image_span(self):
        manager = BlockManager(num_blocks=8, block_size=4, max_prefix_entries=4)
        seq = Sequence(list(range(9)))
        seq.state_slot = 0
        seq.image_state = SimpleNamespace(image_start=4, image_end=8)
        manager.ensure_capacity(seq, 8)

        self.assertTrue(manager.should_cache_prefix(seq, 4))
        self.assertFalse(manager.should_cache_prefix(seq, 8))
        with self.assertRaisesRegex(RuntimeError, "before image tokens"):
            manager.publish_prefix(seq, GDNStateSnapshot(8, ()), 8)


if __name__ == "__main__":
    unittest.main()
