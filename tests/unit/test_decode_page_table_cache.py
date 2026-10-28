"""CPU contracts for reusable Decode page metadata and mapping invalidation."""

import unittest

from nanovllm.engine.batch_common import DecodePageTableCache, build_decode_batch_layout
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence


def decode_chunk(seq):
    return ScheduledChunk(seq, len(seq) - 1, len(seq))


class DecodePageTableCacheTest(unittest.TestCase):
    def test_page_indices_reused_until_physical_page_changes(self):
        manager = BlockManager(num_blocks=40, block_size=16)
        seq = Sequence(list(range(20)))
        manager.ensure_capacity(seq, len(seq))
        cache = DecodePageTableCache()

        first = build_decode_batch_layout((decode_chunk(seq),), 16, cache)
        self.assertEqual(first.paged_kv_indices, tuple(seq.block_table))
        seq.append_token(21)
        second = build_decode_batch_layout((decode_chunk(seq),), 16, cache)
        self.assertIs(first.paged_kv_indices, second.paged_kv_indices)
        self.assertEqual(first.page_table_key, second.page_table_key)
        self.assertEqual(second.paged_kv_last_page_len, (5,))

        while len(seq) < 33:
            seq.append_token(99)
        manager.ensure_capacity(seq, len(seq))
        third = build_decode_batch_layout((decode_chunk(seq),), 16, cache)
        self.assertEqual(len(third.paged_kv_indices), 3)
        self.assertNotEqual(second.page_table_key, third.page_table_key)
        self.assertEqual(third.paged_kv_last_page_len, (1,))

    def test_batch_row_order_and_page_reuse_invalidate_key(self):
        manager = BlockManager(num_blocks=16, block_size=16)
        left, right = Sequence([1] * 17), Sequence([2] * 17)
        manager.ensure_capacity(left, len(left))
        manager.ensure_capacity(right, len(right))
        cache = DecodePageTableCache()
        a = build_decode_batch_layout((decode_chunk(left), decode_chunk(right)), 16, cache)
        b = build_decode_batch_layout((decode_chunk(right), decode_chunk(left)), 16, cache)
        self.assertNotEqual(a.page_table_key, b.page_table_key)
        self.assertEqual(b.paged_kv_indices, tuple(right.block_table + left.block_table))

        # Same sequence ID, same page count, but new physical mapping.
        manager.deallocate(left)
        manager.ensure_capacity(left, len(left))
        c = build_decode_batch_layout((decode_chunk(right), decode_chunk(left)), 16, cache)
        self.assertNotEqual(b.page_table_key, c.page_table_key)
        self.assertEqual(c.paged_kv_indices, tuple(right.block_table + left.block_table))

    def test_shared_prefix_attachment_invalidates_page_table(self):
        manager = BlockManager(num_blocks=24, block_size=16)
        source = Sequence([3] * 32)
        dest = Sequence([3] * 32)
        manager.ensure_capacity(source, 32)
        cache = DecodePageTableCache()
        manager.attach_shared_prefix(dest, tuple(source.block_table), 32)
        result = build_decode_batch_layout((decode_chunk(dest),), 16, cache)
        self.assertEqual(result.paged_kv_indices, tuple(source.block_table))
        self.assertGreater(dest.block_table_version, 0)


if __name__ == "__main__":
    unittest.main()
