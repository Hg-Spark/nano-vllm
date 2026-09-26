import unittest

from nanovllm.engine.prefix_cache import JointPrefixCache, JointPrefixEntry
from nanovllm.engine.state_manager import GDNStateSnapshot


def make_entry(tokens, block_ids, _marker):
    return JointPrefixEntry(
        token_ids=tuple(tokens),
        block_ids=tuple(block_ids),
        num_tokens=len(tokens),
        state_snapshot=GDNStateSnapshot(
            num_tokens=len(tokens),
            layers=(),
        ),
    )


class JointPrefixCacheTest(unittest.TestCase):

    def test_longest_exact_prefix_wins(self):
        cache = JointPrefixCache(max_entries=4)
        short = make_entry([1, 2, 3, 4], [0], "short")
        long = make_entry(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1],
            "long",
        )
        cache.put(short)
        cache.put(long)

        hit = cache.longest_match(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            max_tokens=8,
        )

        self.assertIs(hit, long)

    def test_max_tokens_prevents_full_prompt_hit(self):
        cache = JointPrefixCache(max_entries=2)
        entry = make_entry([1, 2, 3, 4], [0], object())
        cache.put(entry)

        self.assertIsNone(
            cache.longest_match([1, 2, 3, 4], max_tokens=3)
        )

    def test_lru_eviction_returns_entry_for_resource_release(self):
        cache = JointPrefixCache(max_entries=1)
        first = make_entry([1, 2, 3, 4], [0], "first")
        second = make_entry([5, 6, 7, 8], [1], "second")

        self.assertEqual(cache.put(first), [])
        released = cache.put(second)

        self.assertEqual(released, [first])
        self.assertEqual(len(cache), 1)
        self.assertIs(
            cache.longest_match([5, 6, 7, 8, 9], max_tokens=4),
            second,
        )


if __name__ == "__main__":
    unittest.main()
