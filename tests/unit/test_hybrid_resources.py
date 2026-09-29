import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.hybrid_resources import HybridResources
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot, StateSlotManager


class HybridResourcesTest(unittest.TestCase):

    def make_resources(self):
        blocks = BlockManager(
            num_blocks=4,
            block_size=4,
            max_prefix_entries=2,
        )
        states = StateSlotManager(2)
        return blocks, states, HybridResources(blocks, states)

    def test_validate_request_rejects_committed_history_without_resources(self):
        _, _, resources = self.make_resources()
        seq = Sequence([1, 2])
        seq.committed_tokens = 1

        with self.assertRaisesRegex(
            RuntimeError,
            "committed history without hybrid resources",
        ):
            resources.validate_request(seq)

    def test_accounting_tracks_prefix_refs_and_detects_corruption(self):
        blocks, states, resources = self.make_resources()
        seq = Sequence([1, 2, 3, 4, 5])
        blocks.ensure_capacity(seq, 4)
        states.allocate(seq)

        blocks.publish_prefix(
            seq,
            GDNStateSnapshot(num_tokens=4, layers=()),
            4,
        )

        resources.validate_accounting([seq])

        block_id = seq.block_table[0]
        blocks.block_refcounts[block_id] += 1
        with self.assertRaisesRegex(
            RuntimeError,
            "KV cache refcount accounting mismatch",
        ):
            resources.validate_accounting([seq])


if __name__ == "__main__":
    unittest.main()
