import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import StateSlotManager


class StateSlotManagerTest(unittest.TestCase):

    def test_allocate_release_and_reuse(self):
        manager = StateSlotManager(1)
        first = Sequence([1, 2])
        second = Sequence([3, 4])

        self.assertTrue(manager.can_allocate(first))
        slot = manager.allocate(first)
        self.assertEqual(slot, 0)
        self.assertEqual(first.state_slot, 0)
        self.assertEqual(manager.slot_owners[slot], first.seq_id)
        self.assertFalse(manager.can_allocate(second))

        first.committed_tokens = 2
        manager.deallocate(first)
        self.assertTrue(manager.can_allocate(second))
        self.assertEqual(first.state_slot, -1)
        self.assertEqual(first.committed_tokens, 2)
        self.assertIsNone(manager.slot_owners[slot])

        self.assertEqual(manager.allocate(second), slot)
        self.assertEqual(second.state_slot, slot)
        self.assertEqual(manager.slot_owners[slot], second.seq_id)

    def test_stale_slot_is_rejected(self):
        manager = StateSlotManager(1)
        seq = Sequence([1])
        seq.state_slot = 0

        with self.assertRaisesRegex(RuntimeError, "state slot 0 is stale"):
            manager.allocate(seq)

    def test_exhaustion_is_explicit(self):
        manager = StateSlotManager(1)
        manager.allocate(Sequence([1]))

        with self.assertRaisesRegex(RuntimeError, "no free hybrid state slots"):
            manager.allocate(Sequence([2]))

    def test_slot_alias_is_rejected(self):
        manager = StateSlotManager(1)
        first = Sequence([1])
        second = Sequence([2])
        manager.allocate(first)
        second.state_slot = first.state_slot

        self.assertFalse(manager.can_allocate(second))
        with self.assertRaisesRegex(
            RuntimeError,
            "is owned by sequence",
        ):
            manager.allocate(second)

    def test_fresh_slot_rejects_nonzero_committed_progress(self):
        manager = StateSlotManager(1)
        seq = Sequence([1])
        seq.committed_tokens = 1

        with self.assertRaisesRegex(
            RuntimeError,
            "non-zero committed prefix",
        ):
            manager.allocate(seq)


if __name__ == "__main__":
    unittest.main()
