import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import StateSlotManager


class StateSlotManagerTest(unittest.TestCase):

    def test_allocate_release_and_reuse(self):
        manager = StateSlotManager(1)
        first = Sequence([1, 2])
        second = Sequence([3, 4])

        self.assertTrue(manager.can_allocate(first))
        self.assertEqual(manager.allocate(first), 0)
        self.assertEqual(first.state_slot, 0)
        self.assertFalse(manager.can_allocate(second))

        manager.deallocate(first)
        self.assertTrue(manager.can_allocate(second))
        self.assertEqual(first.state_slot, -1)

        self.assertEqual(manager.allocate(second), 0)
        self.assertEqual(second.state_slot, 0)

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

    def test_owner_changes_only_after_release(self):
        manager = StateSlotManager(1)
        first = Sequence([1])
        second = Sequence([2])

        slot = manager.allocate(first)
        self.assertEqual(manager.owner_of(slot), first.seq_id)

        first.committed_tokens = 3
        manager.deallocate(first)
        self.assertEqual(first.committed_tokens, 3)
        self.assertIsNone(manager.owner_of(slot))

        reused = manager.allocate(second)
        self.assertEqual(reused, slot)
        self.assertEqual(manager.owner_of(slot), second.seq_id)

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
