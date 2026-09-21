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


if __name__ == "__main__":
    unittest.main()
