import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import StateSlotManager


class StateSlotManagerTest(unittest.TestCase):

    def test_allocate_release_and_reuse(self):
        manager = StateSlotManager(1)
        first = Sequence([1, 2])
        second = Sequence([3, 4])

        self.assertEqual(manager.allocate(first), 0)
        self.assertEqual(first.state_slot, 0)

        manager.deallocate(first)
        self.assertEqual(first.state_slot, -1)

        self.assertEqual(manager.allocate(second), 0)
        self.assertEqual(second.state_slot, 0)

    def test_exhaustion_is_explicit(self):
        manager = StateSlotManager(1)
        manager.allocate(Sequence([1]))

        with self.assertRaisesRegex(RuntimeError, "no free hybrid state slots"):
            manager.allocate(Sequence([2]))


if __name__ == "__main__":
    unittest.main()
