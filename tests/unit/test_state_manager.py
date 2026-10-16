import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import StateSlotManager


class StateSlotManagerTest(unittest.TestCase):
    def test_exhaustion_is_explicit(self):
        manager = StateSlotManager(1)
        manager.allocate(Sequence([1]))

        with self.assertRaisesRegex(RuntimeError, "no free hybrid state slots"):
            manager.allocate(Sequence([2]))


if __name__ == "__main__":
    unittest.main()
