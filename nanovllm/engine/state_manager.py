from collections import deque

from nanovllm.engine.sequence import Sequence


class StateSlotManager:
    """Manage stable logical slots for per-request recurrent state.

    Hybrid models keep one physical Conv/Recurrent state pool per layer and
    index every pool with the same request-level slot id. This manager owns only
    active-request slot allocation; prefix snapshots are outside its scope.
    """

    def __init__(self, num_slots: int):
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.free_slot_ids: deque[int] = deque(range(num_slots))
        self.used_slot_ids: set[int] = set()

    def can_allocate(self, seq: Sequence) -> bool:
        return seq.state_slot >= 0 or bool(self.free_slot_ids)

    def allocate(self, seq: Sequence) -> int:
        if seq.state_slot >= 0:
            if seq.state_slot not in self.used_slot_ids:
                raise RuntimeError(
                    f"state slot {seq.state_slot} is stale"
                )
            return seq.state_slot
        if not self.free_slot_ids:
            raise RuntimeError("no free hybrid state slots")
        slot_id = self.free_slot_ids.popleft()
        self.used_slot_ids.add(slot_id)
        seq.state_slot = slot_id
        return slot_id

    def deallocate(self, seq: Sequence) -> None:
        slot_id = seq.state_slot
        if slot_id < 0:
            return
        if slot_id not in self.used_slot_ids:
            raise RuntimeError(f"state slot {slot_id} is not allocated")
        self.used_slot_ids.remove(slot_id)
        self.free_slot_ids.append(slot_id)
        seq.state_slot = -1
