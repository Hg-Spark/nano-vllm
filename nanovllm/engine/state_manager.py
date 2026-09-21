from collections import deque

from nanovllm.engine.sequence import Sequence


class StateSlotManager:
    """Manage stable logical slots for per-request recurrent state.

    Every GDN layer indexes its physical Conv/Recurrent state pool with the
    same request-level slot id. slot_owners is the single occupancy source of
    truth; GPU tensors remain inside GDN layers.
    """

    def __init__(self, num_slots: int):
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.free_slot_ids: deque[int] = deque(range(num_slots))
        self.slot_owners: list[int | None] = [None] * num_slots

    def owner_of(self, slot_id: int) -> int | None:
        if not 0 <= slot_id < len(self.slot_owners):
            raise RuntimeError(f"invalid state slot {slot_id}")
        return self.slot_owners[slot_id]

    def owns(self, seq: Sequence) -> bool:
        slot_id = seq.state_slot
        return (
            0 <= slot_id < len(self.slot_owners)
            and self.slot_owners[slot_id] == seq.seq_id
        )

    def validate(self, seq: Sequence) -> int:
        slot_id = seq.state_slot
        if slot_id < 0:
            raise RuntimeError(
                f"sequence {seq.seq_id} has no allocated state slot"
            )
        if not 0 <= slot_id < len(self.slot_owners):
            raise RuntimeError(f"invalid state slot {slot_id}")
        owner = self.slot_owners[slot_id]
        if owner is None:
            raise RuntimeError(f"state slot {slot_id} is stale")
        if owner != seq.seq_id:
            raise RuntimeError(
                f"state slot {slot_id} is owned by sequence "
                f"{owner}, not {seq.seq_id}"
            )
        return slot_id

    def can_allocate(self, seq: Sequence) -> bool:
        if seq.state_slot >= 0:
            return self.owns(seq)
        return bool(self.free_slot_ids)

    def allocate(self, seq: Sequence) -> int:
        if seq.state_slot >= 0:
            return self.validate(seq)
        if seq.num_state_tokens != 0:
            raise RuntimeError(
                "cannot allocate a fresh state slot for a non-zero "
                "committed prefix"
            )
        if not self.free_slot_ids:
            raise RuntimeError("no free hybrid state slots")
        slot_id = self.free_slot_ids.popleft()
        if self.slot_owners[slot_id] is not None:
            raise RuntimeError(f"state slot {slot_id} is unexpectedly occupied")
        self.slot_owners[slot_id] = seq.seq_id
        seq.state_slot = slot_id
        return slot_id

    def deallocate(self, seq: Sequence) -> None:
        slot_id = seq.state_slot
        if slot_id < 0:
            return
        self.validate(seq)
        self.slot_owners[slot_id] = None
        self.free_slot_ids.append(slot_id)
        seq.state_slot = -1
        seq.num_state_tokens = 0
