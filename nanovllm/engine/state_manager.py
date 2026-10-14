from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class GDNStateSnapshot:
    """Host checkpoint preserving GDN state precision at one prefix."""

    num_tokens: int
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class StateSlotManager:
    """Manage stable logical slots for per-request recurrent state."""

    def __init__(self, num_slots: int) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.free_slot_ids: deque[int] = deque(range(num_slots))
        self.slot_owners: list[int | None] = [None] * num_slots

    def owns(self, seq: Sequence) -> bool:
        slot_id = seq.state_slot
        return 0 <= slot_id < len(self.slot_owners) and self.slot_owners[slot_id] == seq.seq_id

    def validate(self, seq: Sequence) -> int:
        slot_id = seq.state_slot
        if not 0 <= slot_id < len(self.slot_owners):
            raise RuntimeError(
                f"sequence {seq.seq_id} has invalid or unallocated state slot {slot_id}"
            )
        owner = self.slot_owners[slot_id]
        if owner is None:
            raise RuntimeError(f"state slot {slot_id} is stale")
        if owner != seq.seq_id:
            raise RuntimeError(
                f"state slot {slot_id} is owned by sequence {owner}, not {seq.seq_id}"
            )
        return slot_id

    def can_allocate(self, seq: Sequence) -> bool:
        return self.owns(seq) if seq.state_slot >= 0 else bool(self.free_slot_ids)

    def allocate(self, seq: Sequence) -> int:
        if seq.state_slot >= 0:
            return self.validate(seq)
        if seq.committed_tokens != 0:
            raise RuntimeError("cannot allocate a fresh state slot for a non-zero committed prefix")
        if not self.free_slot_ids:
            raise RuntimeError("no free hybrid state slots")
        slot_id = self.free_slot_ids.popleft()
        self.slot_owners[slot_id] = seq.seq_id
        seq.state_slot = slot_id
        return slot_id

    def deallocate(self, seq: Sequence) -> None:
        if seq.state_slot < 0:
            return
        slot_id = self.validate(seq)
        self.slot_owners[slot_id] = None
        self.free_slot_ids.append(slot_id)
        seq.state_slot = -1
