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

    def can_allocate(self, seq: Sequence) -> bool:
        return seq.state_slot >= 0 or bool(self.free_slot_ids)

    def allocate(self, seq: Sequence) -> int:
        if seq.state_slot >= 0:
            return seq.state_slot
        if not self.free_slot_ids:
            raise RuntimeError("no free hybrid state slots")
        slot_id = self.free_slot_ids.popleft()
        seq.state_slot = slot_id
        return slot_id

    def deallocate(self, seq: Sequence) -> None:
        if seq.state_slot < 0:
            return
        self.free_slot_ids.append(seq.state_slot)
        seq.state_slot = -1
