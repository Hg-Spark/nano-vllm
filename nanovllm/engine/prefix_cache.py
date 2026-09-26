from collections import OrderedDict
from dataclasses import dataclass

from nanovllm.engine.state_manager import GDNStateSnapshot


@dataclass(frozen=True, slots=True)
class JointPrefixEntry:
    """One reusable hybrid prefix with matching KV and GDN state."""

    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    num_tokens: int
    state_snapshot: GDNStateSnapshot

    def __post_init__(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError("cached prefix must contain at least one token")
        if len(self.token_ids) != self.num_tokens:
            raise ValueError("cached token tuple length must match num_tokens")
        if not self.block_ids:
            raise ValueError("cached prefix must retain at least one KV block")


class JointPrefixCache:
    """Small exact-match LRU for hybrid KV + recurrent checkpoints."""

    def __init__(self, max_entries: int):
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], JointPrefixEntry] = (
            OrderedDict()
        )

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, token_ids: tuple[int, ...]) -> bool:
        return token_ids in self._entries

    def entries(self) -> tuple[JointPrefixEntry, ...]:
        return tuple(self._entries.values())

    def longest_match(
        self,
        token_ids: list[int] | tuple[int, ...],
        max_tokens: int,
    ) -> JointPrefixEntry | None:
        if max_tokens <= 0 or not self._entries:
            return None

        tokens = tuple(token_ids)
        best_key: tuple[int, ...] | None = None
        best_entry: JointPrefixEntry | None = None
        for key, entry in self._entries.items():
            if entry.num_tokens > max_tokens:
                continue
            if entry.num_tokens > len(tokens):
                continue
            if tokens[:entry.num_tokens] != key:
                continue
            if best_entry is None or entry.num_tokens > best_entry.num_tokens:
                best_key = key
                best_entry = entry

        if best_key is not None:
            self._entries.move_to_end(best_key)
        return best_entry

    def put(self, entry: JointPrefixEntry) -> list[JointPrefixEntry]:
        released: list[JointPrefixEntry] = []
        existing = self._entries.pop(entry.token_ids, None)
        if existing is not None:
            released.append(existing)

        self._entries[entry.token_ids] = entry
        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            released.append(evicted)
        return released

    def pop_lru(self) -> JointPrefixEntry | None:
        if not self._entries:
            return None
        _, entry = self._entries.popitem(last=False)
        return entry
