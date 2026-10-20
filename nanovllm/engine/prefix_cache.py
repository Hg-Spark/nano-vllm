from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from struct import pack

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


def _block_hashes(tokens: list[int] | tuple[int, ...], block_size: int):
    """Yield (prefix length, chained SHA-256) at each complete KV block."""
    parent = bytes(32)
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = sha256(
            parent + pack(f"<{block_size}I", *tokens[start : start + block_size])
        ).digest()
        yield start + block_size, parent


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    tokens: tuple[int, ...]
    block_ids: tuple[int, ...]
    num_tokens: int
    state_snapshot: GDNStateSnapshot


class PrefixCache:
    """Hashed joint KV/GDN checkpoints; globally evict longest cached leaves."""

    def __init__(
        self,
        block_size: int,
        max_entries: int,
        retain_blocks: Callable[[Iterable[int]], None],
        release_blocks: Callable[[Iterable[int]], None],
        max_snapshot_bytes: int | None = None,
        snapshot_nbytes: int = 0,
        max_checkpoints_per_path: int = 4,
    ) -> None:
        if max_snapshot_bytes is not None and max_snapshot_bytes < 0:
            raise ValueError("max_snapshot_bytes must be non-negative")
        self.block_size = block_size
        self.max_entries = max_entries
        self.max_snapshot_bytes = max_snapshot_bytes
        self.snapshot_nbytes = snapshot_nbytes
        self.max_checkpoints_per_path = max_checkpoints_per_path
        self._snapshot_bytes = 0
        self._retain_blocks = retain_blocks
        self._release_blocks = release_blocks
        self._entries: dict[bytes, PrefixEntry] = {}

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def snapshot_bytes(self) -> int:
        return self._snapshot_bytes

    def _can_capture(self) -> bool:
        return (
            self.enabled
            and self.max_snapshot_bytes != 0
            and (self.max_snapshot_bytes is None or self.snapshot_nbytes <= self.max_snapshot_bytes)
        )

    def candidates(self, seq: Sequence, start: int, end: int) -> list[tuple[int, bytes]]:
        """Find uncached 64-token checkpoints using one forward block-hash pass."""
        if not self._can_capture():
            return []
        end = min(end, seq.num_prompt_tokens)
        if seq.image_state is not None:
            end = min(end, seq.image_state.image_start)
        return [
            (position, digest)
            for position, digest in _block_hashes(seq.token_ids[:end], self.block_size)
            if position > start and position % 64 == 0 and digest not in self._entries
        ]

    def select_checkpoints(
        self, seq: Sequence, candidates: list[tuple[int, bytes]]
    ) -> list[tuple[int, bytes]]:
        """Capture only the earliest useful root and the deepest new boundary."""
        if not candidates:
            return []
        limit = min(self.max_entries, self.max_checkpoints_per_path)
        if self.max_snapshot_bytes is not None and self.snapshot_nbytes:
            limit = min(limit, self.max_snapshot_bytes // self.snapshot_nbytes)
        if limit <= 0:
            return []
        existing = [
            (position, digest)
            for position, digest in _block_hashes(
                seq.token_ids[: seq.num_prompt_tokens], self.block_size
            )
            if digest in self._entries
        ]
        # With one checkpoint per path, an older, shorter insertion cannot
        # replace a deeper, already cached state.
        if limit == 1 and existing and existing[-1][0] > candidates[-1][0]:
            return []
        if limit == 1 or existing:
            return candidates[-1:]
        return candidates[:1] if len(candidates) == 1 else [candidates[0], candidates[-1]]

    def find(self, seq: Sequence, max_tokens: int) -> PrefixEntry | None:
        if not self.enabled or max_tokens <= 0:
            return None
        if seq.image_state is not None:
            max_tokens = min(max_tokens, seq.image_state.image_start)
        best_entry: PrefixEntry | None = None
        for _, digest in _block_hashes(seq.token_ids[:max_tokens], self.block_size):
            entry = self._entries.get(digest)
            if entry is not None:
                best_entry = entry
        return best_entry

    def publish(self, seq: Sequence, state_snapshot: GDNStateSnapshot, prefix_tokens: int) -> None:
        if not self._can_capture():
            return
        for position, digest in _block_hashes(seq.token_ids[:prefix_tokens], self.block_size):
            if position == prefix_tokens:
                self._publish(seq, state_snapshot, prefix_tokens, digest)
                return
        raise RuntimeError("GDN snapshot boundary is not a complete KV block")

    def publish_many(self, seq: Sequence, snapshots: tuple[GDNStateSnapshot, ...]) -> None:
        if not snapshots or not self._can_capture():
            return
        by_position = {snapshot.num_tokens: snapshot for snapshot in snapshots}
        if len(by_position) != len(snapshots) or any(
            position <= 0
            or position > seq.num_prompt_tokens
            or position % self.block_size
            or position % 64
            for position in by_position
        ):
            raise RuntimeError("invalid GDN snapshot checkpoint boundary")
        for position, digest in _block_hashes(seq.token_ids[: max(by_position)], self.block_size):
            snapshot = by_position.get(position)
            if snapshot is not None:
                self._publish(seq, snapshot, position, digest)

    def _publish(
        self, seq: Sequence, snapshot: GDNStateSnapshot, position: int, digest: bytes
    ) -> None:
        if snapshot.num_tokens != position:
            raise RuntimeError("GDN snapshot boundary does not match KV prefix boundary")
        if digest in self._entries:
            return
        num_blocks = position // self.block_size
        block_ids = tuple(seq.block_table[:num_blocks])
        if len(block_ids) != num_blocks:
            raise RuntimeError("KV block table is shorter than cached prefix")

        self._retain_blocks(block_ids)
        entry = PrefixEntry(tuple(seq.token_ids[:position]), block_ids, position, snapshot)
        self._entries[digest] = entry
        self._snapshot_bytes += self.snapshot_nbytes

        # Reconcile affected descendant paths using the existing hash index,
        # never evicting the checkpoint just captured.
        for leaf in tuple(self._entries.values()):
            if leaf.num_tokens < position or leaf.tokens[:position] != entry.tokens:
                continue
            chain = [
                key
                for _, key in _block_hashes(leaf.tokens, self.block_size)
                if key in self._entries
            ]
            if len(chain) > self.max_checkpoints_per_path:
                old = [key for key in chain if key != digest]
                # For a single-slot path, retain the deepest checkpoint.
                if self.max_checkpoints_per_path == 1 and leaf.num_tokens > position:
                    self._evict_hash(digest)
                    return
                victim = (
                    old[0]
                    if self.max_checkpoints_per_path == 1
                    else (old[1] if len(old) > 1 else old[0])
                )
                self._evict_hash(victim)

        # New checkpoints must not be inserted and immediately evicted.
        while len(self._entries) > self.max_entries or (
            self.max_snapshot_bytes is not None and self._snapshot_bytes > self.max_snapshot_bytes
        ):
            victims = [(key, value) for key, value in self._entries.items() if key != digest]
            if not victims:
                break
            self._evict_hash(max(victims, key=lambda item: item[1].num_tokens)[0])

    def _evict_hash(self, digest: bytes) -> None:
        entry = self._entries.pop(digest)
        self._snapshot_bytes -= self.snapshot_nbytes
        self._release_blocks(entry.block_ids)

    def evict(self, block_refcounts: list[int] | None = None) -> bool:
        if not self._entries:
            return False
        if block_refcounts is None:
            victim = max(self._entries, key=lambda key: self._entries[key].num_tokens)
        else:
            victim = max(
                self._entries,
                key=lambda key: (
                    sum(
                        block_refcounts[block_id] == 1 for block_id in self._entries[key].block_ids
                    ),
                    self._entries[key].num_tokens,
                ),
            )
        self._evict_hash(victim)
        return True
