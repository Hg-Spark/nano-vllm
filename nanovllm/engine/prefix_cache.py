from bisect import bisect_left, bisect_right, insort
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from math import lcm
from struct import pack

from nanovllm.engine.schedule import GDN_CHECKPOINT_INTERVAL
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


def _sequence_block_hashes(seq: Sequence, block_size: int, stop: int) -> tuple[list[bytes], int]:
    """Compute missing block hashes once and reuse indexed results on subsequent lookups."""
    num_blocks = min(stop, len(seq.token_ids)) // block_size
    hashes = seq.prefix_block_hashes
    if len(hashes) < num_blocks:
        parent = hashes[-1] if hashes else bytes(32)
        for index in range(len(hashes), num_blocks):
            start = index * block_size
            parent = sha256(
                parent + pack(f"<{block_size}I", *seq.token_ids[start : start + block_size])
            ).digest()
            hashes.append(parent)
    return hashes, num_blocks


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    block_hashes: tuple[bytes, ...]
    block_ids: tuple[int, ...]
    num_tokens: int
    state_snapshot: GDNStateSnapshot


class PrefixCache:
    """Hashed joint KV/GDN checkpoints with FIFO global replacement."""

    def __init__(
        self,
        block_size: int,
        max_entries: int,
        retain_blocks: Callable[[Iterable[int]], None],
        release_blocks: Callable[[Iterable[int]], None],
        max_snapshot_bytes: int | None = None,
        snapshot_nbytes: int = 0,
    ) -> None:
        if max_snapshot_bytes is not None and max_snapshot_bytes < 0:
            raise ValueError("max_snapshot_bytes must be non-negative")
        self.block_size = block_size
        self.max_entries = max_entries
        self.max_snapshot_bytes = max_snapshot_bytes
        self.snapshot_nbytes = snapshot_nbytes
        self._snapshot_bytes = 0
        self.published_count = 0
        self.evicted_count = 0
        self.peak_snapshot_bytes = 0
        self._retain_blocks = retain_blocks
        self._release_blocks = release_blocks
        self._entries: dict[bytes, PrefixEntry] = {}
        # The index contains resident checkpoint lengths, not every candidate
        # page in a long prompt. Several hashes may share one length.
        self._resident_lengths: list[int] = []
        self._resident_length_counts: dict[int, int] = {}
        # Active request references, not permanent "ever-hit" flags.
        self._active_reuses: dict[bytes, int] = {}
        self._seq_reuses: dict[int, bytes] = {}

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
        """Probe new aligned boundaries; resident length lookups belong to find()."""
        if not self._can_capture():
            return []
        end = min(end, seq.num_prompt_tokens)
        if seq.image_state is not None:
            end = min(end, seq.image_state.image_start)
        hashes, _ = _sequence_block_hashes(seq, self.block_size, end)
        step = lcm(GDN_CHECKPOINT_INTERVAL, self.block_size)
        first_new = (start // step + 1) * step
        return [
            (position, hashes[position // self.block_size - 1])
            for position in range(first_new, end + 1, step)
            if hashes[position // self.block_size - 1] not in self._entries
        ]

    def _capacity(self) -> int:
        """Effective global capacity across entry count and CPU snapshot bytes."""
        if self.max_snapshot_bytes is None or not self.snapshot_nbytes:
            return self.max_entries
        return min(self.max_entries, self.max_snapshot_bytes // self.snapshot_nbytes)

    def select_checkpoints(self, candidates: list[tuple[int, bytes]]) -> list[tuple[int, bytes]]:
        """Capture at most one deepest reusable checkpoint per chunk."""
        if self._capacity() == 0:
            return []
        return candidates[-1:]

    def find(self, seq: Sequence, max_tokens: int) -> PrefixEntry | None:
        if not self.enabled or max_tokens <= 0 or not self._entries:
            return None
        if seq.image_state is not None:
            max_tokens = min(max_tokens, seq.image_state.image_start)
        # Only probe actual resident checkpoint lengths. Unlike an all-pages
        # reverse scan, a long waiting request costs O(log K + K) for K distinct
        # resident lengths even when its own prefix is not cached.
        end = bisect_right(self._resident_lengths, max_tokens)
        if not end:
            return None
        hashes, _ = _sequence_block_hashes(seq, self.block_size, self._resident_lengths[end - 1])
        for position in reversed(self._resident_lengths[:end]):
            digest = hashes[position // self.block_size - 1]
            entry = self._entries.get(digest)
            if entry is not None and entry.num_tokens == position:
                return entry
        return None

    def mark_reused(self, seq: Sequence, entry: PrefixEntry) -> None:
        """Protect a checkpoint while a request uses its cached prefix."""
        if seq.seq_id in self._seq_reuses:
            raise RuntimeError("prefix reuse already registered for request")
        digest = entry.block_hashes[-1]
        self._seq_reuses[seq.seq_id] = digest
        self._active_reuses[digest] = self._active_reuses.get(digest, 0) + 1

    def unmark_reused(self, seq: Sequence) -> None:
        """Release request-local reuse protection, including after cache eviction."""
        digest = self._seq_reuses.pop(seq.seq_id, None)
        if digest is None:
            return
        remaining = self._active_reuses[digest] - 1
        if remaining:
            self._active_reuses[digest] = remaining
        else:
            del self._active_reuses[digest]

    def publish_many(self, seq: Sequence, snapshots: tuple[GDNStateSnapshot, ...]) -> None:
        if not snapshots or not self._can_capture():
            return
        by_position = {snapshot.num_tokens: snapshot for snapshot in snapshots}
        if len(by_position) != len(snapshots) or any(
            position <= 0
            or position > seq.num_prompt_tokens
            or (seq.image_state is not None and position > seq.image_state.image_start)
            or position % self.block_size
            or position % GDN_CHECKPOINT_INTERVAL
            for position in by_position
        ):
            raise RuntimeError("invalid GDN snapshot checkpoint boundary")
        hashes, _ = _sequence_block_hashes(seq, self.block_size, max(by_position))
        for position, snapshot in sorted(by_position.items()):
            digest = hashes[position // self.block_size - 1]
            self._publish(seq, snapshot, position, digest)

    def _publish(
        self,
        seq: Sequence,
        snapshot: GDNStateSnapshot,
        position: int,
        digest: bytes,
    ) -> None:
        if digest in self._entries or self._capacity() == 0:
            return
        num_blocks = position // self.block_size
        block_ids = tuple(seq.block_table[:num_blocks])
        if len(block_ids) != num_blocks:
            raise RuntimeError("KV block table is shorter than cached prefix")

        # Retain before eviction: an old checkpoint may share the new KV pages.
        self._retain_blocks(block_ids)
        hashes, _ = _sequence_block_hashes(seq, self.block_size, position)
        self._entries[digest] = PrefixEntry(
            tuple(hashes[:num_blocks]), block_ids, position, snapshot
        )
        if position not in self._resident_length_counts:
            insort(self._resident_lengths, position)
            self._resident_length_counts[position] = 0
        self._resident_length_counts[position] += 1
        self._snapshot_bytes += self.snapshot_nbytes
        self.published_count += 1
        self.peak_snapshot_bytes = max(self.peak_snapshot_bytes, self._snapshot_bytes)
        while len(self._entries) > self._capacity():
            self._evict_hash(self._choose_victim())

    def _evict_hash(self, digest: bytes) -> None:
        entry = self._entries.pop(digest)
        position = entry.num_tokens
        count = self._resident_length_counts[position] - 1
        if count:
            self._resident_length_counts[position] = count
        else:
            del self._resident_length_counts[position]
            self._resident_lengths.pop(bisect_left(self._resident_lengths, position))
        self.evicted_count += 1
        self._snapshot_bytes -= self.snapshot_nbytes
        # A request may still be using this prefix after its cache entry is evicted.
        # Its reuse reference is released by BlockManager.deallocate(), not here.
        self._release_blocks(entry.block_ids)

    def _choose_victim(self, block_refcounts: list[int] | None = None) -> bytes | None:
        """Cold-first FIFO, with a reclaimable-page filter under GPU pressure."""

        def freed_pages(key: bytes) -> int:
            return sum(block_refcounts[block_id] == 1 for block_id in self._entries[key].block_ids)

        keys = list(self._entries)
        if block_refcounts is not None:
            keys = [key for key in keys if freed_pages(key)]
        if not keys:
            return None
        cold = [key for key in keys if not self._active_reuses.get(key, 0)]
        keys = cold or keys
        return keys[0] if block_refcounts is None else max(keys, key=freed_pages)

    def evict(self, block_refcounts: list[int] | None = None) -> bool:
        victim = self._choose_victim(block_refcounts)
        if victim is None:
            return False
        self._evict_hash(victim)
        return True
