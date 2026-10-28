from bisect import bisect_left, bisect_right, insort
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from struct import pack

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
    kind: str = "prompt"


class PrefixCache:
    """Joint KV/GDN checkpoints committed once when a request finishes."""

    def __init__(
        self,
        block_size: int,
        max_entries: int,
        retain_blocks: Callable[[Iterable[int]], None],
        release_blocks: Callable[[Iterable[int]], None],
        max_snapshot_bytes: int | None = None,
        snapshot_nbytes: int = 0,
        max_active_requests: int = 1,
    ) -> None:
        if max_snapshot_bytes is not None and max_snapshot_bytes < 0:
            raise ValueError("max_snapshot_bytes must be non-negative")
        self.block_size = block_size
        self.max_entries = max_entries
        self.max_snapshot_bytes = max_snapshot_bytes
        self.snapshot_nbytes = snapshot_nbytes
        self._snapshot_bytes = 0
        # Request-owned staging is distinct from globally resident snapshots.
        # Staging is bounded by two slots (prompt + completion) per active request.
        self.max_staging_slots = 2 * max_active_requests
        self._staging_slots: set[tuple[int, str]] = set()
        self.staging_snapshot_bytes = 0
        self.peak_staging_snapshot_bytes = 0
        self.peak_total_snapshot_bytes = 0
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
        return (
            self.max_entries > 0
            and self.max_snapshot_bytes != 0
            and (self.max_snapshot_bytes is None or self.snapshot_nbytes <= self.max_snapshot_bytes)
        )

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def snapshot_bytes(self) -> int:
        return self._snapshot_bytes

    def stage(self, seq: Sequence, kind: str) -> None:
        if kind not in ("prompt", "completion"):
            raise ValueError("invalid snapshot staging kind")
        key = (seq.seq_id, kind)
        if key in self._staging_slots:
            return
        if len(self._staging_slots) >= self.max_staging_slots:
            raise RuntimeError("active snapshot staging slots exhausted")
        self._staging_slots.add(key)
        self.staging_snapshot_bytes += self.snapshot_nbytes
        self.peak_staging_snapshot_bytes = max(
            self.peak_staging_snapshot_bytes, self.staging_snapshot_bytes
        )
        self.peak_total_snapshot_bytes = max(
            self.peak_total_snapshot_bytes, self.staging_snapshot_bytes + self.snapshot_bytes
        )

    def unstage(self, seq: Sequence) -> None:
        for kind in ("prompt", "completion"):
            key = (seq.seq_id, kind)
            if key in self._staging_slots:
                self._staging_slots.remove(key)
                self.staging_snapshot_bytes -= self.snapshot_nbytes

    def _can_capture(self) -> bool:
        return self.enabled

    def _capacity(self) -> int:
        """Effective global capacity across entry count and CPU snapshot bytes."""
        if self.max_snapshot_bytes is None or not self.snapshot_nbytes:
            return self.max_entries
        return min(self.max_entries, self.max_snapshot_bytes // self.snapshot_nbytes)

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

    @staticmethod
    def _kind(seq: Sequence, position: int) -> str:
        # The full-prompt boundary belongs to the rolling completion state:
        # only the checkpoint before the last prompt token is a prompt entry.
        return "prompt" if position < seq.num_prompt_tokens else "conversation"

    def _publish_digest(self, seq: Sequence, position: int) -> bytes | None:
        if not self._can_capture() or self._capacity() == 0 or seq.image_state is not None:
            return None
        if position <= 0 or position > len(seq.token_ids) or position % self.block_size:
            return None
        if len(seq.block_table) < position // self.block_size:
            return None
        hashes, _ = _sequence_block_hashes(seq, self.block_size, position)
        return hashes[position // self.block_size - 1]

    @staticmethod
    def _publication_victim(entries: dict):
        # Prefer reclaiming idle conversation snapshots before idle prompts.
        # Either kind may replace an idle prompt when the global cache is full.
        for digest, (kind, active) in entries.items():
            if kind == "conversation" and not active:
                return digest
        for digest, (_, active) in entries.items():
            if not active:
                return digest
        return None

    def _virtual_plan(self, candidates: list[tuple[Sequence, int]]) -> dict:
        """Simulate the entire batch, retaining original entries if revisited.

        Entries in the returned dictionary are the *exact* desired survivors.
        The third value is the supplied candidate to install, or None if already cached.
        """
        if not self._can_capture() or not self._capacity():
            return {}
        virtual = {
            digest: (entry.kind, self._active_reuses.get(digest, 0) > 0, None)
            for digest, entry in self._entries.items()
        }
        # Consume proposals in request lifecycle order (prompt, then
        # completion). Under tight capacity the latest checkpoint wins.
        for seq, position in candidates:
            digest = self._publish_digest(seq, position)
            if digest is None or digest in virtual:
                continue
            kind = self._kind(seq, position)
            if len(virtual) >= self._capacity():
                victim = self._publication_victim(
                    {key: (value[0], value[1]) for key, value in virtual.items()}
                )
                if victim is None:
                    continue
                del virtual[victim]
            # A prior resident may have been virtually displaced and revisited:
            # retain the original snapshot instead of replacing it.
            candidate = None if digest in self._entries else (seq, position)
            virtual[digest] = (kind, False, candidate)
        return virtual

    def publish_batch(self, candidates: list[tuple[Sequence, GDNStateSnapshot]]) -> None:
        """Plan and commit one batch without independently reselecting victims."""
        if not candidates or not self._can_capture():
            return
        lookup = {}
        for seq, snapshot in candidates:
            position = snapshot.num_tokens
            if (
                position <= 0
                or position > len(seq.token_ids)
                or position % self.block_size
                or seq.image_state is not None
            ):
                raise RuntimeError("invalid GDN snapshot checkpoint boundary")
            key = (seq.seq_id, position)
            if key in lookup:
                raise RuntimeError("duplicate GDN snapshot checkpoint boundary")
            lookup[key] = snapshot

        desired = self._virtual_plan([(seq, snap.num_tokens) for seq, snap in candidates])
        # Remove only entries excluded by the virtual simulation. A previously
        # existing entry reintroduced later in the batch stays resident.
        for digest in tuple(self._entries):
            if digest not in desired:
                self._evict_hash(digest)
        for digest, (_kind, _active, candidate) in desired.items():
            if candidate is None:
                continue
            seq, position = candidate
            if digest in self._entries:
                continue
            if len(self._entries) >= self._capacity():
                raise RuntimeError("prefix publication plan exceeded capacity")
            snapshot = lookup[(seq.seq_id, position)]
            num_blocks = position // self.block_size
            block_ids = tuple(seq.block_table[:num_blocks])
            if len(block_ids) != num_blocks:
                raise RuntimeError("KV block table is shorter than cached prefix")
            self._retain_blocks(block_ids)
            hashes, _ = _sequence_block_hashes(seq, self.block_size, position)
            self._entries[digest] = PrefixEntry(
                tuple(hashes[:num_blocks]),
                block_ids,
                position,
                snapshot,
                self._kind(seq, position),
            )
            if position not in self._resident_length_counts:
                insort(self._resident_lengths, position)
                self._resident_length_counts[position] = 0
            self._resident_length_counts[position] += 1
            self._snapshot_bytes += self.snapshot_nbytes
            self.published_count += 1
            self.peak_snapshot_bytes = max(self.peak_snapshot_bytes, self._snapshot_bytes)
            self.peak_total_snapshot_bytes = max(
                self.peak_total_snapshot_bytes, self.snapshot_bytes + self.staging_snapshot_bytes
            )

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
        conversations = [key for key in keys if self._entries[key].kind == "conversation"]
        keys = conversations or keys
        return keys[0] if block_refcounts is None else max(keys, key=freed_pages)

    def evict(self, block_refcounts: list[int] | None = None) -> bool:
        victim = self._choose_victim(block_refcounts)
        if victim is None:
            return False
        self._evict_hash(victim)
        return True
