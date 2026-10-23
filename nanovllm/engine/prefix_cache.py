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

    def candidates(
        self, seq: Sequence, start: int, end: int
    ) -> tuple[list[tuple[int, bytes]], list[int]]:
        """Find new and cached 64-token boundaries in one block-hash pass."""
        if not self._can_capture():
            return [], []
        end = min(end, seq.num_prompt_tokens)
        if seq.image_state is not None:
            end = min(end, seq.image_state.image_start)
        candidates = []
        hashes, _ = _sequence_block_hashes(seq, self.block_size, end)
        # A joint GDN/KV checkpoint must align to both the 64-token state grid
        # and the KV page boundary. Revisit resident entries (bounded by
        # max_entries) instead of rescanning every old checkpoint each chunk.
        step = lcm(GDN_CHECKPOINT_INTERVAL, self.block_size)
        cached = []
        for position in self._resident_lengths[: bisect_right(self._resident_lengths, end)]:
            if position % step:
                continue
            digest = hashes[position // self.block_size - 1]
            if digest in self._entries:
                cached.append(position)
        first_new = (start // step + 1) * step
        for position in range(first_new, end + 1, step):
            digest = hashes[position // self.block_size - 1]
            if digest not in self._entries:
                candidates.append((position, digest))
        return candidates, cached

    def _path_limit(self) -> int:
        limit = min(self.max_entries, self.max_checkpoints_per_path)
        if self.max_snapshot_bytes is not None and self.snapshot_nbytes:
            limit = min(limit, self.max_snapshot_bytes // self.snapshot_nbytes)
        return limit

    def _plan_publication(
        self, seq: Sequence, position: int, digest: bytes
    ) -> tuple[bytes, ...] | None:
        """Plan all evictions before publishing, including downstream branches.

        None means that a checkpoint cannot be retained without evicting an
        actively reused or terminal descendant checkpoint. This method never
        changes cache ownership, statistics, or resident indexes.
        """
        if digest in self._entries:
            return ()
        limit = self._path_limit()
        if limit < 1:
            return None

        block_count = position // self.block_size
        hashes, _ = _sequence_block_hashes(seq, self.block_size, position)
        own_path = tuple(hashes[:block_count])
        descendant_paths = [
            entry.block_hashes
            for entry in self._entries.values()
            if entry.num_tokens > position and entry.block_hashes[block_count - 1] == digest
        ]
        remaining = set(self._entries)
        remaining.add(digest)
        victims: list[bytes] = []

        for index, full_path in enumerate((own_path, *descendant_paths)):
            chain = [key for key in full_path if key in remaining]
            while len(chain) > limit:
                # A new endpoint may replace its own older ancestors.
                # A backfill cannot destroy the root or terminal descendant
                # of any existing branch just to insert an intermediate node.
                removable = [*chain[1:-1], chain[0]] if index == 0 else chain[1:-1]
                victim = next(
                    (
                        key
                        for key in removable
                        if key != digest and not self._active_reuses.get(key, 0)
                    ),
                    None,
                )
                if victim is None:
                    return None
                remaining.remove(victim)
                chain.remove(victim)
                victims.append(victim)

        capacity = self.max_entries
        if self.max_snapshot_bytes is not None and self.snapshot_nbytes:
            capacity = min(capacity, self.max_snapshot_bytes // self.snapshot_nbytes)
        own_ancestors = set(own_path[:-1])
        while len(remaining) > capacity:
            # Under global pressure, discard unrelated cold branches first.
            # Only then consider intermediate ancestors and finally the root.
            eligible = [
                key
                for key in self._entries
                if key in remaining and not self._active_reuses.get(key, 0)
            ]
            if not eligible:
                return None
            victim = min(
                eligible,
                key=lambda key: (0 if key not in own_ancestors else 2 if key == own_path[0] else 1),
            )
            remaining.remove(victim)
            victims.append(victim)
        return tuple(victims)

    def select_checkpoints(
        self,
        candidates: list[tuple[int, bytes]],
        cached: list[int],
        max_reusable_tokens: int | None = None,
        seq: Sequence | None = None,
    ) -> list[tuple[int, bytes]]:
        """Choose useful checkpoints within the effective global/per-path capacity."""
        if not candidates:
            return []
        limit = self._path_limit()
        if limit <= 0:
            return []
        if max_reusable_tokens is not None and (limit == 1 or cached):
            # A checkpoint at the full prompt length cannot resume another
            # identical prompt, since its last token still needs a forward pass.
            reusable = [pair for pair in candidates if pair[0] <= max_reusable_tokens]
            if reusable:
                candidates = reusable
            elif cached and cached[-1] <= max_reusable_tokens:
                return []
        if limit == 1 and cached and cached[-1] > candidates[-1][0]:
            return []
        if limit == 1 or cached:
            selected = candidates[-1:]
        else:
            selected = candidates[:1] if len(candidates) == 1 else [candidates[0], candidates[-1]]
        if seq is not None:
            selected = [
                (position, digest)
                for position, digest in selected
                if self._plan_publication(seq, position, digest) is not None
            ]
            if not selected:
                # The deepest candidate may conflict with a resident
                # descendant; a shallower boundary can still be publishable.
                # Bound validation work for long prompts (thousands of
                # possible 64-token checkpoints) when a deeper branch is full.
                for position, digest in reversed(candidates[-5:-1]):
                    if self._plan_publication(seq, position, digest) is not None:
                        selected = [(position, digest)]
                        break
        return selected

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
        path = []
        stop = max(by_position)
        hashes, _ = _sequence_block_hashes(seq, self.block_size, stop)
        # Only resident ancestors and new checkpoints can change the path.
        # Scanning every KV page repeats work for long prompts.
        end = bisect_right(self._resident_lengths, stop)
        positions = sorted(set(self._resident_lengths[:end]) | by_position.keys())
        for position in positions:
            digest = hashes[position // self.block_size - 1]
            if digest in self._entries:
                path.append(digest)
            snapshot = by_position.get(position)
            if snapshot is not None:
                self._publish(seq, snapshot, position, digest, path)

    def _publish(
        self,
        seq: Sequence,
        snapshot: GDNStateSnapshot,
        position: int,
        digest: bytes,
        path: list[bytes],
    ) -> None:
        if digest in self._entries:
            return
        num_blocks = position // self.block_size
        block_ids = tuple(seq.block_table[:num_blocks])
        if len(block_ids) != num_blocks:
            raise RuntimeError("KV block table is shorter than cached prefix")

        victims = self._plan_publication(seq, position, digest)
        if victims is None:
            return

        # Preflight all descendant branches before touching any KV references.
        self._retain_blocks(block_ids)
        hashes, _ = _sequence_block_hashes(seq, self.block_size, position)
        entry = PrefixEntry(tuple(hashes[:num_blocks]), block_ids, position, snapshot)
        self._entries[digest] = entry
        if position not in self._resident_length_counts:
            insort(self._resident_lengths, position)
            self._resident_length_counts[position] = 0
        self._resident_length_counts[position] += 1
        self._snapshot_bytes += self.snapshot_nbytes
        self.published_count += 1
        self.peak_snapshot_bytes = max(self.peak_snapshot_bytes, self._snapshot_bytes)
        for victim in victims:
            self._evict_hash(victim)
        # publish_many() may insert more checkpoints on this request's path.
        path[:] = [key for key in (*path, digest) if key in self._entries]

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

    def evict(self, block_refcounts: list[int] | None = None) -> bool:
        if not self._entries:
            return False
        # Evict cold checkpoints first. Active requests retain their own KV
        # references, but dropping a reused checkpoint hurts future hits.
        candidates = [key for key in self._entries if self._active_reuses.get(key, 0) == 0]
        if not candidates:
            candidates = list(self._entries)
        if block_refcounts is None:
            victim = candidates[0]
        else:
            victim = max(
                candidates,
                key=lambda key: sum(
                    block_refcounts[block_id] == 1 for block_id in self._entries[key].block_ids
                ),
            )
            if not any(
                block_refcounts[block_id] == 1 for block_id in self._entries[victim].block_ids
            ):
                return False
        self._evict_hash(victim)
        return True
