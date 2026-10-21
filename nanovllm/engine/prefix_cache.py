from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from struct import pack

from nanovllm.engine.schedule import GDN_CHECKPOINT_INTERVAL
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


def _block_hashes(
    tokens: list[int] | tuple[int, ...],
    block_size: int,
    stop: int | None = None,
    *,
    cached_hashes: list[bytes] | None = None,
):
    """Yield chained SHA-256 hashes, computing only new complete blocks when cached."""
    parent = bytes(32)
    limit = min(stop, len(tokens)) if stop is not None else len(tokens)
    for index, start in enumerate(range(0, limit - block_size + 1, block_size)):
        if cached_hashes is not None and index < len(cached_hashes):
            parent = cached_hashes[index]
        else:
            parent = sha256(
                parent + pack(f"<{block_size}I", *tokens[start : start + block_size])
            ).digest()
            if cached_hashes is not None:
                cached_hashes.append(parent)
        yield start + block_size, parent


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
        self._retain_blocks = retain_blocks
        self._release_blocks = release_blocks
        self._entries: dict[bytes, PrefixEntry] = {}
        self._reused: set[bytes] = set()

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
        cached = []
        for position, digest in _block_hashes(
            seq.token_ids, self.block_size, end, cached_hashes=seq.prefix_block_hashes
        ):
            if position % GDN_CHECKPOINT_INTERVAL:
                continue
            if digest in self._entries:
                cached.append(position)
            elif position > start:
                candidates.append((position, digest))
        return candidates, cached

    def _path_limit(self) -> int:
        limit = min(self.max_entries, self.max_checkpoints_per_path)
        if self.max_snapshot_bytes is not None and self.snapshot_nbytes:
            limit = min(limit, self.max_snapshot_bytes // self.snapshot_nbytes)
        return limit

    def select_checkpoints(
        self, candidates: list[tuple[int, bytes]], cached: list[int]
    ) -> list[tuple[int, bytes]]:
        """Choose useful checkpoints within the effective global/per-path capacity."""
        if not candidates:
            return []
        limit = self._path_limit()
        if limit <= 0:
            return []
        if limit == 1 and cached and cached[-1] > candidates[-1][0]:
            return []
        if limit == 1 or cached:
            return candidates[-1:]
        return candidates[:1] if len(candidates) == 1 else [candidates[0], candidates[-1]]

    def find(self, seq: Sequence, max_tokens: int) -> PrefixEntry | None:
        if not self.enabled or max_tokens <= 0:
            return None
        if seq.image_state is not None:
            max_tokens = min(max_tokens, seq.image_state.image_start)
        best_entry: PrefixEntry | None = None
        for _, digest in _block_hashes(
            seq.token_ids, self.block_size, max_tokens, cached_hashes=seq.prefix_block_hashes
        ):
            entry = self._entries.get(digest)
            if entry is not None:
                best_entry = entry
        return best_entry

    def mark_reused(self, seq: Sequence, entry: PrefixEntry) -> None:
        """Protect only prefixes actually attached to an admitted request."""
        digest_index = entry.num_tokens // self.block_size - 1
        self._reused.add(seq.prefix_block_hashes[digest_index])

    def publish_many(self, seq: Sequence, snapshots: tuple[GDNStateSnapshot, ...]) -> None:
        if not snapshots or not self._can_capture():
            return
        by_position = {snapshot.num_tokens: snapshot for snapshot in snapshots}
        if len(by_position) != len(snapshots) or any(
            position <= 0
            or position > seq.num_prompt_tokens
            or position % self.block_size
            or position % GDN_CHECKPOINT_INTERVAL
            for position in by_position
        ):
            raise RuntimeError("invalid GDN snapshot checkpoint boundary")
        path = []
        for position, digest in _block_hashes(
            seq.token_ids, self.block_size, max(by_position), cached_hashes=seq.prefix_block_hashes
        ):
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

        self._retain_blocks(block_ids)
        block_hashes = tuple(
            digest for _, digest in _block_hashes(
                seq.token_ids, self.block_size, position, cached_hashes=seq.prefix_block_hashes
            )
        )
        entry = PrefixEntry(block_hashes, block_ids, position, snapshot)
        self._entries[digest] = entry
        self._snapshot_bytes += self.snapshot_nbytes

        path.append(digest)
        limit = self._path_limit()
        # Preserve checkpoints actually reused by other requests when growing a
        # branch; prefer removing intermediate, unshared positions.
        if len(path) > limit:
            interior = path[1:-1]
            victim = next((key for key in interior if key not in self._reused), None)
            if victim is None:
                victim = interior[0] if interior else path[0]
            self._evict_hash(victim)
            path.remove(victim)

        # An out-of-order insertion may make existing descendants exceed their
        # path quota. Reject the new intermediate node if no old interior can go.
        for leaf in tuple(self._entries.values()):
            if leaf.num_tokens <= position or leaf.block_hashes[num_blocks - 1] != digest:
                continue
            chain = [key for key in leaf.block_hashes if key in self._entries]
            if len(chain) > limit:
                interior = [key for key in chain[1:-1] if key != digest]
                if not interior:
                    self._evict_hash(digest)
                    if digest in path:
                        path.remove(digest)
                    return
                victim = next((key for key in interior if key not in self._reused), None)
                if victim is None:
                    self._evict_hash(digest)
                    if digest in path:
                        path.remove(digest)
                    return
                self._evict_hash(victim)

        # Replace intermediate checkpoints before evicting globally.
        while len(self._entries) > self.max_entries or (
            self.max_snapshot_bytes is not None and self._snapshot_bytes > self.max_snapshot_bytes
        ):
            if len(path) >= 3:
                interior = path[1:-1]
                victim = next((key for key in interior if key not in self._reused), interior[-1])
                self._evict_hash(victim)
                path.remove(victim)
            else:
                self.evict()
        path[:] = [key for key in path if key in self._entries]

    def _evict_hash(self, digest: bytes) -> None:
        entry = self._entries.pop(digest)
        self._snapshot_bytes -= self.snapshot_nbytes
        self._reused.discard(digest)
        self._release_blocks(entry.block_ids)

    def evict(self, block_refcounts: list[int] | None = None) -> bool:
        if not self._entries:
            return False
        if block_refcounts is None:
            victim = next(iter(self._entries))
        else:
            victim = max(
                self._entries,
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
