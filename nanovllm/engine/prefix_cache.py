from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from struct import pack

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot

PrefixCacheKey = tuple[int, ...]


def _block_hashes(tokens: list[int] | tuple[int, ...], block_size: int):
    """Chain full KV-block hashes so equal blocks require equal preceding context."""
    parent = bytes(32)
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = sha256(
            parent + pack(f"<{block_size}I", *tokens[start : start + block_size])
        ).digest()
        yield parent


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    block_ids: tuple[int, ...]
    num_tokens: int
    state_snapshot: GDNStateSnapshot
    block_hashes: tuple[bytes, ...]


class PrefixCache:
    """Separate full-block hash and GDN checkpoint lookup; joint-state ownership."""

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
        self._entries: dict[PrefixCacheKey, PrefixEntry] = {}
        self._kv_hash_refs: dict[bytes, int] = {}
        self._gdn_checkpoints: dict[bytes, PrefixEntry] = {}

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def snapshot_bytes(self) -> int:
        return self._snapshot_bytes

    @staticmethod
    def _is_text_prefix(seq: Sequence, prefix_tokens: int) -> bool:
        return seq.image_state is None or prefix_tokens <= seq.image_state.image_start

    def find(self, seq: Sequence, max_tokens: int) -> PrefixEntry | None:
        if not self.enabled or max_tokens <= 0:
            return None
        if seq.image_state is not None:
            max_tokens = min(max_tokens, seq.image_state.image_start)
        best_entry: PrefixEntry | None = None
        for block_hash in _block_hashes(seq.token_ids[:max_tokens], self.block_size):
            if block_hash not in self._kv_hash_refs:
                break
            snapshot_entry = self._gdn_checkpoints.get(block_hash)
            if snapshot_entry is not None:
                best_entry = snapshot_entry
        return best_entry

    def should_publish(self, seq: Sequence, end: int) -> bool:
        if not self.enabled or self.max_snapshot_bytes == 0 or end <= seq.committed_tokens:
            return False
        if self.max_snapshot_bytes is not None and self.snapshot_nbytes > self.max_snapshot_bytes:
            return False
        if end >= seq.num_prompt_tokens or end % self.block_size != 0:
            return False
        if not self._is_text_prefix(seq, end):
            return False
        return tuple(seq.token_ids[:end]) not in self._entries

    def publish(
        self,
        seq: Sequence,
        state_snapshot: GDNStateSnapshot,
        prefix_tokens: int,
    ) -> None:
        if not self.enabled or self.max_snapshot_bytes == 0:
            return
        if state_snapshot.num_tokens != prefix_tokens:
            raise RuntimeError("GDN snapshot boundary does not match KV prefix boundary")

        key = tuple(seq.token_ids[:prefix_tokens])
        if key in self._entries:
            return

        if self.max_snapshot_bytes is not None and self.snapshot_nbytes > self.max_snapshot_bytes:
            return

        num_blocks = prefix_tokens // self.block_size
        block_ids = tuple(seq.block_table[:num_blocks])
        if len(block_ids) != num_blocks:
            raise RuntimeError("KV block table is shorter than cached prefix")

        hashes = tuple(_block_hashes(key, self.block_size))
        self._retain_blocks(block_ids)
        entry = PrefixEntry(
            block_ids=block_ids,
            num_tokens=prefix_tokens,
            state_snapshot=state_snapshot,
            block_hashes=hashes,
        )
        self._entries[key] = entry
        for block_hash in hashes:
            self._kv_hash_refs[block_hash] = self._kv_hash_refs.get(block_hash, 0) + 1
        self._gdn_checkpoints[hashes[-1]] = entry
        self._snapshot_bytes += self.snapshot_nbytes
        # A new entry can only increase the depth of its descendant chains.
        for leaf in self._entries:
            if leaf[: len(key)] != key:
                continue
            chain = [prefix for prefix in self._entries if leaf[: len(prefix)] == prefix]
            if len(chain) > self.max_checkpoints_per_path:
                self._evict_key(min(chain, key=len))
                break
        while len(self._entries) > self.max_entries or (
            self.max_snapshot_bytes is not None and self._snapshot_bytes > self.max_snapshot_bytes
        ):
            self.evict()

    def _evict_key(self, key: PrefixCacheKey) -> None:
        entry = self._entries.pop(key)
        del self._gdn_checkpoints[entry.block_hashes[-1]]
        for block_hash in entry.block_hashes:
            refs = self._kv_hash_refs[block_hash] - 1
            if refs:
                self._kv_hash_refs[block_hash] = refs
            else:
                del self._kv_hash_refs[block_hash]
        self._snapshot_bytes -= self.snapshot_nbytes
        self._release_blocks(entry.block_ids)

    def evict(self) -> bool:
        if not self._entries:
            return False
        # The globally longest prefix is necessarily a leaf of cached paths.
        self._evict_key(max(self._entries, key=len))
        return True
