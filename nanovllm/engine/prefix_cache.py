from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


PrefixCacheKey = tuple[
    tuple[int, ...],
    str | None,
    tuple[int, int, int] | None,
]


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    num_tokens: int
    state_snapshot: GDNStateSnapshot
    image_fingerprint: str | None = None
    image_grid: tuple[int, int, int] | None = None


class PrefixCache:
    """Exact-match LRU metadata bound to shared KV block references."""

    def __init__(
        self,
        block_size: int,
        max_entries: int,
        retain_blocks: Callable,
        release_blocks: Callable,
    ):
        self.block_size = block_size
        self.max_entries = max_entries
        self._retain_blocks = retain_blocks
        self._release_blocks = release_blocks
        self._entries: OrderedDict[
            PrefixCacheKey,
            PrefixEntry,
        ] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[PrefixEntry, ...]:
        return tuple(self._entries.values())

    @staticmethod
    def _identity(
        seq: Sequence,
        prefix_tokens: int,
    ) -> tuple[str | None, tuple[int, int, int] | None] | None:
        state = seq.image_state
        if state is None or prefix_tokens <= state.image_start:
            return (None, None)
        if prefix_tokens < state.image_end:
            return None
        return (state.fingerprint, state.grid_signature)

    def find(
        self,
        seq: Sequence,
        max_tokens: int,
    ) -> PrefixEntry | None:
        if not self.enabled or max_tokens <= 0:
            return None

        best_key: PrefixCacheKey | None = None
        best_entry: PrefixEntry | None = None
        tokens = tuple(seq.token_ids)
        for key, entry in self._entries.items():
            if entry.num_tokens > max_tokens or entry.num_tokens > len(tokens):
                continue
            if tokens[:entry.num_tokens] != entry.token_ids:
                continue
            identity = self._identity(seq, entry.num_tokens)
            if identity is None:
                continue
            if (
                entry.image_fingerprint,
                entry.image_grid,
            ) != identity:
                continue
            if best_entry is None or entry.num_tokens > best_entry.num_tokens:
                best_key = key
                best_entry = entry

        if best_key is not None:
            self._entries.move_to_end(best_key)
        return best_entry

    def should_publish(
        self,
        seq: Sequence,
        end: int,
    ) -> bool:
        if not self.enabled:
            return False
        if end <= seq.committed_tokens or seq.state_slot < 0:
            return False
        if end > seq.num_prompt_tokens or end % self.block_size != 0:
            return False

        identity = self._identity(seq, end)
        if identity is None:
            return False
        key: PrefixCacheKey = (
            tuple(seq.token_ids[:end]),
            identity[0],
            identity[1],
        )
        return key not in self._entries

    def publish(
        self,
        seq: Sequence,
        state_snapshot: GDNStateSnapshot,
        prefix_tokens: int,
    ) -> None:
        if not self.enabled:
            return
        if prefix_tokens <= 0:
            raise RuntimeError("cannot cache an empty prefix")
        if prefix_tokens > seq.num_prompt_tokens:
            raise RuntimeError("prefix cache stores prompt prefixes only")
        if prefix_tokens % self.block_size != 0:
            raise RuntimeError(
                "prefix cache requires a full KV block boundary"
            )
        if state_snapshot.num_tokens != prefix_tokens:
            raise RuntimeError(
                "GDN snapshot boundary does not match KV prefix boundary"
            )

        identity = self._identity(seq, prefix_tokens)
        if identity is None:
            raise RuntimeError(
                "cannot cache a prefix inside an image feature interval"
            )
        key: PrefixCacheKey = (
            tuple(seq.token_ids[:prefix_tokens]),
            identity[0],
            identity[1],
        )
        if key in self._entries:
            self._entries.move_to_end(key)
            return

        num_blocks = prefix_tokens // self.block_size
        block_ids = tuple(seq.block_table[:num_blocks])
        if len(block_ids) != num_blocks:
            raise RuntimeError(
                "KV block table is shorter than cached prefix"
            )

        self._retain_blocks(block_ids)
        entry = PrefixEntry(
            token_ids=key[0],
            block_ids=block_ids,
            num_tokens=prefix_tokens,
            state_snapshot=state_snapshot,
            image_fingerprint=identity[0],
            image_grid=identity[1],
        )
        try:
            self._entries[key] = entry
        except Exception:
            self._release_blocks(block_ids)
            raise

        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            self._release_blocks(evicted.block_ids)

    def evict(self) -> bool:
        if not self._entries:
            return False
        _, entry = self._entries.popitem(last=False)
        self._release_blocks(entry.block_ids)
        return True
