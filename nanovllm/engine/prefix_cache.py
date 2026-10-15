from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot

PrefixCacheKey = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    block_ids: tuple[int, ...]
    num_tokens: int
    state_snapshot: GDNStateSnapshot


class PrefixCache:
    """Exact-match LRU metadata bound to shared text-prefix KV block references."""

    def __init__(
        self,
        block_size: int,
        max_entries: int,
        retain_blocks: Callable[[Iterable[int]], None],
        release_blocks: Callable[[Iterable[int]], None],
    ) -> None:
        self.block_size = block_size
        self.max_entries = max_entries
        self._retain_blocks = retain_blocks
        self._release_blocks = release_blocks
        self._entries: OrderedDict[PrefixCacheKey, PrefixEntry] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _is_text_prefix(seq: Sequence, prefix_tokens: int) -> bool:
        return seq.image_state is None or prefix_tokens <= seq.image_state.image_start

    def find(self, seq: Sequence, max_tokens: int) -> PrefixEntry | None:
        if not self.enabled or max_tokens <= 0:
            return None
        if seq.image_state is not None:
            max_tokens = min(max_tokens, seq.image_state.image_start)
        tokens = tuple(seq.token_ids[:max_tokens])

        best_key: PrefixCacheKey | None = None
        best_entry: PrefixEntry | None = None
        for key, entry in self._entries.items():
            if entry.num_tokens <= len(tokens) and tokens[: entry.num_tokens] == key:
                if best_entry is None or entry.num_tokens > best_entry.num_tokens:
                    best_key = key
                    best_entry = entry

        if best_key is not None:
            self._entries.move_to_end(best_key)
        return best_entry

    def should_publish(self, seq: Sequence, end: int) -> bool:
        if not self.enabled or end <= seq.committed_tokens:
            return False
        if end > seq.num_prompt_tokens or end % self.block_size != 0:
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
        if not self.enabled:
            return
        if (
            prefix_tokens <= 0
            or prefix_tokens > seq.num_prompt_tokens
            or prefix_tokens % self.block_size != 0
        ):
            raise RuntimeError("prefix cache requires a valid full-block prompt boundary")
        if state_snapshot.num_tokens != prefix_tokens:
            raise RuntimeError("GDN snapshot boundary does not match KV prefix boundary")
        if not self._is_text_prefix(seq, prefix_tokens):
            raise RuntimeError("prefix cache supports text prefixes before image tokens only")

        key = tuple(seq.token_ids[:prefix_tokens])
        if key in self._entries:
            self._entries.move_to_end(key)
            return

        num_blocks = prefix_tokens // self.block_size
        block_ids = tuple(seq.block_table[:num_blocks])
        if len(block_ids) != num_blocks:
            raise RuntimeError("KV block table is shorter than cached prefix")

        self._retain_blocks(block_ids)
        self._entries[key] = PrefixEntry(
            block_ids=block_ids,
            num_tokens=prefix_tokens,
            state_snapshot=state_snapshot,
        )
        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            self._release_blocks(evicted.block_ids)

    def evict(self) -> bool:
        if not self._entries:
            return False
        _, entry = self._entries.popitem(last=False)
        self._release_blocks(entry.block_ids)
        return True
