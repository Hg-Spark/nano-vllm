from collections import Counter, deque
from collections.abc import Iterable

from nanovllm.engine.prefix_cache import PrefixCache, PrefixEntry
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


class BlockManager:
    """Paged-KV allocator with request-local growth and shared-prefix refs."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        max_prefix_entries: int = 0,
    ) -> None:
        if num_blocks <= 0 or max_prefix_entries < 0:
            raise ValueError("num_blocks must be positive and max_prefix_entries non-negative")
        self.block_size = block_size
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.block_refcounts = [0] * num_blocks
        self.prefix_cache = PrefixCache(
            block_size,
            max_prefix_entries,
            self.retain_blocks,
            self.release_blocks,
        )

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("no free KV cache blocks")
        block_id = self.free_block_ids.popleft()
        self.block_refcounts[block_id] = 1
        return block_id

    def _required_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _checked_block_ids(self, block_ids: Iterable[int]) -> tuple[int, ...]:
        ids = tuple(block_ids)
        for block_id in ids:
            if not 0 <= block_id < len(self.block_refcounts) or self.block_refcounts[block_id] <= 0:
                raise RuntimeError(f"invalid or unallocated KV block {block_id}")
        return ids

    def retain_blocks(self, block_ids: Iterable[int]) -> None:
        for block_id in self._checked_block_ids(block_ids):
            self.block_refcounts[block_id] += 1

    def release_blocks(self, block_ids: Iterable[int]) -> None:
        ids = self._checked_block_ids(block_ids)
        for block_id, count in Counter(ids).items():
            if self.block_refcounts[block_id] < count:
                raise RuntimeError(f"KV block {block_id} release exceeds its refcount")
        for block_id in ids:
            self.block_refcounts[block_id] -= 1
            if self.block_refcounts[block_id] == 0:
                self.free_block_ids.append(block_id)

    def attach_shared_prefix(
        self,
        seq: Sequence,
        block_ids: tuple[int, ...],
        num_tokens: int,
    ) -> None:
        if seq.block_table or seq.committed_tokens != 0:
            raise RuntimeError("shared prefix can only attach to a fresh request")
        if (
            num_tokens <= 0
            or num_tokens % self.block_size != 0
            or len(block_ids) != self._required_blocks(num_tokens)
            or len(set(block_ids)) != len(block_ids)
        ):
            raise RuntimeError("invalid shared KV prefix blocks or boundary")
        self.retain_blocks(block_ids)
        seq.block_table.extend(block_ids)

    @property
    def num_cached_prefixes(self) -> int:
        return len(self.prefix_cache)

    def prefix_entries(self) -> tuple[PrefixEntry, ...]:
        return self.prefix_cache.entries()

    def find_prefix(self, seq: Sequence, max_tokens: int) -> PrefixEntry | None:
        return self.prefix_cache.find(seq, max_tokens)

    def should_cache_prefix(self, seq: Sequence, end: int) -> bool:
        return self.prefix_cache.should_publish(seq, end)

    def publish_prefix(
        self, seq: Sequence, state_snapshot: GDNStateSnapshot, prefix_tokens: int
    ) -> None:
        self.prefix_cache.publish(seq, state_snapshot, prefix_tokens)

    def evict_prefix(self) -> bool:
        return self.prefix_cache.evict()

    def additional_blocks_needed(self, seq: Sequence, target_tokens: int) -> int:
        return max(0, self._required_blocks(target_tokens) - len(seq.block_table))

    def max_schedulable_tokens(
        self,
        seq: Sequence,
        requested_tokens: int,
        reserved_free_blocks: int = 0,
    ) -> int:
        if requested_tokens <= 0:
            return 0
        usable_free = max(0, len(self.free_block_ids) - reserved_free_blocks)
        available = (len(seq.block_table) + usable_free) * self.block_size - seq.committed_tokens
        return max(0, min(requested_tokens, available))

    def ensure_capacity(self, seq: Sequence, target_tokens: int) -> None:
        missing = self._required_blocks(target_tokens) - len(seq.block_table)
        if missing <= 0:
            return
        if missing > len(self.free_block_ids):
            raise RuntimeError("insufficient KV cache blocks")
        seq.block_table.extend(self._allocate_block() for _ in range(missing))

    def truncate_blocks(self, seq: Sequence, keep_num_blocks: int) -> None:
        released = tuple(seq.block_table[keep_num_blocks:])
        if released:
            self.release_blocks(released)
            del seq.block_table[keep_num_blocks:]

    def deallocate(self, seq: Sequence) -> None:
        if seq.block_table:
            self.release_blocks(seq.block_table)
            seq.block_table.clear()
