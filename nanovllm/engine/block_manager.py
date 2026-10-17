from collections import deque
from collections.abc import Iterable

from nanovllm.engine.prefix_cache import PrefixCache
from nanovllm.engine.sequence import Sequence


class BlockManager:
    """Paged-KV allocator with request-local growth and shared-prefix refs."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        max_prefix_entries: int = 0,
        max_prefix_cache_snapshot_bytes: int | None = None,
        snapshot_nbytes: int | None = None,
    ) -> None:
        if num_blocks <= 0 or max_prefix_entries < 0:
            raise ValueError("num_blocks must be positive and max_prefix_entries non-negative")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.block_refcounts = [0] * num_blocks
        self.prefix_cache = PrefixCache(
            block_size,
            max_prefix_entries,
            self.retain_blocks,
            self.release_blocks,
            max_snapshot_bytes=max_prefix_cache_snapshot_bytes,
            snapshot_nbytes=snapshot_nbytes,
        )

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("no free KV cache blocks")
        block_id = self.free_block_ids.popleft()
        self.block_refcounts[block_id] = 1
        return block_id

    def required_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def retain_blocks(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            self.block_refcounts[block_id] += 1

    def release_blocks(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            refcount = self.block_refcounts[block_id] - 1
            if refcount < 0:
                raise RuntimeError(f"KV block {block_id} released twice")
            self.block_refcounts[block_id] = refcount
            if refcount == 0:
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
            or len(block_ids) != self.required_blocks(num_tokens)
        ):
            raise RuntimeError("invalid shared KV prefix blocks or boundary")
        self.retain_blocks(block_ids)
        seq.block_table.extend(block_ids)

    def additional_blocks_needed(self, seq: Sequence, target_tokens: int) -> int:
        return max(0, self.required_blocks(target_tokens) - len(seq.block_table))

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
        missing = self.required_blocks(target_tokens) - len(seq.block_table)
        if missing <= 0:
            return
        if missing > len(self.free_block_ids):
            raise RuntimeError("insufficient KV cache blocks")
        seq.block_table.extend(self._allocate_block() for _ in range(missing))

    def deallocate(self, seq: Sequence) -> None:
        if seq.block_table:
            self.release_blocks(seq.block_table)
            seq.block_table.clear()
