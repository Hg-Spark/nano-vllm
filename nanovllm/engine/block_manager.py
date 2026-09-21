from collections import deque

from nanovllm.engine.sequence import Sequence


class BlockManager:
    """Paged-KV allocator with request-local incremental growth."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.block_size = block_size
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("no free KV cache blocks")
        block_id = self.free_block_ids.popleft()
        self.used_block_ids.add(block_id)
        return block_id

    def _required_blocks(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return (num_tokens + self.block_size - 1) // self.block_size

    def max_schedulable_tokens(
        self,
        seq: Sequence,
        requested_tokens: int,
    ) -> int:
        """Return how many new tokens can be backed by current + free blocks."""
        if requested_tokens <= 0:
            return 0
        max_backed_tokens = (
            len(seq.block_table) + len(self.free_block_ids)
        ) * self.block_size
        available = max_backed_tokens - seq.num_cached_tokens
        return max(0, min(requested_tokens, available))

    def can_ensure_capacity(
        self,
        seq: Sequence,
        target_tokens: int,
    ) -> bool:
        required = self._required_blocks(target_tokens)
        missing = max(0, required - len(seq.block_table))
        return missing <= len(self.free_block_ids)

    def ensure_capacity(
        self,
        seq: Sequence,
        target_tokens: int,
    ) -> None:
        """Grow a request's KV block table only up to target_tokens."""
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        required = self._required_blocks(target_tokens)
        missing = required - len(seq.block_table)
        if missing <= 0:
            return
        if missing > len(self.free_block_ids):
            raise RuntimeError("insufficient KV cache blocks")
        seq.block_table.extend(
            self._allocate_block()
            for _ in range(missing)
        )

    def deallocate(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            if block_id not in self.used_block_ids:
                raise RuntimeError(
                    f"KV block {block_id} is not allocated"
                )
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        target_tokens = len(seq)
        return self.can_ensure_capacity(seq, target_tokens)

    def may_append(self, seq: Sequence) -> None:
        self.ensure_capacity(seq, len(seq))
