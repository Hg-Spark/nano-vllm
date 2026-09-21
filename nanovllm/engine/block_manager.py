from collections import deque

from nanovllm.engine.sequence import Sequence


class BlockManager:
    """Paged-KV allocator with request-local growth and shared-prefix refs."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.block_size = block_size
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.block_refcounts = [0] * num_blocks

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("no free KV cache blocks")
        block_id = self.free_block_ids.popleft()
        if self.block_refcounts[block_id] != 0:
            raise RuntimeError(
                f"free KV block {block_id} has non-zero refcount"
            )
        self.block_refcounts[block_id] = 1
        return block_id

    def _required_blocks(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return (num_tokens + self.block_size - 1) // self.block_size

    def retain_blocks(self, block_ids) -> None:
        ids = tuple(block_ids)
        for block_id in ids:
            if not 0 <= block_id < len(self.block_refcounts):
                raise RuntimeError(f"invalid KV block {block_id}")
            if self.block_refcounts[block_id] <= 0:
                raise RuntimeError(
                    f"KV block {block_id} is not allocated"
                )
        for block_id in ids:
            self.block_refcounts[block_id] += 1

    def release_blocks(self, block_ids) -> None:
        ids = tuple(block_ids)
        for block_id in ids:
            if not 0 <= block_id < len(self.block_refcounts):
                raise RuntimeError(f"invalid KV block {block_id}")
            if self.block_refcounts[block_id] <= 0:
                raise RuntimeError(
                    f"KV block {block_id} is not allocated"
                )

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
            raise RuntimeError(
                "shared prefix can only attach to a fresh request"
            )
        if num_tokens <= 0 or num_tokens % self.block_size != 0:
            raise RuntimeError(
                "shared KV prefix must end on a full block boundary"
            )
        required = self._required_blocks(num_tokens)
        if len(block_ids) != required:
            raise RuntimeError(
                "shared prefix block count does not match token length"
            )
        if len(set(block_ids)) != len(block_ids):
            raise RuntimeError("shared prefix contains duplicate KV blocks")

        self.retain_blocks(block_ids)
        seq.block_table.extend(block_ids)

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
        available = max_backed_tokens - seq.committed_tokens
        return max(0, min(requested_tokens, available))

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
        self.release_blocks(seq.block_table)
        seq.block_table.clear()
