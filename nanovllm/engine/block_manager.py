from collections import deque

from nanovllm.engine.sequence import Sequence


class BlockManager:
    """Paged-KV block allocator without cross-request prefix sharing."""

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

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        if seq.block_table:
            raise RuntimeError("sequence already owns KV blocks")
        if not self.can_allocate(seq):
            raise RuntimeError("insufficient KV cache blocks")
        seq.block_table.extend(
            self._allocate_block()
            for _ in range(seq.num_blocks)
        )
        seq.num_cached_tokens = 0

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
        needs_new_block = len(seq) % self.block_size == 1
        return not needs_new_block or bool(self.free_block_ids)

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())
