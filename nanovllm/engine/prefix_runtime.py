from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.prefix_cache import JointPrefixCache, JointPrefixEntry
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import (
    GDNStateSnapshot,
    StateSlotManager,
)


class PrefixRuntime:
    """Own the lifecycle of reusable hybrid prompt prefixes."""

    def __init__(
        self,
        max_entries: int,
        block_size: int,
        block_manager: BlockManager,
        state_manager: StateSlotManager,
    ):
        self.block_size = block_size
        self.block_manager = block_manager
        self.state_manager = state_manager
        self.cache = JointPrefixCache(max_entries)

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def enabled(self) -> bool:
        return self.cache.max_entries > 0

    def evict_one(self) -> bool:
        entry = self.cache.pop_lru()
        if entry is None:
            return False
        self.block_manager.release_blocks(entry.block_ids)
        return True

    def try_restore(self, seq: Sequence) -> bool:
        if not self.enabled:
            return False
        if (
            seq.committed_tokens != 0
            or seq.block_table
            or seq.state_slot >= 0
        ):
            return False

        # Keep at least one token uncached because prompt logits are not cached.
        entry = self.cache.longest_match(
            seq.token_ids,
            max_tokens=max(0, seq.num_tokens - 1),
        )
        if entry is None:
            return False
        if not self.state_manager.can_allocate(seq):
            return False

        self.state_manager.allocate(seq)
        try:
            self.block_manager.attach_shared_prefix(
                seq,
                entry.block_ids,
                entry.num_tokens,
            )
        except Exception:
            self.state_manager.deallocate(seq)
            raise

        seq.committed_tokens = entry.num_tokens
        seq.pending_state_snapshot = entry.state_snapshot
        return True

    def should_snapshot_after_step(self, seq: Sequence) -> bool:
        if not self.enabled:
            return False
        if seq.num_scheduled_tokens <= 0 or seq.state_slot < 0:
            return False

        target_tokens = (
            seq.committed_tokens + seq.num_scheduled_tokens
        )
        if target_tokens <= 0:
            return False
        if target_tokens > seq.num_prompt_tokens:
            return False
        if target_tokens % self.block_size != 0:
            return False

        key = tuple(seq.token_ids[:target_tokens])
        return not self.cache.contains(key)

    def publish(
        self,
        seq: Sequence,
        state_snapshot: GDNStateSnapshot,
    ) -> None:
        prefix_tokens = seq.committed_tokens
        if prefix_tokens <= 0:
            raise RuntimeError("cannot cache an empty prefix")
        if prefix_tokens > seq.num_prompt_tokens:
            raise RuntimeError(
                "joint prefix cache stores prompt prefixes only"
            )
        if prefix_tokens % self.block_size != 0:
            raise RuntimeError(
                "joint prefix cache requires a full KV block boundary"
            )
        if state_snapshot.num_tokens != prefix_tokens:
            raise RuntimeError(
                "GDN snapshot boundary does not match KV prefix boundary"
            )

        key = tuple(seq.token_ids[:prefix_tokens])
        if self.cache.contains(key):
            return

        num_blocks = prefix_tokens // self.block_size
        block_ids = tuple(seq.block_table[:num_blocks])
        if len(block_ids) != num_blocks:
            raise RuntimeError(
                "KV block table is shorter than cached prefix"
            )

        # Cache ownership is independent from the request's block references.
        self.block_manager.retain_blocks(block_ids)
        entry = JointPrefixEntry(
            token_ids=key,
            block_ids=block_ids,
            num_tokens=prefix_tokens,
            state_snapshot=state_snapshot,
        )
        try:
            released = self.cache.put(entry)
        except Exception:
            self.block_manager.release_blocks(block_ids)
            raise

        for old_entry in released:
            self.block_manager.release_blocks(old_entry.block_ids)
