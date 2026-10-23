"""Logical ownership of paged KV, recurrent state slots, and joint prefix cache."""

from collections.abc import Iterable
from math import lcm

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.prefix_cache import PrefixCache, PrefixEntry
from nanovllm.engine.schedule import GDN_CHECKPOINT_INTERVAL
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot, StateSlotManager


class HybridCacheManager:
    """Coordinate KV pages and GDN slots at one committed token boundary.

    Logical admission and prefix metadata live here. Physical GPU storage
    remains owned by ModelRunner and the GDN layer state pools.
    """

    def __init__(self, config: Config, num_kv_cache_blocks: int, snapshot_nbytes: int = 0) -> None:
        self.block_manager = BlockManager(num_kv_cache_blocks, config.kv_cache_block_size)
        self.state_manager = StateSlotManager(config.max_num_seqs)
        # PrefixCache references the allocator through retain/release callbacks,
        # but its lifecycle and active reuse protection belong to this manager.
        self.prefix_cache = PrefixCache(
            config.kv_cache_block_size,
            config.max_prefix_cache_entries,
            self.block_manager.retain_blocks,
            self.block_manager.release_blocks,
            max_snapshot_bytes=config.max_prefix_cache_snapshot_bytes,
            snapshot_nbytes=snapshot_nbytes,
            max_checkpoints_per_path=config.max_gdn_checkpoints,
        )

    @property
    def total_blocks(self) -> int:
        return self.block_manager.num_blocks

    def required_blocks(self, seq: Sequence) -> int:
        """Admission reservation for the prompt and worst-case generated tokens."""
        tokens = seq.num_prompt_tokens + seq.max_tokens - 1
        return self.block_manager.required_blocks(tokens)

    def reservation_usage(self, active: Iterable[Sequence]) -> tuple[set[int], int]:
        """Return unique occupied pages and worst-case active KV reservations."""
        active = tuple(active)
        if not self.prefix_cache.enabled:
            # Without shared prefixes, count requests directly and skip the
            # full resident page-set construction on every decode step.
            return set(), sum(self.required_blocks(seq) for seq in active)

        occupied = {block_id for seq in active for block_id in seq.block_table}
        reserved = len(occupied) + sum(
            self.required_blocks(seq) - len(seq.block_table) for seq in active
        )
        return occupied, reserved

    def find_prefix(self, seq: Sequence) -> PrefixEntry | None:
        return self.prefix_cache.find(seq, max_tokens=len(seq) - 1)

    def admission_blocks(
        self,
        seq: Sequence,
        entry: PrefixEntry | None,
        occupied_blocks: set[int],
    ) -> int:
        """Incremental KV reservation accounting for physical prefix sharing."""
        blocks = self.required_blocks(seq)
        if entry is not None and entry.block_ids:
            blocks -= len(entry.block_ids)
            blocks += len(set(entry.block_ids) - occupied_blocks)
        return blocks

    def admit(self, seq: Sequence, entry: PrefixEntry | None) -> None:
        """Attach a joint cached prefix, then allocate the recurrent state slot."""
        try:
            if entry is not None:
                self.block_manager.attach_shared_prefix(seq, entry.block_ids, entry.num_tokens)
                self.prefix_cache.mark_reused(seq, entry)
                seq.committed_tokens = entry.num_tokens
                seq.pending_state_snapshot = entry.state_snapshot
            self.state_manager.allocate(seq)
        except Exception:
            # Never retain shared KV refs if GDN slot admission fails.
            self.prefix_cache.unmark_reused(seq)
            self.block_manager.deallocate(seq)
            self.state_manager.deallocate(seq)
            seq.committed_tokens = 0
            seq.pending_state_snapshot = None
            raise

    def fit_prefill(self, seq: Sequence, requested_tokens: int) -> int:
        """Evict expendable checkpoints until a reserved prefill can progress."""
        scheduled = self.block_manager.max_schedulable_tokens(seq, requested_tokens)
        while scheduled < requested_tokens and self.prefix_cache.evict(
            self.block_manager.block_refcounts
        ):
            scheduled = self.block_manager.max_schedulable_tokens(seq, requested_tokens)
        if scheduled == 0:
            raise RuntimeError("admitted prefill cannot acquire reserved KV blocks")
        return scheduled

    def plan_prefill_checkpoints(
        self,
        seq: Sequence,
        start: int,
        end: int,
        planned_prefixes: set[bytes],
    ) -> tuple[int, tuple[int, ...], int]:
        """Choose valid joint KV/GDN checkpoint boundaries for a prefill chunk."""
        step = lcm(GDN_CHECKPOINT_INTERVAL, self.block_manager.block_size)
        if not self.prefix_cache.enabled:
            return end, (), 0

        interval = GDN_CHECKPOINT_INTERVAL
        candidates, cached = self.prefix_cache.candidates(seq, start, end)
        candidates = [pair for pair in candidates if pair[1] not in planned_prefixes]
        scheduled = end - start

        if end < len(seq):
            # Intermediate FlashInfer GDN checkpoints require an aligned start.
            # Recover an off-grid chunk at a useful joint boundary, while
            # avoiding tiny chunks at every subsequent checkpoint position.
            selected = self.prefix_cache.select_checkpoints(candidates, cached)
            max_trim = min(
                step - 1,
                scheduled // 4,
                max(16, scheduled // 8),
            )
            target = None
            if (
                start % step
                and selected
                and (
                    not cached
                    or start - cached[-1] >= step * self.prefix_cache.max_checkpoints_per_path
                )
            ):
                target = selected[-1][0]
            else:
                target = next(
                    (
                        position
                        for position, _ in reversed(selected)
                        if position < end and end - position <= max_trim
                    ),
                    None,
                )
            if target is not None and target < end:
                end = target

        candidates = [pair for pair in candidates if pair[0] <= end]
        if start % step:
            # An off-grid chunk can only publish its final live state,
            # and only when that endpoint lies on the joint grid.
            candidates = [pair for pair in candidates if pair[0] == end]
        candidates = self.prefix_cache.select_checkpoints(candidates, cached)
        positions = tuple(position for position, _ in candidates)
        planned_prefixes.update(digest for _, digest in candidates)

        return end, positions, interval if any(position < end for position in positions) else 0

    def reserve_prefill(self, seq: Sequence, end: int) -> None:
        self.block_manager.ensure_capacity(seq, end)

    def reserve_decode(self, seq: Sequence) -> None:
        while self.block_manager.max_schedulable_tokens(seq, 1) == 0:
            if not self.prefix_cache.evict(self.block_manager.block_refcounts):
                raise RuntimeError("admitted decode cannot acquire reserved KV blocks")
        self.block_manager.ensure_capacity(seq, len(seq))

    def publish_prefixes(self, seq: Sequence, snapshots: tuple[GDNStateSnapshot, ...]) -> None:
        self.prefix_cache.publish_many(seq, snapshots)

    def release(self, seq: Sequence) -> None:
        seq.pending_state_snapshot = None
        self.prefix_cache.unmark_reused(seq)
        self.block_manager.deallocate(seq)
        self.state_manager.deallocate(seq)
        seq.committed_tokens = 0
        seq.clear_runtime_features()

    def clear_prefixes(self) -> None:
        while self.prefix_cache.evict():
            pass
