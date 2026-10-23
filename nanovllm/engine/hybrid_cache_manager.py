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
        """Select at most one checkpoint without depending on cache ancestry."""
        if not self.prefix_cache.enabled:
            return end, (), 0

        step = lcm(GDN_CHECKPOINT_INTERVAL, self.block_manager.block_size)
        candidates, _ = self.prefix_cache.candidates(seq, start, end)
        candidates = [
            pair
            for pair in candidates
            if pair[1] not in planned_prefixes and pair[0] < seq.num_prompt_tokens
        ]
        # Only trim inexpensive tails; never force an additional tiny forward.
        if end < len(seq):
            aligned_end = end - end % step
            if (
                start < aligned_end < end
                and end - aligned_end <= 8
                and any(position == aligned_end for position, _ in candidates)
            ):
                end = aligned_end

        candidates = [pair for pair in candidates if pair[0] <= end]
        if start % step:
            # Internal FlashInfer checkpoints use a chunk-relative grid.
            # Only an aligned final live state is valid for off-grid chunks.
            candidates = [pair for pair in candidates if pair[0] == end]
        selected = self.prefix_cache.select_checkpoints(candidates)
        planned_prefixes.update(digest for _, digest in selected)
        positions = tuple(position for position, _ in selected)
        interval = GDN_CHECKPOINT_INTERVAL if any(p < end for p in positions) else 0
        return end, positions, interval

    def prepare_prefill(
        self, seq: Sequence, start: int, requested: int, planned_prefixes: set[bytes]
    ) -> tuple[int, tuple[int, ...], int]:
        """Commit a runnable range; replan only after an actual KV capacity trim."""
        proposal = set(planned_prefixes)
        end, positions, interval = self.plan_prefill_checkpoints(
            seq, start, start + requested, proposal
        )
        fitted = self.fit_prefill(seq, end - start)
        if fitted < end - start:
            end, positions, interval = self.plan_prefill_checkpoints(
                seq, start, start + fitted, planned_prefixes
            )
        else:
            planned_prefixes.update(proposal)
        self.reserve_prefill(seq, end)
        return end, positions, interval

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
