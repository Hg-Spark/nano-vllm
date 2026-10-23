"""Logical ownership of paged KV, recurrent state slots, and joint prefix cache."""

from collections.abc import Iterable

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.prefix_cache import PrefixCache, PrefixEntry
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
            max_active_requests=config.max_num_seqs,
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
                # The published snapshot is immutable. Rolling staging owns
                # separate CPU tensors and may never overwrite cache entries.
            self.state_manager.allocate(seq)
        except Exception:
            # Never retain shared KV refs if GDN slot admission fails.
            self.prefix_cache.unmark_reused(seq)
            self.block_manager.deallocate(seq)
            self.state_manager.deallocate(seq)
            seq.committed_tokens = 0
            seq.pending_state_snapshot = None
            seq.prompt_snapshot = None
            seq.completion_snapshot = None
            raise

    def fit_prefill(self, seq: Sequence, requested_tokens: int) -> int:
        """Evict expendable checkpoints until a reserved prefill can progress."""
        scheduled = self.block_manager.max_schedulable_tokens(seq, requested_tokens)
        while scheduled == 0 and self.prefix_cache.evict(self.block_manager.block_refcounts):
            scheduled = self.block_manager.max_schedulable_tokens(seq, requested_tokens)
        if scheduled == 0:
            raise RuntimeError("admitted prefill cannot acquire reserved KV blocks")
        return scheduled

    def prepare_prefill(self, seq: Sequence, start: int, requested: int) -> int:
        """Schedule the available token budget, only evicting if progress is blocked."""
        fitted = self.block_manager.max_schedulable_tokens(seq, requested)
        if fitted == 0:
            fitted = self.fit_prefill(seq, requested)
        end = start + fitted
        # Only one prefill checkpoint is needed: stop once just before the
        # prompt's final token, at the last complete KV page. A chunk crossing
        # this boundary must end here so that the GDN final state is exact.
        if self.prefix_cache._can_capture() and seq.image_state is None:
            boundary = (
                (seq.num_prompt_tokens - 1) // self.block_manager.block_size
            ) * self.block_manager.block_size
            if start < boundary < end:
                end = boundary
        self.reserve_prefill(seq, end)
        return end

    def reserve_prefill(self, seq: Sequence, end: int) -> None:
        self.block_manager.ensure_capacity(seq, end)

    def reserve_decode(self, seq: Sequence) -> None:
        while self.block_manager.max_schedulable_tokens(seq, 1) == 0:
            if not self.prefix_cache.evict(self.block_manager.block_refcounts):
                raise RuntimeError("admitted decode cannot acquire reserved KV blocks")
        self.block_manager.ensure_capacity(seq, len(seq))

    def publish_batch(self, candidates: list[tuple[Sequence, GDNStateSnapshot]]) -> None:
        # Completed requests no longer need cache-entry reuse protection:
        # the live request still owns all its KV pages until postprocess().
        for seq in {seq.seq_id: seq for seq, _ in candidates}.values():
            self.prefix_cache.unmark_reused(seq)
            # The snapshot tensors change ownership from request staging to
            # the global cache, or become disposable when capacity is full.
            self.prefix_cache.unstage(seq)
        self.prefix_cache.publish_batch(candidates)

    def release(self, seq: Sequence) -> None:
        seq.pending_state_snapshot = None
        seq.prompt_snapshot = None
        seq.completion_snapshot = None
        self.prefix_cache.unstage(seq)
        self.prefix_cache.unmark_reused(seq)
        self.block_manager.deallocate(seq)
        self.state_manager.deallocate(seq)
        seq.committed_tokens = 0
        seq.clear_runtime_features()

    def clear_prefixes(self) -> None:
        while self.prefix_cache.evict():
            pass
