from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.prefix_cache import JointPrefixCache, JointPrefixEntry
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import StateSlotManager


@dataclass(slots=True)
class SchedulerOutput:
    decode_seqs: list[Sequence] = field(default_factory=list)
    prefill_seqs: list[Sequence] = field(default_factory=list)

    @property
    def decode_tokens(self) -> int:
        return len(self.decode_seqs)

    @property
    def prefill_tokens(self) -> int:
        return sum(
            seq.num_scheduled_tokens
            for seq in self.prefill_seqs
        )

    @property
    def is_empty(self) -> bool:
        return not self.decode_seqs and not self.prefill_seqs


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_ids = set(config.eos_token_ids)
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.state_manager = StateSlotManager(
            config.max_num_state_slots
        )
        self.enable_prefix_cache = getattr(
            config,
            "enable_prefix_cache",
            True,
        )
        self.prefix_cache = JointPrefixCache(
            getattr(config, "max_prefix_cache_entries", 16)
            if self.enable_prefix_cache
            else 0
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _validate_committed_prefix(self, seq: Sequence) -> None:
        if seq.num_cached_tokens != seq.num_state_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} KV/state prefix mismatch: "
                f"kv={seq.num_cached_tokens}, "
                f"state={seq.num_state_tokens}"
            )
        if seq.block_table:
            self.state_manager.validate(seq)
        elif seq.state_slot >= 0:
            raise RuntimeError(
                f"sequence {seq.seq_id} owns state slot "
                "without KV allocation"
            )
        if (
            seq.pending_state_snapshot is not None
            and seq.num_state_tokens == 0
        ):
            raise RuntimeError(
                f"sequence {seq.seq_id} has a pending snapshot "
                "without a cached prefix"
            )

    def _evict_one_cached_prefix(self) -> bool:
        entry = self.prefix_cache.pop_lru()
        if entry is None:
            return False
        self.block_manager.release_blocks(entry.block_ids)
        return True

    def _try_restore_cached_prefix(self, seq: Sequence) -> bool:
        if not self.enable_prefix_cache:
            return False
        if (
            seq.num_cached_tokens != 0
            or seq.num_state_tokens != 0
            or seq.block_table
            or seq.state_slot >= 0
        ):
            return False

        # Keep at least one token uncached. The runtime does not cache prompt
        # logits, so a full-token hit would have nothing to execute/sample.
        entry = self.prefix_cache.longest_match(
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

        seq.num_cached_tokens = entry.num_tokens
        seq.num_state_tokens = entry.num_tokens
        seq.pending_state_snapshot = entry.state_snapshot
        return True

    def should_snapshot_prefix_after_step(
        self,
        seq: Sequence,
    ) -> bool:
        if not self.enable_prefix_cache:
            return False
        if seq.num_scheduled_tokens <= 0 or seq.state_slot < 0:
            return False

        target_tokens = (
            seq.num_cached_tokens + seq.num_scheduled_tokens
        )
        if target_tokens <= 0:
            return False
        if target_tokens > seq.num_prompt_tokens:
            return False
        if target_tokens % self.block_size != 0:
            return False

        key = tuple(seq.token_ids[:target_tokens])
        return not self.prefix_cache.contains(key)

    def _publish_committed_prefix(
        self,
        seq: Sequence,
        state_snapshot,
    ) -> None:
        prefix_tokens = seq.num_cached_tokens
        if prefix_tokens != seq.num_state_tokens:
            raise RuntimeError(
                "cannot cache a divergent KV/GDN prefix"
            )
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
        snapshot_tokens = getattr(
            state_snapshot,
            "num_tokens",
            prefix_tokens,
        )
        if snapshot_tokens != prefix_tokens:
            raise RuntimeError(
                "GDN snapshot boundary does not match KV prefix boundary"
            )

        key = tuple(seq.token_ids[:prefix_tokens])
        if self.prefix_cache.contains(key):
            return

        num_blocks = prefix_tokens // self.block_size
        block_ids = tuple(seq.block_table[:num_blocks])
        if len(block_ids) != num_blocks:
            raise RuntimeError(
                "KV block table is shorter than cached prefix"
            )

        # The cache owns one extra reference independently of the request.
        self.block_manager.retain_blocks(block_ids)
        entry = JointPrefixEntry(
            token_ids=key,
            block_ids=block_ids,
            num_tokens=prefix_tokens,
            state_snapshot=state_snapshot,
        )
        try:
            released = self.prefix_cache.put(entry)
        except Exception:
            self.block_manager.release_blocks(block_ids)
            raise

        for old_entry in released:
            self.block_manager.release_blocks(
                old_entry.block_ids
            )

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        num_batched_tokens = 0
        preempted_this_step = False

        # Decode consumes the shared token budget first. Selected requests are
        # rotated to the back afterwards so a budget smaller than the number
        # of active requests does not starve the same tail forever.
        while (
            self.running
            and len(output.decode_seqs) < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.running.popleft()
            self._validate_committed_prefix(seq)

            while not self.block_manager.can_append(seq):
                # Reclaim idle cached prefixes before retracting live work.
                if self._evict_one_cached_prefix():
                    continue
                if self.running:
                    self.preempt(self.running.pop())
                    preempted_this_step = True
                else:
                    self.preempt(seq)
                    preempted_this_step = True
                    seq = None
                    break

            if seq is None:
                break

            seq.num_scheduled_tokens = 1
            self.block_manager.may_append(seq)
            output.decode_seqs.append(seq)
            num_batched_tokens += 1

        self.running.extend(output.decode_seqs)
        if preempted_this_step:
            return output

        remaining_seq_slots = (
            self.max_num_seqs - len(output.decode_seqs)
        )
        while (
            self.waiting
            and len(output.prefill_seqs) < remaining_seq_slots
        ):
            remaining_tokens = (
                self.max_num_batched_tokens
                - num_batched_tokens
            )
            if remaining_tokens == 0:
                break

            seq = self.waiting[0]
            self._validate_committed_prefix(seq)
            if seq.num_cached_tokens == 0:
                self._try_restore_cached_prefix(seq)
                self._validate_committed_prefix(seq)

            num_remaining = (
                seq.num_tokens - seq.num_cached_tokens
            )
            if num_remaining <= 0:
                raise RuntimeError(
                    f"sequence {seq.seq_id} has no remaining prefill tokens"
                )

            requested_tokens = min(
                num_remaining,
                remaining_tokens,
            )
            scheduled_tokens = (
                self.block_manager.max_schedulable_tokens(
                    seq,
                    requested_tokens,
                )
            )
            while (
                scheduled_tokens == 0
                and self._evict_one_cached_prefix()
            ):
                scheduled_tokens = (
                    self.block_manager.max_schedulable_tokens(
                        seq,
                        requested_tokens,
                    )
                )
            if scheduled_tokens == 0:
                break

            if (
                seq.state_slot < 0
                and not self.state_manager.can_allocate(seq)
            ):
                break

            target_tokens = (
                seq.num_cached_tokens + scheduled_tokens
            )
            if not self.block_manager.can_ensure_capacity(
                seq,
                target_tokens,
            ):
                raise RuntimeError(
                    "KV capacity calculation diverged from reservation"
                )

            if seq.state_slot < 0:
                self.state_manager.allocate(seq)
            self.block_manager.ensure_capacity(
                seq,
                target_tokens,
            )

            seq.num_scheduled_tokens = scheduled_tokens
            num_batched_tokens += scheduled_tokens
            output.prefill_seqs.append(seq)

            if target_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                # One request cannot contribute two dependent prefill chunks to
                # the same forward pass, so stop after a partial chunk.
                break

        if output.is_empty:
            raise RuntimeError("scheduler could not make progress")
        return output

    def recover_failed_step(self, seqs: list[Sequence]) -> None:
        """Discard possibly mutated physical state and replay from history."""
        for seq in reversed(seqs):
            self._validate_committed_prefix(seq)
            if seq in self.running:
                self.running.remove(seq)
            if seq in self.waiting:
                self.waiting.remove(seq)
            seq.num_scheduled_tokens = 0
            seq.pending_state_snapshot = None
            seq.status = SequenceStatus.WAITING
            self.block_manager.deallocate(seq)
            self.state_manager.deallocate(seq)
            self.waiting.appendleft(seq)

    def preempt(self, seq: Sequence):
        """Retract KV and GDN together; later admission restores/replays both."""
        self._validate_committed_prefix(seq)
        seq.status = SequenceStatus.WAITING
        seq.pending_state_snapshot = None
        self.block_manager.deallocate(seq)
        self.state_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int | None],
        is_prefill: bool,
        prefix_snapshots: dict[int, object] | None = None,
    ):
        prefix_snapshots = prefix_snapshots or {}
        for seq, token_id in zip(seqs, token_ids):
            self._validate_committed_prefix(seq)
            committed_tokens = seq.num_scheduled_tokens
            seq.num_cached_tokens += committed_tokens
            seq.num_state_tokens += committed_tokens
            seq.num_scheduled_tokens = 0

            snapshot = prefix_snapshots.get(seq.seq_id)
            if snapshot is not None:
                if not is_prefill:
                    raise RuntimeError(
                        "joint prefix snapshots are valid only for prefill"
                    )
                self._publish_committed_prefix(
                    seq,
                    snapshot,
                )

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                if token_id is not None:
                    raise RuntimeError(
                        "partial prefill produced an unexpected sample"
                    )
                continue

            if token_id is None:
                raise RuntimeError(
                    "completed model step did not produce a token"
                )
            seq.append_token(token_id)

            hit_eos = (
                not seq.ignore_eos
                and token_id in self.eos_token_ids
            )
            if (
                hit_eos
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.state_manager.deallocate(seq)
                self.running.remove(seq)
