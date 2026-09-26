from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.prefix_runtime import PrefixRuntime
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import (
    GDNStateSnapshot,
    StateSlotManager,
)


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

    def __init__(
        self,
        config: Config,
        num_kvcache_blocks: int,
    ):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_ids = set(config.eos_token_ids)
        self.block_manager = BlockManager(
            num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.state_manager = StateSlotManager(
            config.max_num_seqs
        )
        self.prefix_runtime = PrefixRuntime(
            max_entries=config.max_prefix_cache_entries,
            block_size=config.kvcache_block_size,
            block_manager=self.block_manager,
            state_manager=self.state_manager,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _validate_committed_prefix(self, seq: Sequence) -> None:
        if seq.block_table:
            self.state_manager.validate(seq)
        elif seq.state_slot >= 0:
            raise RuntimeError(
                f"sequence {seq.seq_id} owns state slot "
                "without KV allocation"
            )
        elif seq.committed_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} has committed history "
                "without hybrid resources"
            )
        if (
            seq.pending_state_snapshot is not None
            and seq.committed_tokens == 0
        ):
            raise RuntimeError(
                f"sequence {seq.seq_id} has a pending snapshot "
                "without a cached prefix"
            )

    def _reserve_prefill_resources(
        self,
        seq: Sequence,
        target_tokens: int,
    ) -> None:
        """Reserve KV growth and a GDN slot as one scheduler transaction."""
        old_num_blocks = len(seq.block_table)
        allocated_state = seq.state_slot < 0
        try:
            if allocated_state:
                self.state_manager.allocate(seq)
            self.block_manager.ensure_capacity(
                seq,
                target_tokens,
            )
        except Exception:
            self.block_manager.truncate_blocks(
                seq,
                old_num_blocks,
            )
            if allocated_state and seq.state_slot >= 0:
                self.state_manager.deallocate(seq)
            raise

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        num_batched_tokens = 0
        preempted_this_step = False

        # Decode consumes the shared token budget first. Rotating selected
        # requests prevents starvation when the budget is smaller than the
        # number of active requests.
        while (
            self.running
            and len(output.decode_seqs) < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.running.popleft()
            self._validate_committed_prefix(seq)

            while (
                self.block_manager.max_schedulable_tokens(seq, 1) == 0
            ):
                if self.prefix_runtime.evict_one():
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
            self.block_manager.ensure_capacity(seq, len(seq))
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
            if seq.committed_tokens == 0:
                self.prefix_runtime.try_restore(seq)
                self._validate_committed_prefix(seq)

            num_remaining = (
                seq.num_tokens - seq.committed_tokens
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
                and self.prefix_runtime.evict_one()
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
                seq.committed_tokens + scheduled_tokens
            )
            self._reserve_prefill_resources(
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
                # Dependent chunks of one request cannot share one forward.
                break

        if output.is_empty:
            raise RuntimeError("scheduler could not make progress")
        return output

    def _reset_to_waiting(self, seq: Sequence) -> None:
        """Release hybrid history and replay the request from a valid prefix."""
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
        seq.committed_tokens = 0
        self.waiting.appendleft(seq)

    def recover_failed_step(self, seqs: list[Sequence]) -> None:
        """Discard possibly mutated physical state and replay from history."""
        for seq in reversed(seqs):
            self._reset_to_waiting(seq)

    def preempt(self, seq: Sequence):
        """Retract KV and GDN together; later admission restores/replays both."""
        self._reset_to_waiting(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int | None],
        is_prefill: bool,
        prefix_snapshots: dict[int, GDNStateSnapshot] | None = None,
    ):
        prefix_snapshots = prefix_snapshots or {}

        # Validate the whole snapshot batch before committing logical progress.
        for seq in seqs:
            snapshot = prefix_snapshots.get(seq.seq_id)
            if snapshot is None:
                continue
            if not is_prefill:
                raise RuntimeError(
                    "joint prefix snapshots are valid only for prefill"
                )
            if not isinstance(snapshot, GDNStateSnapshot):
                raise RuntimeError("invalid GDN state snapshot")
            target_tokens = (
                seq.committed_tokens + seq.num_scheduled_tokens
            )
            if snapshot.num_tokens != target_tokens:
                raise RuntimeError(
                    "GDN snapshot boundary does not match scheduled prefix"
                )

        for seq, token_id in zip(seqs, token_ids):
            self._validate_committed_prefix(seq)
            seq.committed_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            snapshot = prefix_snapshots.get(seq.seq_id)
            if snapshot is not None:
                self.prefix_runtime.publish(seq, snapshot)

            if is_prefill and seq.committed_tokens < seq.num_tokens:
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
                seq.committed_tokens = 0
                self.running.remove(seq)
