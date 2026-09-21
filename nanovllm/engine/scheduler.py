from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
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

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        num_batched_tokens = 0
        preempted_this_step = False

        # Decode first. Decode mutates paged KV and GDN state together, so
        # requests stay logically running until postprocess commits progress.
        while (
            self.running
            and len(output.decode_seqs) < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.running.popleft()
            self._validate_committed_prefix(seq)
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                    preempted_this_step = True
                else:
                    self.preempt(seq)
                    preempted_this_step = True
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                output.decode_seqs.append(seq)
                num_batched_tokens += 1

        self.running.extendleft(reversed(output.decode_seqs))
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

            # Admission is intentionally narrow: one recurrent slot plus only
            # the KV blocks needed by this scheduled range. A long prompt no
            # longer reserves its unscheduled tail.
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
                # A partial prefill keeps its KV prefix and recurrent slot.
                # Partial scheduling means the current token/KV budget is
                # exhausted, so no later waiting request can make useful
                # progress in this step.
                break

        if output.is_empty:
            raise RuntimeError("scheduler could not make progress")
        return output

    def preempt(self, seq: Sequence):
        self._validate_committed_prefix(seq)
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.state_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int | None],
        is_prefill: bool,
    ):
        for seq, token_id in zip(seqs, token_ids):
            self._validate_committed_prefix(seq)
            committed_tokens = seq.num_scheduled_tokens
            seq.num_cached_tokens += committed_tokens
            seq.num_state_tokens += committed_tokens
            seq.num_scheduled_tokens = 0

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
