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

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        num_batched_tokens = 0
        preempted_this_step = False

        # Decode first. Qwen3.5-MoE decode mutates both KV and recurrent state,
        # so scheduled requests remain in the running queue until postprocess.
        while (
            self.running
            and len(output.decode_seqs) < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.running.popleft()
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
            if not seq.block_table:
                if (
                    not self.block_manager.can_allocate(seq)
                    or not self.state_manager.can_allocate(seq)
                ):
                    break
                self.block_manager.allocate(seq)
                self.state_manager.allocate(seq)

            num_tokens = (
                seq.num_tokens - seq.num_cached_tokens
            )
            if num_tokens <= 0:
                raise RuntimeError(
                    f"sequence {seq.seq_id} has no remaining prefill tokens"
                )

            seq.num_scheduled_tokens = min(
                num_tokens,
                remaining_tokens,
            )
            num_batched_tokens += seq.num_scheduled_tokens
            output.prefill_seqs.append(seq)

            if (
                seq.num_cached_tokens
                + seq.num_scheduled_tokens
                == seq.num_tokens
            ):
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                break

        if output.is_empty:
            raise RuntimeError("scheduler could not make progress")
        return output

    def preempt(self, seq: Sequence):
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
            seq.num_cached_tokens += seq.num_scheduled_tokens
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
