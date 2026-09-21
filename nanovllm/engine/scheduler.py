from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
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
        return sum(seq.num_scheduled_tokens for seq in self.prefill_seqs)

    @property
    def is_empty(self) -> bool:
        return not self.decode_seqs and not self.prefill_seqs


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            enable_prefix_cache=config.enable_prefix_cache,
        )
        self.state_manager = StateSlotManager(config.max_num_state_slots)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        """Build one decode-first scheduling step.

        Decode consumes one token of budget per running sequence. Any remaining
        token budget and sequence slots are used for chunked prefill. Decode and
        prefill are returned separately so the engine can execute decode first
        (CUDA graph) and then run dynamic-length prefill without forcing both
        phases into one model invocation.
        """
        output = SchedulerOutput()
        num_batched_tokens = 0
        preempted_this_step = False

        # 1. Decode first to avoid head-of-line blocking from long prefills.
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
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                output.decode_seqs.append(seq)
                num_batched_tokens += 1

        # Keep scheduled decode sequences in the running queue while the model
        # executes. Finished sequences are removed by postprocess().
        self.running.extendleft(reversed(output.decode_seqs))

        # Under KV pressure, do not immediately re-prefill a sequence that was
        # just preempted in the same step; this avoids deallocate/reallocate
        # thrashing and preserves decode priority.
        if preempted_this_step:
            return output

        # 2. Use the remaining budget for chunked prefill.
        remaining_seq_slots = self.max_num_seqs - len(output.decode_seqs)
        while self.waiting and len(output.prefill_seqs) < remaining_seq_slots:
            remaining_tokens = self.max_num_batched_tokens - num_batched_tokens
            if remaining_tokens == 0:
                break

            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = (
                    seq.num_tokens - num_cached_blocks * self.block_size
                )
                self.block_manager.allocate(seq, num_cached_blocks)
                self.state_manager.allocate(seq)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if num_tokens <= 0:
                raise RuntimeError(
                    f"sequence {seq.seq_id} has no remaining prefill tokens"
                )

            seq.num_scheduled_tokens = min(num_tokens, remaining_tokens)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            output.prefill_seqs.append(seq)

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                # A partially scheduled request stays at the front so its next
                # chunk is considered first on the next scheduling step.
                break

        if output.is_empty:
            raise RuntimeError("scheduler could not make progress")
        return output

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.state_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.state_manager.deallocate(seq)
                self.running.remove(seq)
