from collections import deque
from dataclasses import replace

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.hybrid_resources import HybridResources
from nanovllm.engine.schedule import ScheduledChunk, SchedulerOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import GDNStateSnapshot, StateSlotManager


class Scheduler:
    def __init__(self, config: Config, num_kv_cache_blocks: int) -> None:
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_ids = set(config.eos_token_ids)
        self.block_manager = BlockManager(
            num_kv_cache_blocks,
            config.kv_cache_block_size,
            max_prefix_entries=config.max_prefix_cache_entries,
        )
        self.state_manager = StateSlotManager(config.max_num_seqs)
        self.resources = HybridResources(self.block_manager, self.state_manager)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def _try_restore_prefix(self, seq: Sequence) -> bool:
        if seq.committed_tokens != 0 or seq.block_table or seq.state_slot >= 0:
            return False
        entry = self.block_manager.find_prefix(seq, max_tokens=max(0, seq.num_tokens - 1))
        if entry is None or not self.state_manager.can_allocate(seq):
            return False
        self.resources.restore_prefix(seq, entry.block_ids, entry.num_tokens, entry.state_snapshot)
        return True

    def _decode_headroom_blocks(self, scheduled_decode: list[ScheduledChunk]) -> int:
        scheduled_ids = {chunk.seq.seq_id for chunk in scheduled_decode}
        reserved = 0
        for seq in self.running:
            if seq.seq_id in scheduled_ids:
                if seq.num_completion_tokens + 1 >= seq.max_tokens:
                    continue
                target_tokens = len(seq) + 1
            else:
                target_tokens = len(seq)
            reserved += self.block_manager.additional_blocks_needed(seq, target_tokens)
        return reserved

    def _find_waiting_resource_victim(self) -> Sequence | None:
        for candidate in reversed(self.waiting):
            if candidate.block_table or candidate.state_slot >= 0:
                return candidate
        return None

    def _schedule_prefill_pass(
        self,
        prefill_chunks: list[ScheduledChunk],
        decode_count: int,
        num_batched_tokens: int,
        *,
        reserved_free_blocks: int = 0,
    ) -> int:
        """Scan each waiter once within the token and resource budgets."""
        max_prefill_seqs = self.max_num_seqs - decode_count
        for _ in range(len(self.waiting)):
            if not self.waiting or len(prefill_chunks) >= max_prefill_seqs:
                break
            remaining_tokens = self.max_num_batched_tokens - num_batched_tokens
            if remaining_tokens == 0:
                break

            seq = self.waiting[0]
            self.resources.validate_request(seq)
            self._try_restore_prefix(seq)

            num_remaining = seq.num_tokens - seq.committed_tokens
            requested_tokens = min(num_remaining, remaining_tokens)
            scheduled_tokens = self.block_manager.max_schedulable_tokens(
                seq, requested_tokens, reserved_free_blocks
            )
            while scheduled_tokens == 0 and self.block_manager.evict_prefix():
                scheduled_tokens = self.block_manager.max_schedulable_tokens(
                    seq, requested_tokens, reserved_free_blocks
                )
            if scheduled_tokens == 0 or (
                seq.state_slot < 0 and not self.state_manager.can_allocate(seq)
            ):
                self.waiting.rotate(-1)
                continue

            target_tokens = seq.committed_tokens + scheduled_tokens
            # A complete KV page also needs the GDN state at the same boundary.
            # Stop before a non-aligned tail only when it yields a new cache entry.
            if self.block_manager.prefix_cache.enabled:
                block_size = self.block_manager.block_size
                full_block_end = target_tokens // block_size * block_size
                if (
                    seq.committed_tokens < full_block_end < target_tokens
                    and self.block_manager.should_cache_prefix(seq, full_block_end)
                ):
                    target_tokens = full_block_end
                    scheduled_tokens = target_tokens - seq.committed_tokens

            self.resources.reserve(seq, target_tokens)
            self.waiting.popleft()

            num_batched_tokens += scheduled_tokens
            prefill_chunks.append(ScheduledChunk(seq, seq.committed_tokens, target_tokens))

            if target_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.append(seq)

        return num_batched_tokens

    def _release_runtime_state(self, seq: Sequence) -> None:
        seq.pending_state_snapshot = None
        self.resources.release(seq)
        seq.committed_tokens = 0
        seq.clear_runtime_features()

    def schedule(self) -> SchedulerOutput:
        decode_chunks: list[ScheduledChunk] = []
        prefill_chunks: list[ScheduledChunk] = []
        num_batched_tokens = 0
        preempted_this_step = False

        while (
            self.running
            and len(decode_chunks) < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.running[0]
            self.resources.validate_request(seq)
            if seq.committed_tokens != len(seq) - 1:
                raise RuntimeError(f"sequence {seq.seq_id} decode prefix mismatch")
            self.running.popleft()

            while self.block_manager.max_schedulable_tokens(seq, 1) == 0:
                if self.block_manager.evict_prefix():
                    continue
                victim = self._find_waiting_resource_victim()
                if victim is not None:
                    self.preempt(victim)
                    preempted_this_step = True
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

            self.block_manager.ensure_capacity(seq, len(seq))
            decode_chunks.append(ScheduledChunk(seq, seq.committed_tokens, len(seq)))
            num_batched_tokens += 1

        self.running.extend(chunk.seq for chunk in decode_chunks)
        if preempted_this_step:
            return SchedulerOutput(decode_chunks=tuple(decode_chunks))

        decode_headroom_blocks = self._decode_headroom_blocks(decode_chunks)
        self._schedule_prefill_pass(
            prefill_chunks,
            len(decode_chunks),
            num_batched_tokens,
            reserved_free_blocks=decode_headroom_blocks,
        )

        while not decode_chunks and not prefill_chunks:
            victim = self._find_waiting_resource_victim()
            if victim is None or len(self.waiting) <= 1:
                raise RuntimeError("scheduler could not make progress")
            self.preempt(victim)
            self._schedule_prefill_pass(prefill_chunks, 0, num_batched_tokens)
        return SchedulerOutput(
            decode_chunks=tuple(decode_chunks),
            prefill_chunks=tuple(
                replace(
                    chunk,
                    capture_snapshot=self.block_manager.should_cache_prefix(chunk.seq, chunk.end),
                )
                for chunk in prefill_chunks
            ),
        )

    def _reset_to_waiting(self, seq: Sequence, *, front: bool) -> None:
        self.resources.validate_request(seq)
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        seq.status = SequenceStatus.WAITING
        self._release_runtime_state(seq)
        if front:
            self.waiting.appendleft(seq)
        else:
            self.waiting.append(seq)

    def preempt(self, seq: Sequence) -> None:
        self._reset_to_waiting(seq, front=False)

    def postprocess(
        self,
        chunks: tuple[ScheduledChunk, ...],
        token_ids: list[int | None],
        is_prefill: bool,
        prefix_snapshots: dict[int, GDNStateSnapshot] | None = None,
    ) -> None:
        prefix_snapshots = prefix_snapshots or {}
        if len(chunks) != len(token_ids):
            raise RuntimeError("model result count does not match scheduled sequence count")

        for chunk, token_id in zip(chunks, token_ids):
            seq = chunk.seq
            self.resources.validate_request(seq)
            partial_prefill = is_prefill and chunk.end < len(seq)
            if partial_prefill and token_id is not None:
                raise RuntimeError("partial prefill produced an unexpected sample")
            if not partial_prefill and token_id is None:
                raise RuntimeError("completed model step did not produce a token")

            snapshot = prefix_snapshots.get(seq.seq_id)
            if snapshot is not None:
                if not is_prefill or not isinstance(snapshot, GDNStateSnapshot):
                    raise RuntimeError("invalid GDN state snapshot")
                if snapshot.num_tokens != chunk.end:
                    raise RuntimeError("GDN snapshot boundary does not match scheduled prefix")
                self.block_manager.publish_prefix(seq, snapshot, chunk.end)

            seq.committed_tokens = chunk.end
            if partial_prefill:
                continue

            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id in self.eos_token_ids) or (
                seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self._release_runtime_state(seq)
                self.running.remove(seq)
