from collections import deque
from dataclasses import replace

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.hybrid_resources import HybridResources
from nanovllm.engine.prefix_runtime import PrefixRuntime
from nanovllm.engine.schedule import ScheduledChunk, SchedulerOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import (
    GDNStateSnapshot,
    StateSlotManager,
)

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
        self.resources = HybridResources(
            self.block_manager,
            self.state_manager,
        )
        self.prefix_runtime = PrefixRuntime(
            max_entries=config.max_prefix_cache_entries,
            block_size=config.kvcache_block_size,
            resources=self.resources,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        if self.waiting or self.running:
            return False
        self.validate_resource_accounting()
        return True

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

    def validate_resource_accounting(self) -> None:
        """Verify KV refs and GDN slots match request/cache ownership."""
        seqs = [*self.waiting, *self.running]
        seq_ids = [seq.seq_id for seq in seqs]
        if len(seq_ids) != len(set(seq_ids)):
            raise RuntimeError("sequence appears in multiple scheduler queues")

        expected_refs = [0] * len(self.block_manager.block_refcounts)
        for seq in seqs:
            self._validate_committed_prefix(seq)
            if len(seq.block_table) != len(set(seq.block_table)):
                raise RuntimeError(
                    f"sequence {seq.seq_id} contains duplicate KV blocks"
                )
            for block_id in seq.block_table:
                if not 0 <= block_id < len(expected_refs):
                    raise RuntimeError(f"invalid KV block {block_id}")
                expected_refs[block_id] += 1

        for entry in self.prefix_runtime.entries():
            for block_id in entry.block_ids:
                if not 0 <= block_id < len(expected_refs):
                    raise RuntimeError(f"invalid cached KV block {block_id}")
                expected_refs[block_id] += 1

        if expected_refs != self.block_manager.block_refcounts:
            raise RuntimeError("KV cache refcount accounting mismatch")

        free_blocks = list(self.block_manager.free_block_ids)
        if len(free_blocks) != len(set(free_blocks)):
            raise RuntimeError("duplicate KV block in free list")
        expected_free_blocks = {
            block_id
            for block_id, refcount in enumerate(expected_refs)
            if refcount == 0
        }
        if set(free_blocks) != expected_free_blocks:
            raise RuntimeError("KV cache free-list accounting mismatch")

        expected_owners = [None] * len(self.state_manager.slot_owners)
        for seq in seqs:
            if seq.state_slot < 0:
                continue
            if not 0 <= seq.state_slot < len(expected_owners):
                raise RuntimeError(f"invalid state slot {seq.state_slot}")
            if expected_owners[seq.state_slot] is not None:
                raise RuntimeError(
                    f"state slot {seq.state_slot} has multiple owners"
                )
            expected_owners[seq.state_slot] = seq.seq_id

        if expected_owners != self.state_manager.slot_owners:
            raise RuntimeError("GDN state-slot accounting mismatch")

        free_slots = list(self.state_manager.free_slot_ids)
        if len(free_slots) != len(set(free_slots)):
            raise RuntimeError("duplicate GDN state slot in free list")
        expected_free_slots = {
            slot_id
            for slot_id, owner in enumerate(expected_owners)
            if owner is None
        }
        if set(free_slots) != expected_free_slots:
            raise RuntimeError("GDN state free-list accounting mismatch")

    def _decode_headroom_blocks(
        self,
        scheduled_decode: list[ScheduledChunk],
    ) -> int:
        """Reserve exactly the KV growth needed by the next decode step."""
        scheduled_ids = {
            chunk.seq.seq_id for chunk in scheduled_decode
        }
        reserved = 0
        for seq in self.running:
            if seq.seq_id in scheduled_ids:
                if seq.num_completion_tokens + 1 >= seq.max_tokens:
                    continue
                target_tokens = len(seq) + 1
            else:
                target_tokens = len(seq)
            reserved += self.block_manager.additional_blocks_needed(
                seq,
                target_tokens,
            )
        return reserved

    def _find_waiting_resource_victim(
        self,
        exclude: Sequence | None = None,
    ) -> Sequence | None:
        """Find an idle partial-prefill request that can release resources."""
        for candidate in reversed(self.waiting):
            if candidate is exclude:
                continue
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
        max_new_seqs: int | None = None,
    ) -> int:
        """Scan each waiter once without consuming decode headroom."""
        max_prefill_seqs = self.max_num_seqs - decode_count
        initial_prefill_count = len(prefill_chunks)
        num_waiting_to_scan = len(self.waiting)
        for _ in range(num_waiting_to_scan):
            if (
                not self.waiting
                or len(prefill_chunks) >= max_prefill_seqs
            ):
                break
            if (
                max_new_seqs is not None
                and (
                    len(prefill_chunks) - initial_prefill_count
                    >= max_new_seqs
                )
            ):
                break

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
                    reserved_free_blocks,
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
                        reserved_free_blocks,
                    )
                )
            if scheduled_tokens == 0:
                self.waiting.rotate(-1)
                continue

            if (
                seq.state_slot < 0
                and not self.state_manager.can_allocate(seq)
            ):
                self.waiting.rotate(-1)
                continue

            target_tokens = (
                seq.committed_tokens + scheduled_tokens
            )
            self.resources.reserve(
                seq,
                target_tokens,
            )

            admitted = self.waiting.popleft()
            if admitted is not seq:
                raise RuntimeError("waiting queue changed during admission")

            num_batched_tokens += scheduled_tokens
            prefill_chunks.append(
                ScheduledChunk(
                    seq,
                    seq.committed_tokens,
                    target_tokens,
                )
            )

            if target_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.append(seq)

        return num_batched_tokens

    def schedule(self) -> SchedulerOutput:
        decode_chunks: list[ScheduledChunk] = []
        prefill_chunks: list[ScheduledChunk] = []
        num_batched_tokens = 0
        preempted_this_step = False

        # Decode consumes the shared token budget first. Rotating selected
        # requests prevents starvation when the budget is smaller than the
        # number of active requests.
        while (
            self.running
            and len(decode_chunks) < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.running[0]
            self._validate_committed_prefix(seq)
            if seq.committed_tokens != len(seq) - 1:
                raise RuntimeError(
                    f"sequence {seq.seq_id} decode prefix mismatch"
                )
            self.running.popleft()

            while (
                self.block_manager.max_schedulable_tokens(seq, 1) == 0
            ):
                if self.prefix_runtime.evict_one():
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
            decode_chunks.append(
                ScheduledChunk(seq, seq.committed_tokens, len(seq))
            )
            num_batched_tokens += 1

        self.running.extend(chunk.seq for chunk in decode_chunks)
        if preempted_this_step:
            return SchedulerOutput(decode_chunks=tuple(decode_chunks))

        decode_headroom_blocks = self._decode_headroom_blocks(
            decode_chunks
        )
        num_batched_tokens = self._schedule_prefill_pass(
            prefill_chunks,
            len(decode_chunks),
            num_batched_tokens,
            reserved_free_blocks=decode_headroom_blocks,
        )

        # If only waiting requests remain and every one is blocked by resources
        # held by another waiter, release one holder and retry once. Moving the
        # victim to the tail prevents it from immediately reclaiming the pages
        # that were released to break the circular wait.
        if not decode_chunks and not prefill_chunks:
            victim = self._find_waiting_resource_victim()
            if victim is not None and len(self.waiting) > 1:
                self.preempt(victim)
                num_batched_tokens = self._schedule_prefill_pass(
                    prefill_chunks,
                    len(decode_chunks),
                    num_batched_tokens,
                    max_new_seqs=1,
                )

        if not decode_chunks and not prefill_chunks:
            raise RuntimeError("scheduler could not make progress")
        return SchedulerOutput(
            decode_chunks=tuple(decode_chunks),
            prefill_chunks=tuple(
                replace(
                    chunk,
                    capture_snapshot=(
                        self.prefix_runtime.should_snapshot_after_step(
                            chunk.seq,
                            chunk.end,
                        )
                    ),
                )
                for chunk in prefill_chunks
            ),
        )

    def _reset_to_waiting(
        self,
        seq: Sequence,
        *,
        front: bool,
    ) -> None:
        """Release hybrid history and replay the request from a valid prefix."""
        self._validate_committed_prefix(seq)
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        seq.pending_state_snapshot = None
        seq.status = SequenceStatus.WAITING
        self.resources.release(seq)
        seq.committed_tokens = 0
        if front:
            self.waiting.appendleft(seq)
        else:
            self.waiting.append(seq)

    def recover_failed_step(
        self,
        chunks: tuple[ScheduledChunk, ...],
    ) -> None:
        """Discard possibly mutated physical state and replay from history."""
        for chunk in reversed(chunks):
            self._reset_to_waiting(chunk.seq, front=True)

    def preempt(self, seq: Sequence):
        """Release resources and replay the preempted request after its peers."""
        self._reset_to_waiting(seq, front=False)

    def postprocess(
        self,
        chunks: tuple[ScheduledChunk, ...],
        token_ids: list[int | None],
        is_prefill: bool,
        prefix_snapshots: dict[int, GDNStateSnapshot] | None = None,
    ):
        prefix_snapshots = prefix_snapshots or {}
        if len(chunks) != len(token_ids):
            raise RuntimeError(
                "model result count does not match scheduled sequence count"
            )

        # Validate the whole batch before changing any logical boundary.
        for chunk, token_id in zip(chunks, token_ids):
            seq = chunk.seq
            self._validate_committed_prefix(seq)
            if seq.committed_tokens != chunk.start:
                raise RuntimeError(
                    f"sequence {seq.seq_id} scheduled prefix changed"
                )
            if chunk.end > len(seq):
                raise RuntimeError(
                    f"sequence {seq.seq_id} scheduled past known tokens"
                )
            if not is_prefill and (
                chunk.num_tokens != 1 or chunk.end != len(seq)
            ):
                raise RuntimeError("decode must schedule the final token")
            partial_prefill = is_prefill and chunk.end < len(seq)
            if partial_prefill and token_id is not None:
                raise RuntimeError(
                    "partial prefill produced an unexpected sample"
                )
            if not partial_prefill and token_id is None:
                raise RuntimeError(
                    "completed model step did not produce a token"
                )
            snapshot = prefix_snapshots.get(seq.seq_id)
            if snapshot is None:
                continue
            if not is_prefill:
                raise RuntimeError(
                    "joint prefix snapshots are valid only for prefill"
                )
            if not isinstance(snapshot, GDNStateSnapshot):
                raise RuntimeError("invalid GDN state snapshot")
            if snapshot.num_tokens != chunk.end:
                raise RuntimeError(
                    "GDN snapshot boundary does not match scheduled prefix"
                )

        for chunk, token_id in zip(chunks, token_ids):
            seq = chunk.seq
            seq.committed_tokens = chunk.end

            snapshot = prefix_snapshots.get(seq.seq_id)
            if snapshot is not None:
                self.prefix_runtime.publish(seq, snapshot)

            if is_prefill and chunk.end < len(seq):
                continue

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
                self.resources.release(seq)
                seq.committed_tokens = 0
                self.running.remove(seq)
