from collections import deque

from nanovllm.config import Config
from nanovllm.engine.hybrid_cache_manager import HybridCacheManager
from nanovllm.engine.schedule import ScheduledChunk, SchedulerOutput
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot


class Scheduler:
    def __init__(self, config: Config, num_kv_cache_blocks: int, snapshot_nbytes: int = 0) -> None:
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_mixed_prefill_tokens = config.max_mixed_prefill_tokens
        self.eos_token_ids = set(config.eos_token_ids)
        self.cache_manager = HybridCacheManager(config, num_kv_cache_blocks, snapshot_nbytes)
        self.waiting: deque[Sequence] = deque()
        self.prefilling: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.prefilling and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def _schedule_prefill_pass(
        self,
        prefill_chunks: list[ScheduledChunk],
        decode_count: int,
    ) -> None:
        if (not self.prefilling and not self.waiting) or decode_count >= min(
            self.max_num_seqs, self.max_num_batched_tokens
        ):
            return
        num_batched_tokens = decode_count
        token_limit = self.max_num_batched_tokens
        if decode_count and self.max_mixed_prefill_tokens is not None:
            token_limit = min(token_limit, decode_count + self.max_mixed_prefill_tokens)
        if num_batched_tokens >= token_limit:
            return
        active = (*self.running, *self.prefilling)
        occupied_blocks, reserved_blocks = self.cache_manager.reservation_usage(active)
        active_count = len(self.running) + len(self.prefilling)
        planned_prefixes: set[bytes] = set()

        # Resume admitted prefills round-robin; admit waiting requests in FIFO order.
        for admitted, queue in ((True, self.prefilling), (False, self.waiting)):
            for _ in range(len(queue)):
                if (
                    num_batched_tokens >= token_limit
                    or decode_count + len(prefill_chunks) >= self.max_num_seqs
                ):
                    return

                seq = queue.popleft()
                if not admitted:
                    entry = self.cache_manager.find_prefix(seq)
                    blocks = self.cache_manager.admission_blocks(seq, entry, occupied_blocks)
                    if (
                        active_count >= self.max_num_seqs
                        or reserved_blocks + blocks > self.cache_manager.total_blocks
                    ):
                        queue.appendleft(seq)
                        break
                    self.cache_manager.admit(seq, entry)
                    active_count += 1
                    occupied_blocks.update(entry.block_ids if entry is not None else ())
                    reserved_blocks += blocks

                requested = min(
                    len(seq) - seq.committed_tokens,
                    token_limit - num_batched_tokens,
                )
                start = seq.committed_tokens
                # Checkpoint alignment may shrink the requested chunk. Do not
                # evict cached KV pages for tokens we will not actually run.
                proposed_prefixes = set(planned_prefixes)
                proposed_end, positions, checkpoint_interval = (
                    self.cache_manager.plan_prefill_checkpoints(
                        seq, start, start + requested, proposed_prefixes
                    )
                )
                scheduled = self.cache_manager.fit_prefill(seq, proposed_end - start)
                if scheduled < proposed_end - start:
                    # Capacity is still insufficient; replan on the true range.
                    # Discard candidate bookkeeping from the unexecuted tail.
                    end, positions, checkpoint_interval = (
                        self.cache_manager.plan_prefill_checkpoints(
                            seq, start, start + scheduled, planned_prefixes
                        )
                    )
                else:
                    end = proposed_end
                    planned_prefixes.update(proposed_prefixes)
                self.cache_manager.reserve_prefill(seq, end)
                prefill_chunks.append(
                    ScheduledChunk(
                        seq,
                        start,
                        end,
                        checkpoint_positions=positions,
                        checkpoint_interval=checkpoint_interval,
                    )
                )
                num_batched_tokens += end - start
                if end == len(seq):
                    self.running.append(seq)
                else:
                    self.prefilling.append(seq)

    def _release_runtime_state(self, seq: Sequence) -> None:
        self.cache_manager.release(seq)

    def clear(self) -> None:
        for seq in (*self.waiting, *self.prefilling, *self.running):
            self._release_runtime_state(seq)
        self.waiting.clear()
        self.prefilling.clear()
        self.running.clear()
        self.cache_manager.clear_prefixes()

    def schedule(self) -> SchedulerOutput:
        decode_chunks: list[ScheduledChunk] = []
        prefill_chunks: list[ScheduledChunk] = []

        # Accepted requests already reserve their maximum KV lifetime.
        for _ in range(min(len(self.running), self.max_num_seqs, self.max_num_batched_tokens)):
            seq = self.running.popleft()
            if seq.committed_tokens != len(seq) - 1:
                raise RuntimeError(f"sequence {seq.seq_id} decode prefix mismatch")
            self.cache_manager.reserve_decode(seq)
            decode_chunks.append(ScheduledChunk(seq, seq.committed_tokens, len(seq)))
            self.running.append(seq)

        self._schedule_prefill_pass(prefill_chunks, len(decode_chunks))
        if not decode_chunks and not prefill_chunks and (self.waiting or self.prefilling):
            raise RuntimeError("scheduler could not make progress")
        return SchedulerOutput(
            decode_chunks=tuple(decode_chunks),
            prefill_chunks=tuple(prefill_chunks),
        )

    def postprocess(
        self,
        chunks: tuple[ScheduledChunk, ...],
        token_ids: list[int | None],
        is_prefill: bool,
        prefix_snapshots: dict[int, tuple[GDNStateSnapshot, ...]] | None = None,
    ) -> None:
        prefix_snapshots = prefix_snapshots or {}
        if len(chunks) != len(token_ids):
            raise RuntimeError("model result count does not match scheduled sequence count")

        for chunk, token_id in zip(chunks, token_ids):
            seq = chunk.seq
            partial_prefill = is_prefill and chunk.end < len(seq)
            if partial_prefill and token_id is not None:
                raise RuntimeError("partial prefill produced an unexpected sample")
            if not partial_prefill and token_id is None:
                raise RuntimeError("completed model step did not produce a token")

            self.cache_manager.publish_prefixes(seq, prefix_snapshots.get(seq.seq_id, ()))

            seq.committed_tokens = chunk.end
            if seq.image_state is not None and chunk.end >= seq.image_state.image_end:
                seq.clear_runtime_features()
            if partial_prefill:
                continue

            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id in self.eos_token_ids) or (
                seq.num_completion_tokens == seq.max_tokens
            ):
                seq.finished = True
                self._release_runtime_state(seq)
                self.running.remove(seq)
