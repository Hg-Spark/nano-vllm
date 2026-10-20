from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.prefix_cache import PrefixEntry
from nanovllm.engine.schedule import GDN_CHECKPOINT_INTERVAL, ScheduledChunk, SchedulerOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import GDNStateSnapshot, StateSlotManager


class Scheduler:
    def __init__(self, config: Config, num_kv_cache_blocks: int, snapshot_nbytes: int = 0) -> None:
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.prefix_cache_eager_snapshot = config.prefix_cache_eager_snapshot
        self.eos_token_ids = set(config.eos_token_ids)
        self.block_manager = BlockManager(
            num_kv_cache_blocks,
            config.kv_cache_block_size,
            max_prefix_entries=config.max_prefix_cache_entries,
            max_prefix_cache_snapshot_bytes=config.max_prefix_cache_snapshot_bytes,
            snapshot_nbytes=snapshot_nbytes,
            max_checkpoints_per_path=config.max_gdn_checkpoints,
        )
        self.state_manager = StateSlotManager(config.max_num_seqs)
        self.waiting: deque[Sequence] = deque()
        self.prefilling: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.prefilling and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def _try_restore_prefix(self, seq: Sequence, entry: PrefixEntry | None) -> None:
        if entry is None:
            return
        self.block_manager.attach_shared_prefix(seq, entry.block_ids, entry.num_tokens)
        seq.committed_tokens = entry.num_tokens
        seq.pending_state_snapshot = entry.state_snapshot

    def _request_blocks(self, seq: Sequence) -> int:
        return self.block_manager.required_blocks(seq.num_prompt_tokens + seq.max_tokens - 1)

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
        active = (*self.running, *self.prefilling)
        occupied_blocks = {block for seq in active for block in seq.block_table}
        reserved_blocks = len(occupied_blocks) + sum(
            self._request_blocks(seq) - len(seq.block_table) for seq in active
        )
        active_count = len(self.running) + len(self.prefilling)
        prefix_cache = self.block_manager.prefix_cache
        planned_prefixes: set[bytes] = set()

        # Resume admitted prefills round-robin; admit waiting requests in FIFO order.
        for admitted, queue in ((True, self.prefilling), (False, self.waiting)):
            for _ in range(len(queue)):
                if (
                    num_batched_tokens >= self.max_num_batched_tokens
                    or decode_count + len(prefill_chunks) >= self.max_num_seqs
                ):
                    return

                seq = queue.popleft()
                if not admitted:
                    entry = prefix_cache.find(seq, max_tokens=len(seq) - 1)
                    shared_blocks = entry.block_ids if entry is not None else ()
                    blocks = (
                        self._request_blocks(seq)
                        - len(shared_blocks)
                        + len(set(shared_blocks) - occupied_blocks)
                    )
                    if (
                        active_count >= self.max_num_seqs
                        or reserved_blocks + blocks > self.block_manager.num_blocks
                    ):
                        queue.appendleft(seq)
                        break
                    self._try_restore_prefix(seq, entry)
                    self.state_manager.allocate(seq)
                    active_count += 1
                    occupied_blocks.update(shared_blocks)
                    reserved_blocks += blocks

                requested = min(
                    len(seq) - seq.committed_tokens,
                    self.max_num_batched_tokens - num_batched_tokens,
                )
                scheduled = self.block_manager.max_schedulable_tokens(seq, requested)
                while scheduled < requested and prefix_cache.evict(
                    self.block_manager.block_refcounts
                ):
                    scheduled = self.block_manager.max_schedulable_tokens(seq, requested)
                if scheduled == 0:
                    raise RuntimeError("admitted prefill cannot acquire reserved KV blocks")

                start = seq.committed_tokens
                end = start + scheduled
                positions: tuple[int, ...] = ()
                interval = 0
                if prefix_cache.enabled:
                    interval = GDN_CHECKPOINT_INTERVAL
                    candidates, cached = prefix_cache.candidates(seq, start, end)
                    alignment_candidates = candidates
                    candidates = [pair for pair in candidates if pair[1] not in planned_prefixes]
                    if self.prefix_cache_eager_snapshot:
                        if start % interval:
                            # Final state is available at natural chunk ends even when
                            # intermediate states are not aligned to the 64-token grid.
                            if end % interval == 0:
                                candidates = [pair for pair in candidates if pair[0] == end]
                            else:
                                # Avoid tiny tail chunks while recovering a useful
                                # checkpoint even when the prompt ends unaligned.
                                eligible = [
                                    pair
                                    for pair in candidates
                                    if pair[0] - start >= interval and end - pair[0] >= interval
                                ]
                                if eligible:
                                    end = eligible[-1][0]
                                    scheduled = end - start
                                    candidates = eligible[-1:]
                                else:
                                    candidates = []
                        elif end < len(seq) and end % interval:
                            last = end - end % interval
                            if last > start and any(p == last for p, _ in alignment_candidates):
                                end = last
                                scheduled = end - start
                            candidates = [pair for pair in candidates if pair[0] <= end]
                        candidates = prefix_cache.select_checkpoints(candidates, cached)
                    else:
                        candidates = [pair for pair in candidates if pair[0] == end]
                    positions = tuple(p for p, _ in candidates)
                    planned_prefixes.update(digest for _, digest in candidates)

                self.block_manager.ensure_capacity(seq, end)
                prefill_chunks.append(
                    ScheduledChunk(
                        seq,
                        start,
                        end,
                        checkpoint_positions=positions,
                        checkpoint_interval=interval if any(p < end for p in positions) else 0,
                    )
                )
                num_batched_tokens += scheduled
                if end == len(seq):
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                else:
                    self.prefilling.append(seq)

    def _release_runtime_state(self, seq: Sequence) -> None:
        seq.pending_state_snapshot = None
        self.block_manager.deallocate(seq)
        self.state_manager.deallocate(seq)
        seq.committed_tokens = 0
        seq.clear_runtime_features()

    def clear(self) -> None:
        for seq in (*self.waiting, *self.prefilling, *self.running):
            self._release_runtime_state(seq)
        self.waiting.clear()
        self.prefilling.clear()
        self.running.clear()
        while self.block_manager.prefix_cache.evict():
            pass

    def schedule(self) -> SchedulerOutput:
        decode_chunks: list[ScheduledChunk] = []
        prefill_chunks: list[ScheduledChunk] = []
        prefix_cache = self.block_manager.prefix_cache

        # Accepted requests already reserve their maximum KV lifetime.
        for _ in range(min(len(self.running), self.max_num_seqs, self.max_num_batched_tokens)):
            seq = self.running.popleft()
            if seq.committed_tokens != len(seq) - 1:
                raise RuntimeError(f"sequence {seq.seq_id} decode prefix mismatch")
            while self.block_manager.max_schedulable_tokens(seq, 1) == 0:
                if not prefix_cache.evict(self.block_manager.block_refcounts):
                    raise RuntimeError("admitted decode cannot acquire reserved KV blocks")
            self.block_manager.ensure_capacity(seq, len(seq))
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

            self.block_manager.prefix_cache.publish_many(seq, prefix_snapshots.get(seq.seq_id, ()))

            seq.committed_tokens = chunk.end
            if seq.image_state is not None and chunk.end >= seq.image_state.image_end:
                seq.clear_runtime_features()
            if partial_prefill:
                continue

            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id in self.eos_token_ids) or (
                seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self._release_runtime_state(seq)
                self.running.remove(seq)
