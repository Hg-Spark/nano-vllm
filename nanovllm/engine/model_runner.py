from dataclasses import dataclass

import torch
from torch.profiler import record_function

from nanovllm.config import Config
from nanovllm.engine.batch import prepare_decode, prepare_prefill
from nanovllm.engine.cache_runtime import (
    allocate_runtime_caches,
    capture_gdn_state,
    restore_pending_states,
)
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
from nanovllm.utils.context import use_context
from nanovllm.utils.loader import load_model


def _resolve_dtype(config) -> torch.dtype:
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return dtype or torch.bfloat16


@dataclass(slots=True)
class BatchResult:
    token_ids: list[int | None]
    prefix_snapshots: dict[int, GDNStateSnapshot]


class ModelRunner:

    def __init__(self, config: Config):
        self.config = config
        self.block_size = config.kvcache_block_size
        self._closed = False

        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        dtype = _resolve_dtype(config.text_config)
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")
        try:
            self.model = Qwen3_5MoeForCausalLM(config.text_config)
            load_model(self.model, config.model)
            self.model.eval()
            self.sampler = Sampler()
            self.warmup_model()
            (
                self.kv_cache,
                self.num_kvcache_blocks,
            ) = allocate_runtime_caches(
                self.model,
                config,
            )
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)

    def exit(self):
        if self._closed:
            return
        self._closed = True

        torch.cuda.synchronize()
        self.kv_cache = None
        self.model = None
        self.sampler = None
        torch.cuda.empty_cache()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seq_len = min(
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
            32,
        )
        if seq_len <= 0:
            raise ValueError("warmup sequence length must be positive")
        seq = Sequence([0] * seq_len)
        self.run((ScheduledChunk(seq, 0, seq_len),), True)
        torch.cuda.empty_cache()

    def _sample_indices(
        self,
        chunks: tuple[ScheduledChunk, ...],
        is_prefill: bool,
    ) -> list[int]:
        if not is_prefill:
            return list(range(len(chunks)))
        return [
            idx
            for idx, chunk in enumerate(chunks)
            if chunk.end == len(chunk.seq)
        ]

    @torch.inference_mode()
    def run(
        self,
        chunks: tuple[ScheduledChunk, ...],
        is_prefill: bool,
    ) -> BatchResult:
        seqs = [chunk.seq for chunk in chunks]
        if not is_prefill and any(
            seq.pending_state_snapshot is not None
            for seq in seqs
        ):
            raise RuntimeError(
                "pending GDN prefix restore is valid only for prefill"
            )
        restore_pending_states(self.model, seqs)

        if is_prefill:
            batch = prepare_prefill(chunks, self.block_size)
        else:
            batch = prepare_decode(chunks, self.block_size)

        profile_range = (
            "nanovllm::prefill_model"
            if is_prefill
            else "nanovllm::decode_model"
        )
        with use_context(batch.context):
            with record_function(profile_range):
                hidden_states = self.model(
                    batch.input_ids,
                    batch.positions,
                )

            sample_indices = self._sample_indices(
                chunks,
                is_prefill,
            )
            results: list[int | None] = [None for _ in seqs]
            if sample_indices:
                logits = self.model.compute_logits(
                    hidden_states,
                    sample_indices if is_prefill else None,
                )
                temperatures = torch.tensor(
                    [
                        seqs[idx].temperature
                        for idx in sample_indices
                    ],
                    dtype=torch.float32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                sampled = self.sampler(
                    logits,
                    temperatures,
                ).tolist()
                for idx, token_id in zip(
                    sample_indices,
                    sampled,
                ):
                    results[idx] = token_id

        prefix_snapshots = {
            chunk.seq.seq_id: capture_gdn_state(self.model, chunk)
            for chunk in chunks
            if chunk.capture_snapshot
        }
        return BatchResult(results, prefix_snapshots)
