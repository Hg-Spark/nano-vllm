from dataclasses import dataclass, replace

import flashinfer
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
from nanovllm.utils.context import Context, use_context
from nanovllm.utils.loader import load_model


_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024


def _validate_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("nano-vLLM requires a CUDA GPU")
    if torch.version.cuda != "13.0":
        raise RuntimeError(
            "this branch targets PyTorch CUDA 13.0 (cu130); "
            f"found torch.version.cuda={torch.version.cuda!r}"
        )


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

        _validate_cuda_runtime()
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

            # FlashInfer recommends a 128 MiB caller-owned workspace. The
            # prefill and decode wrappers execute serially, so they share it.
            self.attention_workspace = torch.zeros(
                _FLASHINFER_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device="cuda",
            )
            self.prefill_wrapper = (
                flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.attention_workspace,
                    kv_layout="NHD",
                    backend="auto",
                )
            )
            self.decode_wrapper = (
                flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self.attention_workspace,
                    kv_layout="NHD",
                    backend="auto",
                )
            )

            # Warmup intentionally runs before persistent cache allocation so
            # the cache budget accounts for peak model activation memory.
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
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.attention_workspace = None
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

    def _plan_attention(self, context: Context) -> Context:
        if context.paged_kv_indptr is None:
            # Startup warmup has no persistent KV pages yet.
            return context

        if (
            context.paged_kv_indices is None
            or context.paged_kv_last_page_len is None
        ):
            raise RuntimeError("incomplete FlashInfer paged-KV metadata")

        model_dtype = next(self.model.parameters()).dtype
        kv_dtype = self.kv_cache.dtype
        config = self.config.text_config
        num_qo_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        common = dict(
            q_data_type=model_dtype,
            kv_data_type=kv_dtype,
            o_data_type=model_dtype,
            pos_encoding_mode="NONE",
            sm_scale=head_dim ** -0.5,
        )

        if context.is_prefill:
            if context.qo_indptr is None:
                raise RuntimeError(
                    "FlashInfer paged prefill requires query offsets"
                )
            self.prefill_wrapper.plan(
                context.qo_indptr,
                context.paged_kv_indptr,
                context.paged_kv_indices,
                context.paged_kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                self.block_size,
                causal=True,
                **common,
            )
            wrapper = self.prefill_wrapper
        else:
            self.decode_wrapper.plan(
                context.paged_kv_indptr,
                context.paged_kv_indices,
                context.paged_kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                self.block_size,
                **common,
            )
            wrapper = self.decode_wrapper

        return replace(context, attention_wrapper=wrapper)

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
        context = self._plan_attention(batch.context)

        profile_range = (
            "nanovllm::prefill_model"
            if is_prefill
            else "nanovllm::decode_model"
        )
        with use_context(context):
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
