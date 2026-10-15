import logging
from dataclasses import dataclass, replace

import flashinfer
import torch
from torch.profiler import record_function

from nanovllm.config import Config
from nanovllm.engine.cache_runtime import (
    allocate_runtime_caches,
    capture_gdn_state,
    restore_pending_states,
)
from nanovllm.engine.decode_batch import build_decode_batch_layout, prepare_decode
from nanovllm.engine.decode_graph import DecodeGraphBucket
from nanovllm.engine.moe_runtime import initialize_moe_runtime
from nanovllm.engine.prefill_batch import prepare_prefill
from nanovllm.engine.schedule import BatchResult, ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.layers.attention import flashinfer_attention_spec
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.utils.context import Context, use_context
from nanovllm.utils.loader import load_model

_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.block_size = config.kv_cache_block_size
        self._closed = False
        self.model = None
        self.sampler = None
        self.kv_cache = None
        self.attention_workspace = None
        self.moe_workspace = None
        self.moe_output = None
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.decode_graph_buckets: dict[int, DecodeGraphBucket] = {}

        previous_default_device = torch.get_default_device()
        previous_cuda_device = torch.cuda.current_device()
        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(getattr(torch, config.dtype))
        torch.set_default_device("cuda")
        try:
            self.model = Qwen3_5ForCausalLM(
                config.root_config,
                load_vision=config.max_visual_tokens_per_request != 0,
            )
            load_model(self.model, config.model)
            self.model.pack_linear_projections()
            self.model.eval()
            self.sampler = Sampler()
            self._init_moe_runtime()
            self.decode_graph_buckets = {
                batch_size: DecodeGraphBucket(
                    batch_size=batch_size,
                    max_model_len=config.max_model_len,
                    block_size=config.kv_cache_block_size,
                    model_config=config.text_config,
                )
                for batch_size in config.decode_graph_batch_sizes
            }
            self.attention_workspace = torch.zeros(
                _FLASHINFER_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device="cuda",
            )
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.attention_workspace, kv_layout="NHD", backend="auto"
            )
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.attention_workspace, kv_layout="NHD", backend="auto"
            )
            self.kv_cache, self.num_kv_cache_blocks = allocate_runtime_caches(self.model, config)
            if self.decode_graph_buckets:
                free_before, _ = torch.cuda.mem_get_info()
                allocated_before = torch.cuda.memory_allocated()
                reserved_before = torch.cuda.memory_reserved()
                for batch_size, bucket in self.decode_graph_buckets.items():
                    bucket.capture(self.model, self.kv_cache, self.num_kv_cache_blocks)
                    free_after, _ = torch.cuda.mem_get_info()
                    logger.info(
                        "Decode Graph bucket=%d: free_delta=%.1f MiB, "
                        "allocator_allocated_delta=%.1f MiB, allocator_reserved_delta=%.1f MiB, "
                        "remaining_free=%.1f MiB",
                        batch_size,
                        (free_before - free_after) / 2**20,
                        (torch.cuda.memory_allocated() - allocated_before) / 2**20,
                        (torch.cuda.memory_reserved() - reserved_before) / 2**20,
                        free_after / 2**20,
                    )
                    free_before = free_after
                    allocated_before = torch.cuda.memory_allocated()
                    reserved_before = torch.cuda.memory_reserved()
        except Exception:
            self._release_cuda_resources()
            raise
        finally:
            torch.set_default_device(previous_default_device)
            torch.set_default_dtype(default_dtype)
            torch.cuda.set_device(previous_cuda_device)

    def _init_moe_runtime(self) -> None:
        if self.config.is_moe:
            self.moe_workspace, self.moe_output = initialize_moe_runtime(self.model, self.config)
        else:
            self.moe_workspace = self.moe_output = None

    def _release_cuda_resources(self) -> None:
        buckets = getattr(self, "decode_graph_buckets", {})
        for bucket in buckets.values():
            bucket.release()
        buckets.clear()
        self.kv_cache = None
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.attention_workspace = None
        self.moe_workspace = None
        self.moe_output = None
        self.model = None
        self.sampler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._release_cuda_resources()

    def _plan_attention(self, context: Context) -> Context:
        if (
            context.paged_kv_indptr is None
            or context.paged_kv_indices is None
            or context.paged_kv_last_page_len is None
        ):
            raise RuntimeError("regular attention requires paged KV metadata")

        model_dtype = next(self.model.parameters()).dtype
        num_qo_heads, num_kv_heads, head_dim, common = flashinfer_attention_spec(
            self.config.text_config, model_dtype, self.kv_cache.dtype
        )
        if context.is_prefill:
            if context.qo_indptr is None:
                raise RuntimeError("FlashInfer paged prefill requires query offsets")
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

    def _ensure_visual_features(self, seq: Sequence) -> torch.Tensor:
        state = seq.image_state
        if state is None:
            raise RuntimeError("sequence has no image state")
        if state.visual_features is None:
            features = self.model.encode_image(
                state.pixel_values.cuda(non_blocking=True),
                state.image_grid_thw.cuda(non_blocking=True),
            )
            if features.shape[0] != state.image_end - state.image_start:
                raise RuntimeError("visual feature count does not match placeholder span")
            state.visual_features = features.detach()
        return state.visual_features

    def _sample_logits(
        self,
        logits: torch.Tensor,
        seqs: list[Sequence],
        sample_indices: list[int],
    ) -> list[int | None]:
        results: list[int | None] = [None for _ in seqs]
        if not sample_indices:
            return results
        if all(seqs[idx].temperature <= 1e-10 for idx in sample_indices):
            token_ids = logits.argmax(dim=-1).tolist()
        else:
            temperatures = torch.tensor(
                [seqs[idx].temperature for idx in sample_indices],
                dtype=torch.float32,
                pin_memory=True,
            ).cuda(non_blocking=True)
            token_ids = self.sampler(logits, temperatures).tolist()
        for idx, token_id in zip(sample_indices, token_ids):
            results[idx] = token_id
        return results

    def _run_decode_graph(
        self,
        chunks: tuple[ScheduledChunk, ...],
        seqs: list[Sequence],
    ) -> BatchResult | None:
        bucket = self.decode_graph_buckets.get(len(chunks))
        if bucket is None:
            return None
        layout = build_decode_batch_layout(chunks, self.block_size)
        if not bucket.can_run(layout):
            raise RuntimeError(f"configured decode graph bucket {len(chunks)} cannot run layout")
        bucket.prepare(layout, self.model, self.kv_cache)
        logits = bucket.replay()
        return BatchResult(self._sample_logits(logits, seqs, list(range(len(chunks)))), {})

    @torch.inference_mode()
    def run(
        self,
        chunks: tuple[ScheduledChunk, ...],
        is_prefill: bool,
    ) -> BatchResult:
        seqs = [chunk.seq for chunk in chunks]
        if not is_prefill and any(seq.pending_state_snapshot is not None for seq in seqs):
            raise RuntimeError("pending GDN prefix restore is valid only for prefill")
        restore_pending_states(self.model, seqs)

        if not is_prefill:
            graph_result = self._run_decode_graph(chunks, seqs)
            if graph_result is not None:
                return graph_result

        batch = (
            prepare_prefill(chunks, self.block_size)
            if is_prefill
            else prepare_decode(chunks, self.block_size)
        )
        context = self._plan_attention(batch.context)
        profile_range = "nanovllm::prefill_model" if is_prefill else "nanovllm::decode_model"

        with use_context(context):
            with record_function(profile_range):
                if batch.image_copy_spans:
                    inputs_embeds = self.model.embed_input_ids(batch.input_ids)
                    for span in batch.image_copy_spans:
                        features = self._ensure_visual_features(span.seq)
                        dst_end = span.packed_dst_start + span.length
                        src_end = span.feature_src_start + span.length
                        inputs_embeds[span.packed_dst_start : dst_end].copy_(
                            features[span.feature_src_start : src_end].to(dtype=inputs_embeds.dtype)
                        )
                    hidden_states = self.model(None, batch.positions, inputs_embeds=inputs_embeds)
                else:
                    hidden_states = self.model(batch.input_ids, batch.positions)

            sample_indices = (
                [idx for idx, chunk in enumerate(chunks) if chunk.end == len(chunk.seq)]
                if is_prefill
                else list(range(len(chunks)))
            )
            if sample_indices:
                logits = self.model.compute_logits(
                    hidden_states, sample_indices if is_prefill else None
                )
                results = self._sample_logits(logits, seqs, sample_indices)
            else:
                results = [None for _ in seqs]

        prefix_snapshots = {
            chunk.seq.seq_id: capture_gdn_state(self.model, chunk)
            for chunk in chunks
            if chunk.capture_snapshot
        }
        return BatchResult(results, prefix_snapshots)
