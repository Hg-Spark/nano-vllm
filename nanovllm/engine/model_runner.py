import logging

import torch
from torch.profiler import record_function

from nanovllm.config import Config
from nanovllm.engine.batch_common import (
    DecodeBatchLayout,
    DecodePageTableCache,
    PinnedMetadataPool,
    build_decode_batch_layout,
    prepare_decode,
    prepare_prefill,
)
from nanovllm.engine.cache_runtime import (
    GDNCompletionBuffer,
    GDNRestoreStaging,
    allocate_runtime_caches,
    restore_pending_states,
)
from nanovllm.engine.decode_graph import FLASHINFER_GRAPH_WORKSPACE_BYTES, DecodeGraphBucket
from nanovllm.engine.schedule import BatchResult, ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.layers.attention import flashinfer_attention_spec, plan_flashinfer_attention
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.models.qwen3_5_moe import initialize_moe_runtime
from nanovllm.utils.context import use_context
from nanovllm.utils.loader import load_model

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, config: Config) -> None:
        import flashinfer

        self.config = config
        self.block_size = config.kv_cache_block_size
        self._closed = False
        self._cleanup_complete = False
        self.model = None
        self.sampler = None
        # Optional benchmark observer; never enabled during formal latency measurements.
        self.logits_observer = None
        self.kv_cache = None
        self.attention_workspace = None
        self.metadata_pool = PinnedMetadataPool()
        self.decode_page_table_cache = DecodePageTableCache()
        self.gdn_restore_staging = GDNRestoreStaging()
        self.gdn_completion_buffer = None
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.decode_graph_buckets: dict[int, DecodeGraphBucket] = {}
        self.decode_graph_fallbacks = {"missing_bucket": 0, "layout_exceeded": 0}

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(getattr(torch, config.dtype))
        try:
            if torch.cuda.get_device_capability()[0] not in (9, 10, 12):
                raise RuntimeError("FlashInfer GDN requires SM90/SM100/SM120")
            from transformers.initialization import no_init_weights

            # Every registered parameter is replaced by load_model(), which
            # rejects missing weights. Avoid initializing large GPU parameter
            # tensors only to overwrite them immediately from the checkpoint.
            with no_init_weights(), torch.device("cuda"):
                self.model = Qwen3_5ForCausalLM(
                    config.root_config,
                    load_vision=config.max_visual_tokens_per_request != 0,
                )
            load_model(self.model, config.model)
            self.model.eval()
            self.sampler = Sampler()
            if config.is_moe:
                initialize_moe_runtime(self.model, config)
            self.attention_workspace = torch.zeros(
                FLASHINFER_GRAPH_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device="cuda",
            )
            self.decode_graph_buckets = {
                batch_size: DecodeGraphBucket(
                    batch_size=batch_size,
                    max_model_len=config.max_model_len,
                    block_size=config.kv_cache_block_size,
                    workspace=self.attention_workspace,
                )
                for batch_size in config.decode_graph_batch_sizes
            }
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.attention_workspace, kv_layout="NHD", backend="auto"
            )
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.attention_workspace, kv_layout="NHD", backend="auto"
            )
            self.kv_cache, self.num_kv_cache_blocks = allocate_runtime_caches(self.model, config)
            self.snapshot_nbytes = sum(
                module.state_cache_nbytes(1) for module in self.model.state_cache_modules()
            )
            if config.max_prefix_cache_entries > 0 and (
                config.max_prefix_cache_snapshot_bytes is None
                or config.max_prefix_cache_snapshot_bytes >= self.snapshot_nbytes
            ):
                self.gdn_completion_buffer = GDNCompletionBuffer(self.model.state_cache_modules())
            self.attention_spec = flashinfer_attention_spec(
                config.text_config,
                self.model.model.embed_tokens.weight.dtype,
                self.kv_cache.dtype,
            )
            if self.decode_graph_buckets:
                free_before, _ = torch.cuda.mem_get_info()
                allocated_before = torch.cuda.memory_allocated()
                reserved_before = torch.cuda.memory_reserved()
                for batch_size, bucket in self.decode_graph_buckets.items():
                    bucket.capture(
                        self.model, self.kv_cache, self.num_kv_cache_blocks, self.attention_spec
                    )
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
            torch.set_default_dtype(default_dtype)

    def _release_cuda_resources(self) -> None:
        # Tear down independent owners even if an earlier release fails. Keep
        # failed owners reachable so exit() can retry their cleanup.
        failures = []
        for batch_size, bucket in tuple(self.decode_graph_buckets.items()):
            try:
                bucket.release()
            except BaseException as exc:
                failures.append(exc)
            else:
                del self.decode_graph_buckets[batch_size]

        for name in (
            "metadata_pool",
            "decode_page_table_cache",
            "gdn_restore_staging",
            "gdn_completion_buffer",
        ):
            owner = getattr(self, name, None)
            if owner is None:
                continue
            try:
                owner.clear()
            except BaseException as exc:
                failures.append(exc)
            else:
                setattr(self, name, None)

        self.kv_cache = None
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.attention_workspace = None
        self.model = None
        self.sampler = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise failures[0]

    def exit(self) -> None:
        if getattr(self, "_cleanup_complete", False):
            return
        # Closed prevents further execution, but it is not a cleanup-success flag.
        self._closed = True
        failure = None
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except BaseException as exc:
            failure = exc
        try:
            self._release_cuda_resources()
        except BaseException as exc:
            if failure is None:
                failure = exc
        if failure is not None:
            raise failure
        self._cleanup_complete = True

    def _ensure_visual_features(self, seq: Sequence) -> torch.Tensor:
        state = seq.image_state
        if state is None:
            raise RuntimeError("sequence has no image state")
        if state.visual_features is None:
            features = self.model.encode_image(
                state.pixel_values.cuda(non_blocking=True),
                state.image_grid_thw.cuda(non_blocking=True),
                state.image_end - state.image_start,
            )
            state.visual_features = features.detach().to(
                dtype=self.model.model.embed_tokens.weight.dtype
            )
            state.pixel_values = None
            state.image_grid_thw = None
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
            values = [seqs[idx].temperature for idx in sample_indices]
            positions = [index for index, value in enumerate(values) if value > 1e-10]
            with self.metadata_pool.batch() as upload:
                temperatures = upload("sample_temperatures", values, "float32")
                sampled_indices = (
                    upload("sample_positions", positions, "int64")
                    if len(positions) < len(values)
                    else None
                )
            token_ids = self.sampler(
                logits, temperatures, sample_positions=sampled_indices
            ).tolist()
        observer = getattr(self, "logits_observer", None)
        if observer is not None:
            # Observe the exact sampling rows, including CUDA Graph replay outputs.
            # Diagnostic observers may synchronize; keep them strictly opt-in.
            observer(logits, [seqs[idx] for idx in sample_indices], token_ids)
        for idx, token_id in zip(sample_indices, token_ids):
            results[idx] = token_id
        return results

    def _run_decode_graph(
        self,
        chunks: tuple[ScheduledChunk, ...],
        seqs: list[Sequence],
        layout: DecodeBatchLayout | None,
    ) -> BatchResult | None:
        bucket = self.decode_graph_buckets.get(len(chunks))
        if bucket is None:
            self.decode_graph_fallbacks["missing_bucket"] += 1
            return None
        if layout is None:
            raise RuntimeError("a configured Decode Graph bucket requires a prepared layout")
        if not bucket.can_run(layout):
            self.decode_graph_fallbacks["layout_exceeded"] += 1
            return None
        bucket.prepare(layout)
        logits = bucket.replay()
        return BatchResult(self._sample_logits(logits, seqs, list(range(len(chunks)))))

    @torch.inference_mode()
    def run(
        self,
        chunks: tuple[ScheduledChunk, ...],
        is_prefill: bool,
    ) -> BatchResult:
        seqs = [chunk.seq for chunk in chunks]
        if not is_prefill and any(seq.pending_state_snapshot is not None for seq in seqs):
            raise RuntimeError("pending GDN prefix restore is valid only for prefill")
        decode_layout = None
        if is_prefill:
            restore_pending_states(self.model, seqs, self.gdn_restore_staging)
        elif self.decode_graph_buckets:
            if len(chunks) in self.decode_graph_buckets:
                decode_layout = build_decode_batch_layout(
                    chunks, self.block_size, self.decode_page_table_cache
                )
            graph_result = self._run_decode_graph(chunks, seqs, decode_layout)
            if graph_result is not None:
                return graph_result
        else:
            # Preserve benchmark accounting for deliberately disabled CUDA Graph.
            self.decode_graph_fallbacks["missing_bucket"] += 1

        batch = (
            prepare_prefill(chunks, self.block_size, self.metadata_pool)
            if is_prefill
            else prepare_decode(
                chunks,
                self.block_size,
                self.metadata_pool,
                self.decode_page_table_cache,
                layout=decode_layout,
            )
        )
        context = plan_flashinfer_attention(
            batch,
            self.metadata_pool,
            self.prefill_wrapper,
            self.decode_wrapper,
            self.attention_spec,
            self.block_size,
        )
        profile_range = "nanovllm::prefill_model" if is_prefill else "nanovllm::decode_model"

        with use_context(context):
            with record_function(profile_range):
                if is_prefill and batch.layout.image_copy_spans:
                    inputs_embeds = self.model.model.embed_tokens(batch.input_ids)
                    for span in batch.layout.image_copy_spans:
                        features = self._ensure_visual_features(span.seq)
                        dst_end = span.packed_dst_start + span.length
                        src_end = span.feature_src_start + span.length
                        inputs_embeds[span.packed_dst_start : dst_end].copy_(
                            features[span.feature_src_start : src_end]
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
                if is_prefill:
                    # Reuse the pinned metadata pool instead of creating a
                    # fresh CUDA tensor from a Python list on each prefill.
                    last_indices = [
                        batch.layout.q_offsets[index + 1] - 1 for index in sample_indices
                    ]
                    indices = self.metadata_pool.to_cuda(
                        "logit_token_indices", last_indices, "int64"
                    )
                    logits = self.model.compute_logits(hidden_states, token_indices=indices)
                else:
                    logits = self.model.compute_logits(hidden_states)
                results = self._sample_logits(logits, seqs, sample_indices)
            else:
                results = [None for _ in seqs]

        return BatchResult(results)

    def stage_completion_states(self, candidates: list[tuple[Sequence, int]]) -> None:
        if candidates and self.gdn_completion_buffer is not None:
            self.gdn_completion_buffer.update(self.model.state_cache_modules(), candidates)

    def take_completion_states(self, finished: list[Sequence]) -> dict[int, GDNStateSnapshot]:
        if self.gdn_completion_buffer is None:
            return {}
        return self.gdn_completion_buffer.take(finished)

    @torch.inference_mode()
    def capture_live_states(
        self, candidates: list[tuple[Sequence, int]]
    ) -> dict[int, GDNStateSnapshot]:
        """Capture immutable prompt checkpoints at their exact KV-page boundary."""
        if not candidates:
            return {}
        modules = self.model.state_cache_modules()
        conv_cpu = torch.stack(
            [module.slot(seq.state_slot)[0] for seq, _ in candidates for module in modules]
        ).cpu()
        recurrent_cpu = torch.stack(
            [module.slot(seq.state_slot)[1] for seq, _ in candidates for module in modules]
        ).cpu()
        num_layers = len(modules)
        return {
            seq.seq_id: GDNStateSnapshot(
                position,
                tuple(
                    (
                        conv_cpu[index * num_layers + layer].clone(),
                        recurrent_cpu[index * num_layers + layer].clone(),
                    )
                    for layer in range(num_layers)
                ),
            )
            for index, (seq, position) in enumerate(candidates)
        }
