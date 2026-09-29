from dataclasses import dataclass, replace

import flashinfer
import torch
import warnings
from torch.profiler import record_function

from nanovllm.config import Config
from nanovllm.engine.batch import (
    build_decode_batch_layout,
    prepare_decode,
    prepare_prefill,
)
from nanovllm.engine.cache_runtime import (
    allocate_runtime_caches,
    capture_gdn_state,
    restore_pending_states,
)
from nanovllm.engine.decode_graph import (
    DecodeGraphBucket,
    DecodeGraphPlanError,
)
from nanovllm.engine.moe_runtime import initialize_moe_runtime
from nanovllm.engine.runtime_warmup import warmup_runtime
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.layers.attention import flashinfer_attention_spec
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
        self.model = None
        self.sampler = None
        self.kv_cache = None
        self.attention_workspace = None
        self.moe_workspace = None
        self.moe_output = None
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.decode_graph_buckets: dict[int, DecodeGraphBucket] = {}

        _validate_cuda_runtime()
        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        dtype = _resolve_dtype(config.text_config)
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")
        try:
            self.model = Qwen3_5MoeForCausalLM(config.root_config)
            load_model(self.model, config.model)
            self.model.eval()
            self.sampler = Sampler()
            self._init_moe_runtime()
            self._init_decode_graph_buckets()

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
            self._capture_decode_graphs()
        except Exception:
            self._release_cuda_resources()
            raise
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)

    def _release_cuda_resources(self):
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
        torch.cuda.empty_cache()

    def exit(self):
        if self._closed:
            return
        self._closed = True

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._release_cuda_resources()

    def _init_moe_runtime(self) -> None:
        self.moe_workspace, self.moe_output = initialize_moe_runtime(
            self.model,
            self.config,
        )

    def _init_decode_graph_buckets(self) -> None:
        buckets: dict[int, DecodeGraphBucket] = {}
        for batch_size in self.config.decode_graph_batch_sizes:
            try:
                bucket = DecodeGraphBucket(
                    batch_size=batch_size,
                    max_model_len=self.config.max_model_len,
                    block_size=self.config.kvcache_block_size,
                    model_config=self.config.text_config,
                )
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                warnings.warn(
                    "decode CUDA graph workspace allocation ran out of memory "
                    f"at B={batch_size}; disabling this and larger buckets and "
                    f"using eager decode instead: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            buckets[batch_size] = bucket
        self.decode_graph_buckets = buckets

    def _capture_decode_graphs(self) -> None:
        if not self.decode_graph_buckets:
            return

        items = sorted(self.decode_graph_buckets.items())
        captured: dict[int, DecodeGraphBucket] = {}
        for index, (batch_size, bucket) in enumerate(items):
            try:
                bucket.capture(
                    self.model,
                    self.kv_cache,
                    self.num_kvcache_blocks,
                )
            except torch.OutOfMemoryError as exc:
                bucket.release()
                for _, larger_bucket in items[index + 1 :]:
                    larger_bucket.release()
                torch.cuda.empty_cache()
                warnings.warn(
                    "decode CUDA graph capture ran out of memory at "
                    f"B={batch_size}; keeping smaller captured buckets and "
                    "using eager decode for this and larger batches: "
                    f"{exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            except Exception as exc:
                bucket.release()
                warnings.warn(
                    "disabling decode CUDA graph bucket "
                    f"B={batch_size}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            captured[batch_size] = bucket
        self.decode_graph_buckets = captured

    def warmup_model(self):
        warmup_runtime(
            self.model,
            self.config,
            self.decode_graph_buckets,
            self.run,
        )

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
        config = self.config.text_config
        (
            num_qo_heads,
            num_kv_heads,
            head_dim,
            common,
        ) = flashinfer_attention_spec(
            config,
            model_dtype,
            self.kv_cache.dtype,
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

    def _ensure_visual_features(
        self,
        seq: Sequence,
    ) -> torch.Tensor:
        state = seq.image_state
        if state is None:
            raise RuntimeError("sequence has no image state")
        if state.visual_features is None:
            pixel_values = state.pixel_values.cuda(non_blocking=True)
            image_grid_thw = state.image_grid_thw.cuda(
                non_blocking=True
            )
            features = self.model.encode_image(
                pixel_values,
                image_grid_thw,
            )
            expected = state.image_end - state.image_start
            if features.shape[0] != expected:
                raise RuntimeError(
                    "visual feature count does not match placeholder span"
                )
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
        temperatures = torch.tensor(
            [
                seqs[idx].temperature
                for idx in sample_indices
            ],
            dtype=torch.float32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        sampled = self.sampler(logits, temperatures).tolist()
        for idx, token_id in zip(sample_indices, sampled):
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

        layout = build_decode_batch_layout(
            chunks,
            self.block_size,
        )
        if not bucket.can_run(layout):
            return None

        try:
            bucket.prepare(layout, self.model, self.kv_cache)
        except DecodeGraphPlanError as exc:
            bucket.release()
            self.decode_graph_buckets.pop(len(chunks), None)
            warnings.warn(
                "decode graph planning failed; disabling bucket and "
                f"using eager decode: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # replay() mutates KV/GDN state. Failures must propagate so the
        # engine can use its existing failed-step recovery path.
        logits = bucket.replay()
        sample_indices = list(range(len(chunks)))
        return BatchResult(
            self._sample_logits(
                logits,
                seqs,
                sample_indices,
            ),
            {},
        )

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

        if not is_prefill:
            graph_result = self._run_decode_graph(chunks, seqs)
            if graph_result is not None:
                return graph_result

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
                if batch.image_copy_spans:
                    inputs_embeds = self.model.embed_input_ids(
                        batch.input_ids
                    )
                    for span in batch.image_copy_spans:
                        features = self._ensure_visual_features(span.seq)
                        dst_start = span.packed_dst_start
                        dst_end = dst_start + span.length
                        src_start = span.feature_src_start
                        src_end = src_start + span.length
                        inputs_embeds[dst_start:dst_end].copy_(
                            features[src_start:src_end].to(
                                dtype=inputs_embeds.dtype
                            )
                        )
                    hidden_states = self.model(
                        None,
                        batch.positions,
                        inputs_embeds=inputs_embeds,
                    )
                else:
                    hidden_states = self.model(
                        batch.input_ids,
                        batch.positions,
                    )

            sample_indices = self._sample_indices(
                chunks,
                is_prefill,
            )
            if sample_indices:
                logits = self.model.compute_logits(
                    hidden_states,
                    sample_indices if is_prefill else None,
                )
                results = self._sample_logits(
                    logits,
                    seqs,
                    sample_indices,
                )
            else:
                results = [None for _ in seqs]

        prefix_snapshots = {
            chunk.seq.seq_id: capture_gdn_state(self.model, chunk)
            for chunk in chunks
            if chunk.capture_snapshot
        }
        return BatchResult(results, prefix_snapshots)
