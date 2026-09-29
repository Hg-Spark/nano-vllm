import math
from dataclasses import dataclass

import flashinfer
import torch

from nanovllm.engine.batch import DecodeBatchLayout
from nanovllm.layers.attention import flashinfer_attention_spec
from nanovllm.utils.context import Context, use_context


FLASHINFER_GRAPH_WORKSPACE_BYTES = 128 * 1024 * 1024


class DecodeGraphPlanError(RuntimeError):
    """Planning failed before model state was mutated."""


@dataclass
class DecodeGraphBucket:
    batch_size: int
    max_model_len: int
    block_size: int
    model_config: object
    workspace_bytes: int = FLASHINFER_GRAPH_WORKSPACE_BYTES

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("decode graph batch size must be positive")
        if self.block_size <= 0:
            raise ValueError("decode graph block size must be positive")

        max_pages_per_seq = math.ceil(
            self.max_model_len / self.block_size
        )
        self.max_page_indices = (
            self.batch_size * max_pages_per_seq
        )
        self.workspace = torch.zeros(
            self.workspace_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        self.input_ids = torch.zeros(
            self.batch_size,
            dtype=torch.int64,
            device="cuda",
        )
        self.rope_positions = torch.zeros(
            3,
            self.batch_size,
            dtype=torch.int64,
            device="cuda",
        )
        self.slot_mapping = torch.zeros(
            self.batch_size,
            dtype=torch.int32,
            device="cuda",
        )
        self.state_slot_ids = torch.zeros(
            self.batch_size,
            dtype=torch.int32,
            device="cuda",
        )
        self.paged_kv_indptr = torch.arange(
            self.batch_size + 1,
            dtype=torch.int32,
            device="cuda",
        )
        self.paged_kv_indices = torch.zeros(
            self.max_page_indices,
            dtype=torch.int32,
            device="cuda",
        )
        self.paged_kv_last_page_len = torch.ones(
            self.batch_size,
            dtype=torch.int32,
            device="cuda",
        )
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=self.paged_kv_indptr,
            paged_kv_indices_buffer=self.paged_kv_indices,
            paged_kv_last_page_len_buffer=(
                self.paged_kv_last_page_len
            ),
            backend="auto",
        )
        self.context = Context(
            is_prefill=False,
            slot_mapping=self.slot_mapping,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            attention_wrapper=self.wrapper,
            state_slot_ids=self.state_slot_ids,
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None

    @property
    def captured(self) -> bool:
        return self.graph is not None and self.logits is not None

    def can_run(self, layout: DecodeBatchLayout) -> bool:
        return (
            len(layout.input_ids) == self.batch_size
            and len(layout.paged_kv_indices)
            <= self.max_page_indices
        )

    @staticmethod
    def _cpu_tensor(values, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            values,
            dtype=dtype,
            pin_memory=True,
        )

    def load(self, layout: DecodeBatchLayout) -> None:
        if not self.can_run(layout):
            raise ValueError("decode layout does not fit graph bucket")

        self.input_ids.copy_(
            self._cpu_tensor(layout.input_ids, torch.int64),
            non_blocking=True,
        )
        self.rope_positions.copy_(
            self._cpu_tensor(layout.rope_positions, torch.int64),
            non_blocking=True,
        )
        self.slot_mapping.copy_(
            self._cpu_tensor(layout.slot_mapping, torch.int32),
            non_blocking=True,
        )
        self.state_slot_ids.copy_(
            self._cpu_tensor(layout.state_slots, torch.int32),
            non_blocking=True,
        )
        self.paged_kv_indptr.copy_(
            self._cpu_tensor(layout.paged_kv_indptr, torch.int32),
            non_blocking=True,
        )
        num_indices = len(layout.paged_kv_indices)
        self.paged_kv_indices[:num_indices].copy_(
            self._cpu_tensor(layout.paged_kv_indices, torch.int32),
            non_blocking=True,
        )
        self.paged_kv_last_page_len.copy_(
            self._cpu_tensor(
                layout.paged_kv_last_page_len,
                torch.int32,
            ),
            non_blocking=True,
        )

    def _plan(
        self,
        num_page_indices: int,
        model,
        kv_cache: torch.Tensor,
    ) -> None:
        config = self.model_config
        model_dtype = next(model.parameters()).dtype
        (
            num_qo_heads,
            num_kv_heads,
            head_dim,
            plan_kwargs,
        ) = flashinfer_attention_spec(
            config,
            model_dtype,
            kv_cache.dtype,
        )
        try:
            self.wrapper.plan(
                self.paged_kv_indptr,
                self.paged_kv_indices[:num_page_indices],
                self.paged_kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                self.block_size,
                **plan_kwargs,
            )
        except Exception as exc:
            raise DecodeGraphPlanError(str(exc)) from exc

    def prepare(
        self,
        layout: DecodeBatchLayout,
        model,
        kv_cache: torch.Tensor,
    ) -> None:
        self.load(layout)
        self._plan(
            len(layout.paged_kv_indices),
            model,
            kv_cache,
        )

    def _seed_capture_inputs(self) -> None:
        self.input_ids.zero_()
        self.rope_positions.zero_()
        self.slot_mapping.copy_(
            torch.arange(
                self.batch_size,
                dtype=torch.int32,
                device="cuda",
            )
            * self.block_size
        )
        self.state_slot_ids.copy_(
            torch.arange(
                self.batch_size,
                dtype=torch.int32,
                device="cuda",
            )
        )
        self.paged_kv_indptr.copy_(
            torch.arange(
                self.batch_size + 1,
                dtype=torch.int32,
                device="cuda",
            )
        )
        self.paged_kv_indices[: self.batch_size].copy_(
            torch.arange(
                self.batch_size,
                dtype=torch.int32,
                device="cuda",
            )
        )
        self.paged_kv_last_page_len.fill_(1)

    def _clear_capture_state(
        self,
        model,
        kv_cache: torch.Tensor,
    ) -> None:
        kv_cache[:, :, : self.batch_size].zero_()
        for module in model.state_cache_modules():
            module.conv_state[: self.batch_size].zero_()
            module.recurrent_state[: self.batch_size].zero_()

    def capture(
        self,
        model,
        kv_cache: torch.Tensor,
        num_kvcache_blocks: int,
    ) -> None:
        if num_kvcache_blocks < self.batch_size:
            raise RuntimeError(
                "not enough KV pages to seed decode graph capture"
            )

        self._seed_capture_inputs()
        capture_stream = torch.cuda.Stream()
        try:
            self._clear_capture_state(model, kv_cache)
            self._plan(self.batch_size, model, kv_cache)

            # Trigger lazy Triton/FlashInfer compilation on a side stream.
            # CUDA Graph capture must not rely on the legacy/default stream.
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                with torch.inference_mode(), use_context(self.context):
                    hidden_states = model(
                        self.input_ids,
                        self.rope_positions,
                    )
                    model.compute_logits(hidden_states)
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()

            self._clear_capture_state(model, kv_cache)
            self._plan(self.batch_size, model, kv_cache)
            capture_stream.wait_stream(torch.cuda.current_stream())

            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(
                graph,
                stream=capture_stream,
            ):
                with use_context(self.context):
                    hidden_states = model(
                        self.input_ids,
                        self.rope_positions,
                    )
                    logits = model.compute_logits(hidden_states)

            torch.cuda.current_stream().wait_stream(capture_stream)
            self.graph = graph
            self.logits = logits
        finally:
            self._clear_capture_state(model, kv_cache)
            torch.cuda.synchronize()

    def replay(self) -> torch.Tensor:
        if not self.captured:
            raise RuntimeError("decode graph bucket has not been captured")
        self.graph.replay()
        return self.logits

    def release(self) -> None:
        graph = self.graph
        self.graph = None
        self.logits = None
        if graph is not None and hasattr(graph, "reset"):
            graph.reset()
        self.wrapper = None
        self.context = None
        self.workspace = None
