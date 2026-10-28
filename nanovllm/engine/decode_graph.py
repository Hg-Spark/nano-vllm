import math
from dataclasses import dataclass

import torch

from nanovllm.engine.batch_common import DecodeBatchLayout, stage_pinned
from nanovllm.utils.context import Context, use_context

FLASHINFER_GRAPH_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass
class DecodeGraphBucket:
    batch_size: int
    max_model_len: int
    block_size: int
    workspace: torch.Tensor

    def __post_init__(self) -> None:
        import flashinfer

        if self.batch_size <= 0 or self.block_size <= 0:
            raise ValueError("decode graph batch_size and block_size must be positive")

        self.max_page_indices = self.batch_size * math.ceil(self.max_model_len / self.block_size)
        self.input_ids = torch.zeros(self.batch_size, dtype=torch.int64, device="cuda")
        self.rope_positions = torch.zeros(1, self.batch_size, dtype=torch.int64, device="cuda")
        self.kv_batch_indices = torch.arange(self.batch_size, dtype=torch.int32, device="cuda")
        self.kv_positions = torch.zeros(self.batch_size, dtype=torch.int32, device="cuda")
        self.state_slot_ids = torch.zeros(self.batch_size, dtype=torch.int32, device="cuda")
        self.paged_kv_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32, device="cuda")
        self.paged_kv_indices = torch.zeros(self.max_page_indices, dtype=torch.int32, device="cuda")
        self.paged_kv_last_page_len = torch.ones(self.batch_size, dtype=torch.int32, device="cuda")
        # Reuse page-locked host staging and wait before modifying an in-flight buffer.
        self._host_buffers = {
            name: torch.empty_like(getattr(self, name), device="cpu", pin_memory=True)
            for name in (
                "input_ids",
                "rope_positions",
                "kv_positions",
                "state_slot_ids",
                "paged_kv_indptr",
                "paged_kv_indices",
                "paged_kv_last_page_len",
            )
        }
        self._host_transfer_done = torch.cuda.Event()
        self._has_pending_host_transfer = False
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=self.paged_kv_indptr,
            paged_kv_indices_buffer=self.paged_kv_indices,
            paged_kv_last_page_len_buffer=self.paged_kv_last_page_len,
            backend="auto",
        )
        self.context = Context(
            is_prefill=False,
            kv_batch_indices=self.kv_batch_indices,
            kv_positions=self.kv_positions,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            attention_wrapper=self.wrapper,
            state_slot_ids=self.state_slot_ids,
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None
        self.replay_count = 0
        self._page_table_key = None

    @property
    def captured(self) -> bool:
        return self.graph is not None and self.logits is not None

    def can_run(self, layout: DecodeBatchLayout) -> bool:
        return (
            len(layout.input_ids) == self.batch_size
            and len(layout.paged_kv_indices) <= self.max_page_indices
        )

    def _copy_layout(self, name: str, values) -> None:
        host = self._host_buffers[name]
        staged = stage_pinned(host, values)
        device = getattr(self, name)
        if name == "paged_kv_indices":
            device = device[: staged.numel()]
        device.copy_(staged, non_blocking=True)

    def load(self, layout: DecodeBatchLayout) -> None:
        if not self.can_run(layout):
            raise ValueError("decode layout does not fit graph bucket")
        if self._has_pending_host_transfer:
            self._host_transfer_done.synchronize()
        self._copy_layout("input_ids", layout.input_ids)
        self._copy_layout("rope_positions", layout.rope_positions)
        self._copy_layout("kv_positions", layout.kv_positions)
        self._copy_layout("state_slot_ids", layout.state_slots)
        stage_pinned(self._host_buffers["paged_kv_indptr"], layout.paged_kv_indptr)
        # The physical page mapping remains stable across tokens in the same
        # request rows. Only indptr/last-page-length need fresh planning.
        key = getattr(layout, "page_table_key", None)
        if key is None or key != getattr(self, "_page_table_key", None):
            self._copy_layout("paged_kv_indices", layout.paged_kv_indices)
        self._page_table_key = key
        stage_pinned(self._host_buffers["paged_kv_last_page_len"], layout.paged_kv_last_page_len)

    def _plan(self, num_page_indices: int) -> None:
        num_qo_heads, num_kv_heads, head_dim, plan_kwargs = self.attention_spec
        self.wrapper.plan(
            self._host_buffers["paged_kv_indptr"],
            self.paged_kv_indices[:num_page_indices],
            self._host_buffers["paged_kv_last_page_len"],
            num_qo_heads,
            num_kv_heads,
            head_dim,
            self.block_size,
            **plan_kwargs,
        )

    def prepare(self, layout: DecodeBatchLayout) -> None:
        self.load(layout)
        self._plan(len(layout.paged_kv_indices))
        # Include FlashInfer's async H2D copies before reusing pinned buffers.
        self._host_transfer_done.record(torch.cuda.current_stream())
        self._has_pending_host_transfer = True

    def _seed_capture_inputs(self) -> None:
        seq_ids = torch.arange(self.batch_size, dtype=torch.int32, device="cuda")
        self.input_ids.zero_()
        self.rope_positions.zero_()
        self.kv_positions.zero_()
        self.state_slot_ids.copy_(seq_ids)
        self.paged_kv_indptr.copy_(
            torch.arange(self.batch_size + 1, dtype=torch.int32, device="cuda")
        )
        self.paged_kv_indices[: self.batch_size].copy_(seq_ids)
        self.paged_kv_last_page_len.fill_(1)
        self._host_buffers["paged_kv_indptr"].copy_(
            torch.arange(self.batch_size + 1, dtype=torch.int32)
        )
        self._host_buffers["paged_kv_last_page_len"].fill_(1)

    def _clear_capture_state(self, model, kv_cache: torch.Tensor) -> None:
        kv_cache[:, :, : self.batch_size].zero_()
        for module in model.state_cache_modules():
            module.conv_state[: self.batch_size].zero_()
            module.recurrent_state[: self.batch_size].zero_()

    def capture(
        self, model, kv_cache: torch.Tensor, num_kv_cache_blocks: int, attention_spec: tuple
    ) -> None:
        if num_kv_cache_blocks < self.batch_size:
            raise RuntimeError("not enough KV pages to seed decode graph capture")

        self.attention_spec = attention_spec
        self._seed_capture_inputs()
        self._page_table_key = None
        capture_stream = torch.cuda.Stream()
        try:
            for is_capture in (False, True):
                self._clear_capture_state(model, kv_cache)
                self._plan(self.batch_size)
                capture_stream.wait_stream(torch.cuda.current_stream())
                if not is_capture:
                    with (
                        torch.cuda.stream(capture_stream),
                        torch.inference_mode(),
                        use_context(self.context),
                    ):
                        model.compute_logits(model(self.input_ids, self.rope_positions))
                    torch.cuda.current_stream().wait_stream(capture_stream)
                    torch.cuda.synchronize()
                else:
                    graph = torch.cuda.CUDAGraph()
                    with (
                        torch.inference_mode(),
                        torch.cuda.graph(graph, stream=capture_stream),
                        use_context(self.context),
                    ):
                        logits = model.compute_logits(model(self.input_ids, self.rope_positions))
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
        self.replay_count += 1
        return self.logits

    def release(self) -> None:
        # Pinned buffers must remain alive until their pending H2D copies finish.
        if self._has_pending_host_transfer:
            self._host_transfer_done.synchronize()
            self._has_pending_host_transfer = False
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
        self._host_buffers = {}
        self._page_table_key = None
        self._host_transfer_done = None
        self.logits = self.wrapper = self.context = self.workspace = None
