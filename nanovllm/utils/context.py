from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class Context:
    is_prefill: bool = False
    kv_batch_indices: torch.Tensor | None = None
    kv_positions: torch.Tensor | None = None
    qo_indptr: torch.Tensor | None = None
    paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_last_page_len: torch.Tensor | None = None
    attention_wrapper: object | None = None
    state_slots: tuple[int, ...] | None = None
    state_prefix_lens: tuple[int, ...] | None = None
    prefill_q_offsets: tuple[int, ...] | None = None
    state_slot_ids: torch.Tensor | None = None


_CONTEXT: ContextVar[Context] = ContextVar("nanovllm_context", default=Context())


def get_context() -> Context:
    return _CONTEXT.get()


@contextmanager
def use_context(context: Context) -> Iterator[None]:
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)
