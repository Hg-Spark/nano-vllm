from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Context:
    is_prefill: bool = False
    slot_mapping: torch.Tensor | None = None

    # FlashInfer paged-attention metadata. qo_indptr is needed only for
    # packed prefill; decode has one query token per request.
    qo_indptr: torch.Tensor | None = None
    paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_last_page_len: torch.Tensor | None = None
    attention_wrapper: object | None = None

    # GDN state metadata stays CPU-side where possible.
    state_slots: tuple[int, ...] | None = None
    state_prefix_lens: tuple[int, ...] | None = None
    prefill_q_offsets: tuple[int, ...] | None = None


_CONTEXT: ContextVar[Context] = ContextVar(
    "nanovllm_context",
    default=Context(),
)


def get_context() -> Context:
    return _CONTEXT.get()


@contextmanager
def use_context(context: Context):
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)
