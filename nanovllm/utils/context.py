from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
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
