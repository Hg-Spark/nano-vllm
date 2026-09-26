from __future__ import annotations

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import TYPE_CHECKING

from nanovllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from nanovllm.engine.state_manager import GDNStateSnapshot


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
    ):
        if not token_ids:
            raise ValueError("sequence must contain at least one token")
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        # Single logical boundary represented by both paged KV and GDN state.
        # Hybrid history is committed atomically after a successful model step.
        self.committed_tokens = 0
        self.block_table: list[int] = []
        self.state_slot = -1
        # Set only on a joint-prefix hit. ModelRunner consumes the snapshot
        # immediately before executing the resumed prefill chunk.
        self.pending_state_snapshot: GDNStateSnapshot | None = None
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def last_token(self):
        return self.token_ids[-1]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
