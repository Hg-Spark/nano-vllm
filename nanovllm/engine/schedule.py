from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class ScheduledChunk:
    """One request's token interval reserved for the current forward."""

    seq: Sequence
    start: int
    end: int
    capture_snapshot: bool = False

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError("scheduled chunk must have a non-empty range")

    @property
    def num_tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    decode_chunks: tuple[ScheduledChunk, ...] = ()
    prefill_chunks: tuple[ScheduledChunk, ...] = ()

    @property
    def decode_tokens(self) -> int:
        return sum(chunk.num_tokens for chunk in self.decode_chunks)

    @property
    def prefill_tokens(self) -> int:
        return sum(chunk.num_tokens for chunk in self.prefill_chunks)
