import math
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.ignore_eos, bool):
            raise ValueError("ignore_eos must be boolean")

    def __post_init__(self) -> None:
        self.validate()
