import torch
from torch import nn


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        *,
        sample_positions: torch.Tensor | None = None,
    ):
        logits = logits.float()
        if sample_positions is None:
            return self._sample(logits, temperatures)
        result = logits.argmax(dim=-1)
        if sample_positions.numel():
            selected = self._sample(
                logits.index_select(0, sample_positions),
                temperatures.index_select(0, sample_positions),
            )
            result.index_copy_(0, sample_positions, selected)
        return result

    @staticmethod
    def _sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        from flashinfer.sampling import sampling_from_probs, softmax

        probs = softmax(logits, temperature=temperatures)
        return sampling_from_probs(probs).long()
