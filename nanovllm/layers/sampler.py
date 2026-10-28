import torch
from torch import nn


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        *,
        sample_positions: list[int] | None = None,
    ):
        logits = logits.float()
        if sample_positions is None:
            return self._sample(logits, temperatures)
        result = logits.argmax(dim=-1)
        if sample_positions:
            indices = torch.tensor(sample_positions, device=logits.device, dtype=torch.long)
            selected = self._sample(
                logits.index_select(0, indices), temperatures.index_select(0, indices)
            )
            result.index_copy_(0, indices, selected)
        return result

    @staticmethod
    def _sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        from flashinfer.sampling import sampling_from_probs, softmax

        probs = softmax(logits, temperature=temperatures)
        return sampling_from_probs(probs).long()
