import torch
from torch import nn


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        *,
        all_sampling: bool = False,
    ):
        logits = logits.float()
        if all_sampling:
            return self._sample(logits, temperatures)
        greedy = temperatures <= 1e-10
        result = logits.argmax(dim=-1)

        sample_mask = ~greedy
        if sample_mask.any():
            result[sample_mask] = self._sample(logits[sample_mask], temperatures[sample_mask])
        return result

    @staticmethod
    def _sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
