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
        result[sample_mask] = self._sample(logits[sample_mask], temperatures[sample_mask])
        return result

    @staticmethod
    def _sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        from flashinfer.sampling import sampling_from_probs, softmax

        probs = softmax(logits, temperature=temperatures)
        return sampling_from_probs(probs).long()
