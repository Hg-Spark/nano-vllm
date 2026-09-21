import torch
from torch import nn


class Sampler(nn.Module):

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ):
        logits = logits.float()
        greedy = temperatures <= 1e-10
        result = logits.argmax(dim=-1)

        sample_mask = ~greedy
        if sample_mask.any():
            sampled_logits = (
                logits[sample_mask]
                / temperatures[sample_mask].unsqueeze(1)
            )
            probs = torch.softmax(
                sampled_logits,
                dim=-1,
            )
            sampled = probs.div_(
                torch.empty_like(probs)
                .exponential_(1)
                .clamp_min_(1e-10)
            ).argmax(dim=-1)
            result[sample_mask] = sampled
        return result
