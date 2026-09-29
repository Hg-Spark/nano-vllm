from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    half_rotary_dim = cos.shape[-1]
    rotary_dim = half_rotary_dim * 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = torch.chunk(x_rot.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y = torch.cat((y1, y2), dim=-1).to(x.dtype)
    if x_pass.numel():
        y = torch.cat((y, x_pass), dim=-1)
    return y


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        del max_position_embeddings
        assert 0 < rotary_dim <= head_size and rotary_dim % 2 == 0
        self.rotary_dim = rotary_dim
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float)
                / rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _recompose(
        self,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        if self.mrope_section is None:
            return frequencies[0]

        output = frequencies[0].clone()
        for dim, offset in ((1, 1), (2, 2)):
            stop = self.mrope_section[dim] * 3
            output[..., offset:stop:3] = frequencies[
                dim, ..., offset:stop:3
            ]
        return output

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.shape[0] not in (1, 3):
            raise ValueError(
                "RoPE positions must have shape [tokens] or [3, tokens]"
            )
        if positions.shape[0] == 1:
            positions = positions.expand(3, -1)

        frequencies = (
            positions[:, :, None].float()
            * self.inv_freq[None, None, :].float()
        )
        frequencies = self._recompose(frequencies)
        cos = frequencies.cos().unsqueeze(1)
        sin = frequencies.sin().unsqueeze(1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(8)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    mrope_section: tuple[int, int, int] | None = None,
):
    return RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        mrope_section,
    )
