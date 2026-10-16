import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        base: float,
        mrope_section: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        assert 0 < rotary_dim <= head_size and rotary_dim % 2 == 0
        self.head_size = head_size
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def prepare(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # One row per packed token, shared by all attention layers in the batch.
        frequencies = positions[:, :, None].float() * self.inv_freq[None, None, :].float()
        if self.mrope_section is None:
            frequencies = frequencies[0]
        else:
            frequencies = frequencies[0].clone()
            for axis in (1, 2):
                stop = self.mrope_section[axis] * 3
                frequencies[..., axis:stop:3] = (
                    positions[axis, :, None].float() * self.inv_freq[None, :].float()
                )[..., axis:stop:3]
        cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).contiguous()
        row_ids = torch.arange(positions.size(1), dtype=torch.int32, device=positions.device)
        return row_ids, cache

    @staticmethod
    def apply_cached(
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        prepared: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.rope import apply_rope_with_cos_sin_cache

        row_ids, cache = prepared
        return apply_rope_with_cos_sin_cache(
            row_ids, query.contiguous(), key.contiguous(), head_size, cache
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply_cached(query, key, self.head_size, self.prepare(positions))
