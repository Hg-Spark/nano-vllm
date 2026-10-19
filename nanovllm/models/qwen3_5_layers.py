"""Shared Qwen3.5 dense/MoE normalization, Attention and SwiGLU."""

import torch
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.rotary_embedding import RotaryEmbedding


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from flashinfer.norm import gemma_rmsnorm

        # FlashInfer expects a 2D contiguous input; attention Q/K tensors can be strided 3D views.
        flat = x.reshape(-1, x.size(-1)).contiguous()
        return gemma_rmsnorm(flat, self.weight, eps=self.eps).reshape(x.shape)

    def forward_with_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        from flashinfer.norm import gemma_fused_add_rmsnorm

        # Gemma-style weights represent (1 + weight); update residual and output in place.
        gemma_fused_add_rmsnorm(x, residual, self.weight, eps=self.eps)
        return x


class Qwen3_5Attention(nn.Module):
    def __init__(self, config, head_dim: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.q_size * 2,
            bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.q_size,
            self.hidden_size,
            bias=attention_bias,
        )

        self.q_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        self.attn = Attention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        q_gate = self.q_proj(hidden_states).view(
            -1,
            self.num_heads,
            self.head_dim * 2,
        )
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k_proj(hidden_states).view(
            -1,
            self.num_kv_heads,
            self.head_dim,
        )
        v = self.v_proj(hidden_states).view(
            -1,
            self.num_kv_heads,
            self.head_dim,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = RotaryEmbedding.apply_cached(q, k, self.head_dim, rope_cache)

        output = self.attn(q, k, v) * torch.sigmoid(gate)
        return self.o_proj(output.flatten(1))


class Qwen3_5MLP(nn.Module):
    def __init__(self, config, intermediate_size: int) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported Qwen3.5 activation: {config.hidden_act}")
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from flashinfer.activation import silu_and_mul

        return self.down_proj(silu_and_mul(self.gate_up_proj(x)))
