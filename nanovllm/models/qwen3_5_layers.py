"""Shared Qwen3.5 dense/MoE normalization, Attention and SwiGLU."""

import torch
from flashinfer.activation import silu_and_mul
from flashinfer.norm import gemma_rmsnorm
from torch import nn

from nanovllm.config import resolve_head_dim, resolve_qwen3_5_rope
from nanovllm.layers.attention import Attention
from nanovllm.layers.rotary_embedding import get_rope


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FlashInfer expects a 2D contiguous input; attention Q/K tensors can be strided 3D views.
        flat = x.reshape(-1, x.size(-1)).contiguous()
        return gemma_rmsnorm(flat, self.weight, eps=self.eps).reshape(x.shape)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = resolve_head_dim(config)
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

        rotary_dim, rope_theta, mrope_section = resolve_qwen3_5_rope(config)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            base=rope_theta,
            mrope_section=mrope_section,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
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
        q, k = self.rotary_emb(positions, q, k)

        output = self.attn(q, k, v).flatten(1)
        output = output * torch.sigmoid(gate.flatten(1))
        return self.o_proj(output)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config, intermediate_size: int) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported Qwen3.5 activation: {config.hidden_act}")
        self.gate_proj = nn.Linear(
            config.hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def pack_projections(self) -> None:
        # Called once after strict checkpoint loading has completed.
        packed = nn.Linear(
            self.gate_proj.in_features,
            self.gate_proj.out_features + self.up_proj.out_features,
            bias=False,
            device=self.gate_proj.weight.device,
            dtype=self.gate_proj.weight.dtype,
        )
        with torch.no_grad():
            gate_rows = self.gate_proj.weight.size(0)
            packed.weight[:gate_rows].copy_(self.gate_proj.weight)
            packed.weight[gate_rows:].copy_(self.up_proj.weight)
        self.gate_up_proj = packed
        del self.gate_proj, self.up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_and_mul(self.gate_up_proj(x)))
