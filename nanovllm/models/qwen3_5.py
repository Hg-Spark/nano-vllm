import torch
import torch.distributed as dist
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope


class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5 zero-centered RMSNorm: scale is (1 + weight)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(
            output.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        output = output * (1.0 + self.weight.float())
        return output.to(x.dtype)


class Qwen3_5Attention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        if dist.get_world_size() != 1:
            raise NotImplementedError(
                "Qwen3.5 correctness path currently requires TP=1"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

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

        partial_rotary_factor = getattr(
            config,
            "partial_rotary_factor",
            0.25,
        )
        rotary_dim = int(self.head_dim * partial_rotary_factor)
        rope_parameters = getattr(config, "rope_parameters", None)
        rope_theta = getattr(config, "rope_theta", 10000.0)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", rope_theta)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
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

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"unsupported Qwen3.5 activation: {config.hidden_act}"
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_up_proj(hidden_states))
        )


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config)
        else:
            raise ValueError(
                f"unsupported Qwen3.5 layer type: {self.block_type}"
            )

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(
                positions,
                hidden_states,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5ForCausalLM(nn.Module):
    # Official multimodal checkpoints store the text tower below
    # model.language_model.*, while nano-vLLM instantiates only that tower.
    weight_name_prefixes = (("model.language_model.", "model."),)
    skip_weight_prefixes = ("model.visual.", "mtp.")

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        if dist.get_world_size() != 1:
            raise NotImplementedError(
                "Qwen3.5 phase-1 text backbone currently requires TP=1"
            )
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
