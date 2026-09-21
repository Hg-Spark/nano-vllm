import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.gated_delta import GatedRMSNorm, causal_conv1d_stateful, recurrent_gated_delta
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import get_context


class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5 zero-centered RMSNorm: scale is (1 + weight)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * (1.0 + self.weight.float())).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm(x)
        residual = (x.float() + residual.float()).to(x.dtype)
        return self._norm(residual), residual


class Qwen3_5Attention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim * 2,
            bias=attention_bias,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000000))
        partial_factor = rope_parameters.get(
            "partial_rotary_factor",
            getattr(config, "partial_rotary_factor", 1.0),
        )
        rotary_dim = int(self.head_dim * partial_factor)
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

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q_and_gate = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim * 2)
        q, gate = q_and_gate.split(self.head_dim, dim=-1)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)
        output = output * torch.sigmoid(gate)
        return self.o_proj(output.flatten(1, -1))


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act
        assert self.activation == "silu"
        assert self.num_v_heads % self.num_k_heads == 0

        # Stage 1 favors correctness/readability. GDN weights are replicated under
        # tensor parallelism; a later optimization can shard them by head.
        self.in_proj_qkv = ReplicatedLinear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = ReplicatedLinear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = ReplicatedLinear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = ReplicatedLinear(config.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = GatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = ReplicatedLinear(self.value_dim, config.hidden_size, bias=False)
        self.state_manager = None

    def _forward_sequence(self, hidden_states: torch.Tensor, seq_id: int) -> torch.Tensor:
        if self.state_manager is None:
            raise RuntimeError("GDN state manager has not been attached")

        state = self.state_manager.get(seq_id, self.layer_idx)
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv, next_conv_state = causal_conv1d_stateful(
            mixed_qkv,
            state.conv_state,
            self.conv1d.weight,
        )
        state.conv_state.copy_(next_conv_state.to(state.conv_state.dtype))

        query, key, value = mixed_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        token_count = hidden_states.shape[0]
        query = query.view(token_count, self.num_k_heads, self.head_k_dim)
        key = key.view(token_count, self.num_k_heads, self.head_k_dim)
        value = value.view(token_count, self.num_v_heads, self.head_v_dim)
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=1)
            key = key.repeat_interleave(repeat, dim=1)

        z = self.in_proj_z(hidden_states).view(token_count, self.num_v_heads, self.head_v_dim)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        a = self.in_proj_a(hidden_states)
        decay_log = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        output, next_recurrent_state = recurrent_gated_delta(
            query,
            key,
            value,
            decay_log,
            beta,
            state.recurrent_state,
        )
        state.recurrent_state.copy_(next_recurrent_state.to(state.recurrent_state.dtype))
        output = self.norm(output, z)
        return self.out_proj(output.reshape(token_count, self.value_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if not context.seq_ids or not context.seq_lens:
            raise RuntimeError("hybrid context is missing seq_ids/seq_lens")
        if sum(context.seq_lens) != hidden_states.shape[0]:
            raise RuntimeError("packed GDN token count does not match sequence metadata")

        outputs = []
        start = 0
        for seq_id, seq_len in zip(context.seq_ids, context.seq_lens):
            end = start + seq_len
            outputs.append(self._forward_sequence(hidden_states[start:end], seq_id))
            start = end
        return torch.cat(outputs, dim=0)


class Qwen3_5MLP(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config)
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {self.block_type}")

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5TextModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5TextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    @staticmethod
    def map_weight_name(weight_name: str) -> str | None:
        if (
            weight_name.startswith("model.visual.")
            or weight_name.startswith("visual.")
            or ".mtp." in weight_name
            or weight_name.startswith("mtp.")
        ):
            return None
        if weight_name.startswith("model.language_model."):
            return "model." + weight_name[len("model.language_model."):]
        if weight_name.startswith("language_model."):
            return "model." + weight_name[len("language_model."):]
        return weight_name

    @property
    def num_full_attention_layers(self) -> int:
        return sum(layer.block_type == "full_attention" for layer in self.model.layers)

    def set_state_manager(self, state_manager) -> None:
        for layer in self.model.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn.state_manager = state_manager

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
