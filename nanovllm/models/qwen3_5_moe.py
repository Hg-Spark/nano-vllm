import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function

from nanovllm.layers.attention import Attention
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.gdn_state import GDNStatePool
from nanovllm.layers.rotary_embedding import get_rope


class Qwen3_5MoeRMSNorm(nn.Module):

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


class Qwen3_5MoeAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
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

        self.q_norm = Qwen3_5MoeRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3_5MoeRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        rope_parameters = getattr(config, "rope_parameters", None)
        partial_rotary_factor = getattr(
            config,
            "partial_rotary_factor",
            None,
        )
        rope_theta = getattr(config, "rope_theta", 10000.0)
        if isinstance(rope_parameters, dict):
            partial_rotary_factor = rope_parameters.get(
                "partial_rotary_factor",
                partial_rotary_factor,
            )
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
        if partial_rotary_factor is None:
            partial_rotary_factor = 0.25
        rotary_dim = int(self.head_dim * partial_rotary_factor)

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


class Qwen3_5MoeMLP(nn.Module):

    def __init__(self, config, intermediate_size: int) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"unsupported Qwen3.5-MoE activation: {config.hidden_act}"
            )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class Qwen3_5MoeExperts(nn.Module):
    """Correctness-first packed expert implementation.

    Official Qwen3.5-MoE checkpoints already store routed expert weights as
    packed 3D tensors, so the reference path keeps that layout and performs
    explicit per-expert dispatch.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size,
        ))
        self.down_proj = nn.Parameter(torch.empty(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        ))

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)

        # This is intentionally an eager numerical reference, not a fused MoE
        # kernel. Only experts selected by at least one token are executed.
        active_experts = torch.unique(selected_experts).tolist()
        for expert_idx in active_experts:
            token_idx, topk_pos = torch.where(
                selected_experts == expert_idx
            )
            current = hidden_states[token_idx]
            gate, up = F.linear(
                current,
                self.gate_up_proj[expert_idx],
            ).chunk(2, dim=-1)
            current = F.silu(gate) * up
            current = F.linear(
                current,
                self.down_proj[expert_idx],
            )
            current = current * routing_weights[
                token_idx,
                topk_pos,
                None,
            ].to(current.dtype)
            output.index_add_(
                0,
                token_idx,
                current.to(output.dtype),
            )

        return output


class Qwen3_5MoeTopKRouter(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(
            self.num_experts,
            self.hidden_size,
        ))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(
            router_logits,
            dtype=torch.float32,
            dim=-1,
        )
        routing_weights, selected_experts = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1,
            keepdim=True,
        )
        return (
            routing_weights.to(router_logits.dtype),
            selected_experts,
        )


class Qwen3_5MoeSparseMoeBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MoeMLP(
            config,
            config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)

        shared_output = self.shared_expert(hidden_states)
        shared_output = (
            torch.sigmoid(self.shared_expert_gate(hidden_states))
            * shared_output
        )

        routing_weights, selected_experts = self.gate(hidden_states)
        routed_output = self.experts(
            hidden_states,
            selected_experts,
            routing_weights,
        )
        return (routed_output + shared_output).reshape(original_shape)


class Qwen3_5MoeDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config)
        else:
            raise ValueError(
                f"unsupported Qwen3.5-MoE layer type: {self.block_type}"
            )

        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
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
            with record_function("nanovllm::gdn_layer"):
                hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(
                positions,
                hidden_states,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        with record_function("nanovllm::moe_layer"):
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5MoeModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen3_5MoeDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3_5MoeRMSNorm(
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


class Qwen3_5MoeForCausalLM(nn.Module):
    # Checkpoint naming belongs to the model adapter. The generic loader only
    # applies these declarative rules and validates names/shapes strictly.
    checkpoint_prefix_map = (
        ("model.language_model.", "model."),
    )
    checkpoint_skip_prefixes = (
        "model.visual.",
        "mtp.",
    )

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3_5MoeModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def kv_cache_modules(self) -> list[Attention]:
        return [
            layer.self_attn.attn
            for layer in self.model.layers
            if layer.block_type == "full_attention"
        ]

    def state_cache_modules(self) -> list[GDNStatePool]:
        return [
            layer.linear_attn.state_pool
            for layer in self.model.layers
            if layer.block_type == "linear_attention"
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sequence_indices: list[int] | None = None,
    ) -> torch.Tensor:
        from nanovllm.utils.context import get_context

        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            if sequence_indices is not None:
                last_indices = last_indices[sequence_indices]
            hidden_states = hidden_states[last_indices].contiguous()
        elif sequence_indices is not None:
            hidden_states = hidden_states[sequence_indices]
        return self.lm_head(hidden_states)
