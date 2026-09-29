import torch
import torch.nn.functional as F
from flashinfer.fused_moe import cutlass_fused_moe
from torch import nn
from torch.profiler import record_function
from transformers import AutoModel

from nanovllm.layers.attention import Attention
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.gdn_state import GDNStatePool
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import get_context


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

        mrope_section = None
        if isinstance(rope_parameters, dict):
            section = rope_parameters.get("mrope_section")
            if section is not None:
                if len(section) != 3:
                    raise ValueError("Qwen3.5 mRoPE requires three sections")
                mrope_section = tuple(int(value) for value in section)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            mrope_section=mrope_section,
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
    """Packed routed experts with a readable reference fallback.

    Qwen3.5-MoE checkpoints store packed rows as [gate, up]. The loader
    converts them once on CPU to FlashInfer/CUTLASS runtime order [up, gate];
    both the fused decode path and the reference fallback use that layout.
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
        self._moe_workspace: torch.Tensor | None = None
        self._moe_output: torch.Tensor | None = None

    def bind_runtime_buffers(
        self,
        workspace: torch.Tensor | None,
        output: torch.Tensor | None,
    ) -> None:
        self._moe_workspace = workspace
        self._moe_output = output

    def _can_use_fused_decode(
        self,
        hidden_states: torch.Tensor,
    ) -> bool:
        return (
            hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and self._moe_workspace is not None
            and self._moe_output is not None
            and hidden_states.size(0) <= self._moe_output.size(0)
        )

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)

        # Correctness-first prefill reference. Decode uses the fixed-shape
        # device path below so runtime expert IDs never reach Python.
        active_experts = torch.unique(selected_experts).tolist()
        for expert_idx in active_experts:
            token_idx, topk_pos = torch.where(
                selected_experts == expert_idx
            )
            current = hidden_states[token_idx]
            up, gate = F.linear(
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

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._can_use_fused_decode(hidden_states):
            output = self._moe_output[: hidden_states.size(0)]
            result = cutlass_fused_moe(
                hidden_states,
                selected_experts.to(torch.int32),
                routing_weights.float(),
                self.gate_up_proj,
                self.down_proj,
                hidden_states.dtype,
                quant_scales=[],
                output=output,
                use_fused_finalize=False,
                tune_max_num_tokens=self._moe_output.size(0),
                enable_pdl=False,
                workspace_buffer=self._moe_workspace,
            )
            if isinstance(result, (list, tuple)):
                return result[0]
            return result

        # Correctness fallback for unsupported devices. Runtime weights use
        # FlashInfer's [up, gate] packed-row convention.
        gate_up_weight = self.gate_up_proj[selected_experts]
        projected = torch.einsum(
            "bh,bkdh->bkd",
            hidden_states,
            gate_up_weight,
        )
        up, gate = projected.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up

        down_weight = self.down_proj[selected_experts]
        expert_output = torch.einsum(
            "bki,bkhi->bkh",
            intermediate,
            down_weight,
        )
        expert_output = expert_output * routing_weights[
            ..., None
        ].to(expert_output.dtype)
        return expert_output.sum(dim=1).to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if get_context().is_prefill:
            return self._forward_prefill(
                hidden_states,
                selected_experts,
                routing_weights,
            )
        return self._forward_decode(
            hidden_states,
            selected_experts,
            routing_weights,
        )


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
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "specify exactly one of input_ids or inputs_embeds"
            )
        hidden_states = (
            self.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5MoeForCausalLM(nn.Module):
    # Checkpoint naming belongs to the model adapter. The generic loader only
    # applies these declarative rules and validates names/shapes strictly.
    checkpoint_prefix_map = (
        ("model.language_model.", "model."),
        ("model.visual.", "visual."),
    )
    checkpoint_skip_prefixes = (
        "mtp.",
    )

    def __init__(self, config) -> None:
        super().__init__()
        text_config = getattr(config, "text_config", config)
        vision_config = getattr(config, "vision_config", None)
        self.root_config = config
        self.visual = (
            AutoModel.from_config(vision_config)
            if vision_config is not None
            else None
        )
        self.model = Qwen3_5MoeModel(text_config)
        self.lm_head = nn.Linear(
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
        )

    @staticmethod
    def transform_checkpoint_weight(
        weight_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if not weight_name.endswith(".mlp.experts.gate_up_proj"):
            return weight
        if weight.size(1) % 2:
            raise ValueError("packed expert gate/up rows must be even")
        split = weight.size(1) // 2
        gate = weight[:, :split].clone()
        weight[:, :split].copy_(weight[:, split:])
        weight[:, split:].copy_(gate)
        return weight

    def bind_moe_runtime(
        self,
        workspace: torch.Tensor | None,
        output: torch.Tensor | None,
    ) -> None:
        for layer in self.model.layers:
            layer.mlp.experts.bind_runtime_buffers(
                workspace,
                output,
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

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if self.visual is None:
            raise RuntimeError("checkpoint has no visual tower")
        visual_dtype = next(self.visual.parameters()).dtype
        output = self.visual(
            pixel_values.to(dtype=visual_dtype),
            grid_thw=image_grid_thw,
            return_dict=True,
        )
        features = output.pooler_output
        if isinstance(features, (tuple, list)):
            if len(features) != 1:
                raise RuntimeError(
                    "ImagePrompt requires exactly one visual feature group"
                )
            features = features[0]

        spatial_merge_size = int(
            self.root_config.vision_config.spatial_merge_size
        )
        expected = int(
            image_grid_thw[0].prod().item()
            // (spatial_merge_size ** 2)
        )
        if features.ndim != 2 or features.shape[0] != expected:
            raise RuntimeError(
                "visual feature count does not match image grid: "
                f"features={tuple(features.shape)}, expected={expected}"
            )
        return features

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sequence_indices: list[int] | None = None,
    ) -> torch.Tensor:
        from nanovllm.utils.context import get_context

        context = get_context()
        if context.is_prefill:
            last_indices = context.qo_indptr[1:] - 1
            if sequence_indices is not None:
                last_indices = last_indices[sequence_indices]
            hidden_states = hidden_states[last_indices].contiguous()
        elif sequence_indices is not None:
            hidden_states = hidden_states[sequence_indices]
        return self.lm_head(hidden_states)
