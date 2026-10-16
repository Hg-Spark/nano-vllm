"""Common Qwen3.5 hybrid decoder and dense/MoE causal language model."""

import torch
from torch import nn
from torch.profiler import record_function
from transformers import AutoModel

from nanovllm.config import resolve_head_dim, resolve_qwen3_5_rope
from nanovllm.layers.attention import Attention
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.gdn_state import GDNStatePool
from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.models.qwen3_5_layers import Qwen3_5Attention, Qwen3_5MLP, Qwen3_5RMSNorm
from nanovllm.models.qwen3_5_moe import Qwen3_5SparseMoeBlock


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, head_dim: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, head_dim)
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {self.block_type}")

        self.is_moe = config.model_type == "qwen3_5_moe_text"
        self.mlp = (
            Qwen3_5SparseMoeBlock(config)
            if self.is_moe
            else Qwen3_5MLP(config, config.intermediate_size)
        )
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
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm.forward_with_residual(hidden_states, residual)

        if self.block_type == "linear_attention":
            with record_function("nanovllm::gdn_layer"):
                hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, rope_cache)
        hidden_states = self.post_attention_layernorm.forward_with_residual(hidden_states, residual)
        with record_function("nanovllm::moe_layer" if self.is_moe else "nanovllm::ffn_layer"):
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        head_dim = resolve_head_dim(config)
        rotary_dim, rope_theta, mrope_section = resolve_qwen3_5_rope(config)
        self.rotary_emb = RotaryEmbedding(
            head_dim, rotary_dim=rotary_dim, base=rope_theta, mrope_section=mrope_section
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx, head_dim)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(
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
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        rope_cache = self.rotary_emb.prepare(positions)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, rope_cache)
        return self.norm.forward_with_residual(hidden_states, residual)


class Qwen3_5ForCausalLM(nn.Module):
    # Checkpoint naming belongs to the model adapter. The generic loader only
    # applies these declarative rules and validates names/shapes strictly.
    checkpoint_prefix_map = (
        ("model.language_model.", "model."),
        ("model.visual.", "visual."),
    )
    checkpoint_skip_prefixes = ("mtp.",)

    def __init__(self, config, *, load_vision: bool = True) -> None:
        super().__init__()
        text_config = getattr(config, "text_config", config)
        vision_config = getattr(config, "vision_config", None)
        self.root_config = config
        self.visual = (
            AutoModel.from_config(vision_config)
            if load_vision and vision_config is not None
            else None
        )
        if not load_vision:
            self.checkpoint_skip_prefixes = (*self.checkpoint_skip_prefixes, "model.visual.")
        self.model = Qwen3_5Model(text_config)
        self.lm_head = nn.Linear(
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
        )
        if getattr(text_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def packed_checkpoint_shards(self) -> dict[str, tuple[str, int, int]]:
        """Map HF split weights to row slices of the final packed parameters."""
        shards = {}
        for index, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{index}"
            mlp_name = f"{prefix}.mlp"
            mlp = layer.mlp
            if layer.is_moe:
                mlp_name += ".shared_expert"
                mlp = mlp.shared_expert
            half = mlp.gate_up_proj.weight.size(0) // 2
            target = f"{mlp_name}.gate_up_proj.weight"
            shards[f"{mlp_name}.gate_proj.weight"] = (target, 0, half)
            shards[f"{mlp_name}.up_proj.weight"] = (target, half, 2 * half)

            if layer.block_type == "linear_attention":
                gdn = layer.linear_attn
                source = f"{prefix}.linear_attn"
                target = f"{source}.in_proj.weight"
                widths = (
                    ("in_proj_qkv", gdn.conv_dim),
                    ("in_proj_z", gdn.value_dim),
                    ("in_proj_b", gdn.num_v_heads),
                    ("in_proj_a", gdn.num_v_heads),
                )
                offset = 0
                for name, width in widths:
                    shards[f"{source}.{name}.weight"] = (target, offset, offset + width)
                    offset += width
        return shards

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
                raise RuntimeError("ImagePrompt requires exactly one visual feature group")
            features = features[0]

        spatial_merge_size = int(self.root_config.vision_config.spatial_merge_size)
        expected = int(image_grid_thw[0].prod().item() // (spatial_merge_size**2))
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
