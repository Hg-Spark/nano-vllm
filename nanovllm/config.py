from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nanovllm.models.family import is_qwen3_5_moe

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_FLASHINFER_PAGE_SIZES = (16, 32, 64, 128)


def _load_hf_config(model_path: str) -> PretrainedConfig:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


def resolve_head_dim(text_config: PretrainedConfig) -> int:
    explicit = getattr(text_config, "head_dim", None)
    if explicit is not None:
        return int(explicit)
    return text_config.hidden_size // text_config.num_attention_heads


def resolve_qwen3_5_rope(text_config: PretrainedConfig):
    """Resolve the supported Qwen3.5 RoPE settings once for validation and construction."""
    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    if not isinstance(rope_parameters, Mapping):
        raise ValueError("Qwen3.5 rope_parameters must be a mapping")
    rope_type = rope_parameters.get("rope_type", "default")
    if rope_type != "default":
        raise NotImplementedError(
            f"nano-vLLM currently supports Qwen3.5 default RoPE only; found rope_type={rope_type!r}"
        )
    if rope_parameters.get("mrope_interleaved", True) is not True:
        raise NotImplementedError("Qwen3.5 requires interleaved mRoPE in nano-vLLM")

    head_dim = resolve_head_dim(text_config)
    factor = rope_parameters.get(
        "partial_rotary_factor", getattr(text_config, "partial_rotary_factor", None)
    )
    rotary_dim = int(head_dim * float(0.25 if factor is None else factor))
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("Qwen3.5 rotary dimension must be positive, even and <= head_dim")
    section = rope_parameters.get("mrope_section")
    if section is not None:
        if len(section) != 3:
            raise ValueError("Qwen3.5 mRoPE requires three sections")
        section = tuple(int(value) for value in section)
        if any(value < 0 for value in section) or sum(section) != rotary_dim // 2:
            raise ValueError(
                "Qwen3.5 mRoPE sections must non-negatively cover half rotary_dim: "
                f"sections={section}, rotary_dim={rotary_dim}"
            )
    return (
        rotary_dim,
        float(rope_parameters.get("rope_theta", getattr(text_config, "rope_theta", 10000.0))),
        section,
    )


def _validate_qwen3_5_rope_support(
    root_config: PretrainedConfig, text_config: PretrainedConfig
) -> None:
    _, _, section = resolve_qwen3_5_rope(text_config)
    if section is None and getattr(root_config, "vision_config", None) is not None:
        raise ValueError("multimodal Qwen3.5 requires rope_parameters.mrope_section")


@dataclass(slots=True)
class Config:
    model: str
    dtype: str
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 4
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    kv_cache_block_size: int = 16
    max_prefix_cache_entries: int = 0
    prefix_cache_eager_snapshot: bool = True
    max_visual_tokens_per_request: int | None = 0
    kv_cache_dtype: str = "auto"
    kv_cache_k_scale: float = 1.0
    kv_cache_v_scale: float = 1.0
    decode_graph_batch_sizes: tuple[int, ...] = ()

    root_config: PretrainedConfig = field(init=False, repr=False)
    text_config: PretrainedConfig = field(init=False, repr=False)
    is_moe: bool = field(init=False)
    eos_token_ids: tuple[int, ...] = field(init=False, default=(), repr=False)
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError("dtype must be 'float16' or 'bfloat16'")
        if not os.path.isdir(self.model):
            raise ValueError(f"model path does not exist: {self.model}")
        if self.max_num_batched_tokens <= 0 or self.max_num_seqs <= 0 or self.max_model_len <= 0:
            raise ValueError(
                "max_num_batched_tokens, max_num_seqs, and max_model_len must be positive"
            )
        graph_sizes = tuple(int(size) for size in self.decode_graph_batch_sizes)
        if len(graph_sizes) != len(set(graph_sizes)):
            raise ValueError("decode_graph_batch_sizes must not contain duplicates")
        if any(size <= 0 or size > self.max_num_seqs for size in graph_sizes):
            raise ValueError("decode_graph_batch_sizes must be within [1, max_num_seqs]")
        self.decode_graph_batch_sizes = tuple(sorted(graph_sizes))
        if self.max_prefix_cache_entries < 0:
            raise ValueError("max_prefix_cache_entries must be non-negative")
        if self.max_visual_tokens_per_request is not None and not (
            0 <= self.max_visual_tokens_per_request <= self.max_model_len
        ):
            raise ValueError("max_visual_tokens_per_request must be within [0, max_model_len]")
        if self.kv_cache_dtype not in ("auto", "fp8_e4m3"):
            raise ValueError("kv_cache_dtype must be 'auto' or 'fp8_e4m3'")
        for name, scale in (
            ("kv_cache_k_scale", self.kv_cache_k_scale),
            ("kv_cache_v_scale", self.kv_cache_v_scale),
        ):
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.kv_cache_block_size not in _FLASHINFER_PAGE_SIZES:
            raise ValueError(f"kv_cache_block_size must be one of {_FLASHINFER_PAGE_SIZES}")

        hf_config = _load_hf_config(self.model)
        self.root_config = hf_config
        self.text_config = getattr(hf_config, "text_config", hf_config)
        self.is_moe = is_qwen3_5_moe(hf_config)
        if getattr(hf_config, "quantization_config", None):
            raise NotImplementedError("quantized Qwen3.5 checkpoints are not supported")

        required = (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "layer_types",
            "linear_num_value_heads",
            "linear_num_key_heads",
        ) + (
            (
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            )
            if self.is_moe
            else ("intermediate_size",)
        )
        missing = [name for name in required if not hasattr(self.text_config, name)]
        if missing:
            raise ValueError("invalid Qwen3.5 text config, missing: " + ", ".join(missing))
        for name in ("hidden_size", "num_attention_heads", "num_key_value_heads"):
            value = getattr(self.text_config, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Qwen3.5 {name} must be a positive integer")
        if self.text_config.num_attention_heads % self.text_config.num_key_value_heads:
            raise ValueError("Qwen3.5 attention heads must be divisible by KV heads")
        if (
            getattr(self.text_config, "head_dim", None) is None
            and self.text_config.hidden_size % self.text_config.num_attention_heads
        ):
            raise ValueError("Qwen3.5 hidden_size must be divisible by attention heads")
        self.head_dim = resolve_head_dim(self.text_config)
        if self.head_dim <= 0:
            raise ValueError("Qwen3.5 head_dim must be positive")
        if self.is_moe and self.text_config.num_experts_per_tok > self.text_config.num_experts:
            raise ValueError("num_experts_per_tok exceeds num_experts")
        if not self.is_moe and self.text_config.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

        _validate_qwen3_5_rope_support(self.root_config, self.text_config)
        self.max_model_len = min(self.max_model_len, self.text_config.max_position_embeddings)
