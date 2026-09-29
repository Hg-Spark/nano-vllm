import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


_FLASHINFER_PAGE_SIZES = (16, 32, 64, 128)


def _validate_qwen35_rope_support(
    root_config: PretrainedConfig,
    text_config: PretrainedConfig,
) -> None:
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = {}
    if not isinstance(rope_parameters, Mapping):
        raise ValueError("Qwen3.5-MoE rope_parameters must be a mapping")

    rope_type = rope_parameters.get("rope_type", "default")
    if rope_type != "default":
        raise NotImplementedError(
            "nano-vLLM currently supports Qwen3.5-MoE default RoPE only; "
            f"found rope_type={rope_type!r}"
        )
    if rope_parameters.get("mrope_interleaved", True) is not True:
        raise NotImplementedError(
            "Qwen3.5-MoE requires interleaved mRoPE in nano-vLLM"
        )

    section = rope_parameters.get("mrope_section")
    if section is None:
        if getattr(root_config, "vision_config", None) is not None:
            raise ValueError(
                "multimodal Qwen3.5-MoE requires rope_parameters.mrope_section"
            )
        return
    if len(section) != 3:
        raise ValueError("Qwen3.5 mRoPE requires three sections")

    head_dim = int(
        getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
    )
    partial_rotary_factor = float(
        rope_parameters.get(
            "partial_rotary_factor",
            getattr(text_config, "partial_rotary_factor", 0.25),
        )
    )
    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("Qwen3.5 rotary dimension must be positive and even")
    section = tuple(int(value) for value in section)
    if any(value < 0 for value in section):
        raise ValueError("Qwen3.5 mRoPE sections must be non-negative")
    if sum(section) != rotary_dim // 2:
        raise ValueError(
            "Qwen3.5 mRoPE sections must cover half the rotary dimension: "
            f"sections={section}, rotary_dim={rotary_dim}"
        )


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 4
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 16
    max_prefix_cache_entries: int = 0
    kv_cache_dtype: str = "auto"
    kv_cache_k_scale: float = 1.0
    kv_cache_v_scale: float = 1.0
    decode_graph_batch_sizes: tuple[int, ...] = ()

    # Derived model/runtime metadata. These are populated during startup and
    # are intentionally not constructor knobs.
    root_config: PretrainedConfig = field(init=False, repr=False)
    text_config: PretrainedConfig = field(init=False, repr=False)
    eos_token_ids: tuple[int, ...] = field(
        init=False,
        default=(),
        repr=False,
    )

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"model path does not exist: {self.model}")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        graph_sizes = tuple(
            int(size) for size in self.decode_graph_batch_sizes
        )
        if len(graph_sizes) != len(set(graph_sizes)):
            raise ValueError(
                "decode_graph_batch_sizes must not contain duplicates"
            )
        if any(
            size <= 0 or size > self.max_num_seqs
            for size in graph_sizes
        ):
            raise ValueError(
                "decode_graph_batch_sizes must be within "
                "[1, max_num_seqs]"
            )
        self.decode_graph_batch_sizes = tuple(sorted(graph_sizes))
        if self.max_prefix_cache_entries < 0:
            raise ValueError(
                "max_prefix_cache_entries must be non-negative"
            )
        if self.kv_cache_dtype not in ("auto", "fp8_e4m3"):
            raise ValueError(
                "kv_cache_dtype must be 'auto' or 'fp8_e4m3'"
            )
        for name, scale in (
            ("kv_cache_k_scale", self.kv_cache_k_scale),
            ("kv_cache_v_scale", self.kv_cache_v_scale),
        ):
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                "gpu_memory_utilization must be in (0, 1]"
            )
        if self.kvcache_block_size not in _FLASHINFER_PAGE_SIZES:
            raise ValueError(
                "kvcache_block_size must be one of "
                f"{_FLASHINFER_PAGE_SIZES}"
            )
        hf_config = AutoConfig.from_pretrained(self.model)
        self.root_config = hf_config
        self.text_config = getattr(
            hf_config,
            "text_config",
            hf_config,
        )

        root_type = getattr(hf_config, "model_type", None)
        text_type = getattr(self.text_config, "model_type", None)
        if (
            root_type != "qwen3_5_moe"
            and text_type != "qwen3_5_moe_text"
        ):
            raise ValueError(
                "only Qwen3.5-MoE checkpoints are supported: "
                f"root={root_type!r}, text={text_type!r}"
            )

        if getattr(hf_config, "quantization_config", None):
            raise NotImplementedError(
                "quantized Qwen3.5-MoE checkpoints are not supported"
            )

        required = (
            "layer_types",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "linear_num_value_heads",
            "linear_num_key_heads",
        )
        missing = [
            name
            for name in required
            if not hasattr(self.text_config, name)
        ]
        if missing:
            raise ValueError(
                "invalid Qwen3.5-MoE text config, missing: "
                + ", ".join(missing)
            )

        if self.text_config.num_experts_per_tok > self.text_config.num_experts:
            raise ValueError("num_experts_per_tok exceeds num_experts")

        _validate_qwen35_rope_support(
            self.root_config,
            self.text_config,
        )

        self.max_model_len = min(
            self.max_model_len,
            self.text_config.max_position_embeddings,
        )
