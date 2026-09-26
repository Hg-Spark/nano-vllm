import math
import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 4
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    kvcache_block_size: int = 256
    max_prefix_cache_entries: int = 16
    kv_cache_dtype: str = "auto"
    kv_cache_k_scale: float = 1.0
    kv_cache_v_scale: float = 1.0

    # Derived model/runtime metadata. These are populated during startup and
    # are intentionally not constructor knobs.
    hf_config: PretrainedConfig = field(init=False, repr=False)
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
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256")
        if self.tensor_parallel_size != 1:
            raise NotImplementedError(
                "Qwen3.5-MoE reference runtime supports TP=1 only"
            )
        if not self.enforce_eager:
            raise NotImplementedError(
                "Qwen3.5-MoE reference runtime is eager-only"
            )

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.text_config = getattr(
            self.hf_config,
            "text_config",
            self.hf_config,
        )

        root_type = getattr(self.hf_config, "model_type", None)
        text_type = getattr(self.text_config, "model_type", None)
        if (
            root_type != "qwen3_5_moe"
            and text_type != "qwen3_5_moe_text"
        ):
            raise ValueError(
                "only Qwen3.5-MoE checkpoints are supported: "
                f"root={root_type!r}, text={text_type!r}"
            )

        if getattr(self.hf_config, "quantization_config", None):
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

        self.max_model_len = min(
            self.max_model_len,
            self.text_config.max_position_embeddings,
        )
