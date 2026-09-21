import os
from dataclasses import dataclass

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
    hf_config: PretrainedConfig | None = None
    text_config: PretrainedConfig | None = None
    eos_token_ids: tuple[int, ...] = ()
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    max_num_state_slots: int | None = None

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"model path does not exist: {self.model}")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
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
        if self.max_num_state_slots is None:
            self.max_num_state_slots = min(self.max_num_seqs, 4)
        if self.max_num_state_slots <= 0:
            raise ValueError("max_num_state_slots must be positive")

        active_limit = min(
            self.max_num_seqs,
            self.max_num_state_slots,
        )
        self.max_num_seqs = active_limit
        self.max_num_state_slots = active_limit
