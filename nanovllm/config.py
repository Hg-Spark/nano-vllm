import os
from dataclasses import dataclass, field
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_prefix_cache: bool = True
    hf_config: AutoConfig | None = None
    model_config: AutoConfig | None = field(init=False, default=None)
    is_hybrid: bool = field(init=False, default=False)
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.model_config = getattr(self.hf_config, "text_config", self.hf_config)
        root_type = getattr(self.hf_config, "model_type", "")
        text_type = getattr(self.model_config, "model_type", "")
        self.is_hybrid = root_type.startswith("qwen3_5") or text_type.startswith("qwen3_5")
        if self.is_hybrid:
            # Prefix reuse is unsafe until recurrent/conv states are cached together
            # with the full-attention KV blocks.
            self.enable_prefix_cache = False
        self.max_model_len = min(self.max_model_len, self.model_config.max_position_embeddings)
