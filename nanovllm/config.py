import os
from dataclasses import dataclass

from transformers import AutoConfig, PretrainedConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: PretrainedConfig | None = None
    text_config: PretrainedConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    is_hybrid: bool = False
    enable_prefix_cache: bool = True
    max_num_state_slots: int | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.text_config = getattr(self.hf_config, "text_config", self.hf_config)
        self.max_model_len = min(
            self.max_model_len,
            self.text_config.max_position_embeddings,
        )

        layer_types = getattr(self.text_config, "layer_types", None) or ()
        self.is_hybrid = "linear_attention" in layer_types
        if self.is_hybrid:
            # Until KV + recurrent-state prefix snapshots are implemented,
            # reusing KV-only prefixes would produce an inconsistent GDN state.
            self.enable_prefix_cache = False

            # The reference GDN path uses Python-visible state-slot routing and
            # therefore must stay eager until the state-aware graph workspace is
            # consumed by a capture-safe CUDA kernel.
            self.enforce_eager = True

            if self.max_num_state_slots is None:
                self.max_num_state_slots = min(self.max_num_seqs, 32)
            self.max_num_seqs = min(
                self.max_num_seqs,
                self.max_num_state_slots,
            )
        elif self.max_num_state_slots is None:
            self.max_num_state_slots = self.max_num_seqs
