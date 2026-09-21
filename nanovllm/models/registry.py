from torch import nn

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
    "Qwen3_5ForCausalLM": Qwen3_5ForCausalLM,
    "Qwen3_5ForConditionalGeneration": Qwen3_5ForCausalLM,
    "qwen3_5": Qwen3_5ForCausalLM,
    "qwen3_5_text": Qwen3_5ForCausalLM,
}


def _iter_model_configs(hf_config):
    yield hf_config
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and text_config is not hf_config:
        yield text_config


def get_model_class(hf_config) -> type[nn.Module]:
    """Resolve a nano-vLLM text implementation from HF architecture metadata."""
    checked = []
    for config in _iter_model_configs(hf_config):
        architectures = getattr(config, "architectures", None) or ()
        model_type = getattr(config, "model_type", None)
        checked.append((list(architectures), model_type))

        for architecture in architectures:
            model_cls = _MODEL_REGISTRY.get(architecture)
            if model_cls is not None:
                return model_cls

        model_cls = _MODEL_REGISTRY.get(model_type)
        if model_cls is not None:
            return model_cls

    raise ValueError(
        "unsupported model architecture: "
        f"checked={checked}"
    )
