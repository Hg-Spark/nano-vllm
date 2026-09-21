from torch import nn

from nanovllm.models.qwen3_5_moe import Qwen3_5MoeForCausalLM


_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "qwen3_5_moe_text": Qwen3_5MoeForCausalLM,
}


def get_model_class(config) -> type[nn.Module]:
    text_config = getattr(config, "text_config", config)
    model_type = getattr(text_config, "model_type", None)
    model_class = _MODEL_CLASSES.get(model_type)
    if model_class is None:
        raise ValueError(
            "unsupported model type: "
            f"{model_type!r}; only Qwen3.5-MoE text is implemented"
        )
    return model_class


def create_model(config) -> nn.Module:
    text_config = getattr(config, "text_config", config)
    return get_model_class(text_config)(text_config)
