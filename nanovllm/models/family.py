"""Model family check shared by Config and verification without GPU imports."""

_TEXT_FAMILIES = {"qwen3_5_text": "qwen3_5", "qwen3_5_moe_text": "qwen3_5_moe"}


def is_qwen3_5_moe(config) -> bool:
    text_config = getattr(config, "text_config", config)
    root_type = getattr(config, "model_type", None)
    text_type = getattr(text_config, "model_type", None)
    if text_type not in _TEXT_FAMILIES or root_type not in (
        text_type,
        _TEXT_FAMILIES.get(text_type),
    ):
        raise ValueError(
            "only Qwen3.5 dense or MoE checkpoints are supported: "
            f"root={root_type!r}, text={text_type!r}"
        )
    return text_type == "qwen3_5_moe_text"
