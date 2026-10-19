from types import SimpleNamespace


def make_text_config():
    return SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
        shared_expert_intermediate_size=5,
        layer_types=["linear_attention", "full_attention"],
        num_hidden_layers=2,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        max_position_embeddings=1024,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_bias=False,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
    )


def make_dense_config():
    config = make_text_config()
    config.model_type = "qwen3_5_text"
    config.intermediate_size = 12
    config.tie_word_embeddings = True
    for name in (
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        delattr(config, name)
    return config
