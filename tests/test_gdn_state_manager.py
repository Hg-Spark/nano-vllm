from types import SimpleNamespace
import torch

from nanovllm.engine.gdn_state import GDNStateManager


def make_config():
    return SimpleNamespace(
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=5,
        linear_conv_kernel_dim=4,
        dtype=torch.float32,
        mamba_ssm_dtype="float32",
    )


def test_state_isolated_reset_and_release():
    manager = GDNStateManager(make_config(), device="cpu")

    a = manager.get(10, 0)
    b = manager.get(11, 0)
    a.conv_state.fill_(7)
    a.recurrent_state.fill_(3)

    assert torch.count_nonzero(b.conv_state) == 0
    assert torch.count_nonzero(b.recurrent_state) == 0

    manager.reset_sequence(10)
    a2 = manager.get(10, 0)
    assert torch.count_nonzero(a2.conv_state) == 0
    assert torch.count_nonzero(a2.recurrent_state) == 0

    manager.release_sequences([10, 11])
    assert manager._states == {}


def test_bytes_per_sequence_counts_only_linear_layers():
    config = make_config()
    manager = GDNStateManager(config, device="cpu")

    conv_elems = (2 * (2 * 3) + 4 * 5) * (4 - 1)
    recurrent_elems = 4 * 3 * 5
    expected_one_layer = (conv_elems + recurrent_elems) * 4
    assert manager.bytes_per_sequence == 2 * expected_one_layer
