import torch

from nanovllm.layers.rotary_embedding import RotaryEmbedding


def test_partial_rope_rotates_prefix_and_preserves_tail():
    head_dim = 8
    rotary_dim = 4
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=16,
        base=10000,
    )

    q = torch.randn(3, 2, head_dim)
    k = torch.randn(3, 2, head_dim)
    positions = torch.tensor([1, 2, 3])

    q_out, k_out = rope(positions, q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    torch.testing.assert_close(q_out[..., rotary_dim:], q[..., rotary_dim:])
    torch.testing.assert_close(k_out[..., rotary_dim:], k[..., rotary_dim:])
    assert not torch.equal(q_out[..., :rotary_dim], q[..., :rotary_dim])


def test_three_axis_text_positions_match_scalar_positions():
    rope = RotaryEmbedding(
        head_size=12,
        rotary_dim=12,
        max_position_embeddings=16,
        base=10000,
        mrope_section=(2, 2, 2),
    )
    q = torch.randn(2, 1, 12)
    k = torch.randn(2, 1, 12)
    scalar = torch.tensor([2, 3])
    axial = scalar.unsqueeze(0).expand(3, -1)

    q_scalar, k_scalar = rope(scalar, q, k)
    q_axial, k_axial = rope(axial, q, k)

    torch.testing.assert_close(q_scalar, q_axial)
    torch.testing.assert_close(k_scalar, k_axial)


def test_mrope_uses_spatial_axes():
    rope = RotaryEmbedding(
        head_size=12,
        rotary_dim=12,
        max_position_embeddings=16,
        base=10000,
        mrope_section=(2, 2, 2),
    )
    q = torch.randn(1, 1, 12)
    k = torch.randn(1, 1, 12)
    base_positions = torch.tensor([[2], [2], [2]])
    changed_height = torch.tensor([[2], [5], [2]])

    q_base, _ = rope(base_positions, q, k)
    q_height, _ = rope(changed_height, q, k)

    assert not torch.equal(q_base, q_height)
