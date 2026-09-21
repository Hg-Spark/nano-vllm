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
