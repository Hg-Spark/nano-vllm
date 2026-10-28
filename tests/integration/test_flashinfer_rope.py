import unittest

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")


from nanovllm.layers.rotary_embedding import RotaryEmbedding


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for FlashInfer RoPE")
class RotaryEmbeddingTest(unittest.TestCase):
    def test_partial_rope_rotates_prefix_and_preserves_tail(self):
        head_dim = 64
        rotary_dim = 32
        rope = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=rotary_dim,
            base=10000,
        ).cuda()

        q = torch.randn(3, 2, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(3, 2, head_dim, device="cuda", dtype=torch.bfloat16)
        positions = torch.tensor([[1, 2, 3]] * 3, device="cuda")

        q_out, k_out = rope.apply_cached(q, k, q.size(-1), rope.prepare(positions))

        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)
        torch.testing.assert_close(q_out[..., rotary_dim:], q[..., rotary_dim:])
        torch.testing.assert_close(k_out[..., rotary_dim:], k[..., rotary_dim:])
        frequencies = positions[0, :, None].float() * rope.inv_freq[None, :]
        cos, sin = frequencies.cos()[:, None, :], frequencies.sin()[:, None, :]
        for actual, source in ((q_out, q), (k_out, k)):
            left, right = source[..., :rotary_dim].float().chunk(2, dim=-1)
            expected = torch.cat(
                (left * cos - right * sin, right * cos + left * sin),
                dim=-1,
            ).to(source.dtype)
            torch.testing.assert_close(actual[..., :rotary_dim], expected, atol=2e-2, rtol=2e-2)

    def test_flashinfer_mrope_matches_reference(self):
        rope = RotaryEmbedding(64, 64, 10000, mrope_section=(11, 11, 10)).cuda()
        torch.manual_seed(5)
        q = torch.randn(3, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(3, 1, 64, device="cuda", dtype=torch.bfloat16)
        positions = torch.tensor([[1, 2, 3], [4, 6, 8], [7, 9, 11]], device="cuda")
        frequencies = positions[:, :, None].float() * rope.inv_freq[None, None, :]
        selected = frequencies[0].clone()
        for axis, count in ((1, 11), (2, 10)):
            selected[..., axis : count * 3 : 3] = frequencies[axis, ..., axis : count * 3 : 3]
        cos = selected.cos()[:, None, :]
        sin = selected.sin()[:, None, :]

        def reference(x):
            a, b = x.float().chunk(2, dim=-1)
            return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)

        actual_q, actual_k = rope.apply_cached(q, k, q.size(-1), rope.prepare(positions))
        torch.testing.assert_close(actual_q, reference(q), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_k, reference(k), atol=2e-2, rtol=2e-2)

        # Text-only Decode uses a single RoPE axis; repeated axes must be equivalent.
        text_positions = torch.tensor([[1, 2, 3]] * 3, device="cuda")
        one_q, one_k = rope.apply_cached(q, k, q.size(-1), rope.prepare(text_positions[:1]))
        three_q, three_k = rope.apply_cached(q, k, q.size(-1), rope.prepare(text_positions))
        torch.testing.assert_close(one_q, three_q, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(one_k, three_k, atol=2e-2, rtol=2e-2)

        changed_height = positions.clone()
        changed_height[1, 0] += 3
        changed_q, _ = rope.apply_cached(q, k, q.size(-1), rope.prepare(changed_height))
        self.assertFalse(torch.equal(changed_q[0], actual_q[0]))


if __name__ == "__main__":
    unittest.main()
