"""FlashInfer GDN packed prefill and indexed decode state contracts."""

import unittest
from types import SimpleNamespace

import torch

from nanovllm.layers.gated_delta_net import GatedDeltaNet


def supported_gpu():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (9, 10, 12)


@unittest.skipUnless(supported_gpu(), "requires SM90+ FlashInfer GDN kernels")
class FlashInferGDNTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        config = SimpleNamespace(
            hidden_size=256,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
        )
        self.layer = GatedDeltaNet(config).cuda().to(torch.bfloat16)
        with torch.no_grad():
            self.layer.A_log.fill_(0)
        self.layer.state_pool.allocate_state_cache(2)

    @torch.no_grad()
    def test_prefill_delta_rule_matches_independent_recurrence(self):
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                length, heads, dim = 9, 2, 128
                q = torch.randn(length, 1, dim, device="cuda", dtype=dtype)
                k = torch.randn_like(q)
                v = torch.randn(length, heads, dim, device="cuda", dtype=dtype)
                q = (
                    (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6))
                    .to(dtype)
                    .contiguous()
                )
                k = (
                    (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6))
                    .to(dtype)
                    .contiguous()
                )
                alpha = (-torch.rand(length, heads, device="cuda") * 0.2).exp()
                beta = torch.rand(length, heads, device="cuda")
                pool = torch.randn(3, heads, dim, dim, device="cuda") * 0.01
                reference_state = pool[2].clone()
                reference_outputs = []
                for index in range(length):
                    key = k[index].float().expand(heads, -1)
                    query = q[index].float().expand(heads, -1)
                    reference_state = reference_state * alpha[index, :, None, None]
                    predicted = (reference_state * key[:, None, :]).sum(-1)
                    delta = (v[index].float() - predicted) * beta[index, :, None]
                    reference_state = reference_state + delta[:, :, None] * key[:, None, :]
                    reference_outputs.append(
                        (reference_state * query[:, None, :]).sum(-1) * dim**-0.5
                    )

                output, _ = chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g=alpha,
                    beta=beta,
                    scale=dim**-0.5,
                    initial_state=pool,
                    output_state=pool,
                    output_final_state=True,
                    cu_seqlens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
                    state_indices=torch.tensor([2], device="cuda", dtype=torch.int32),
                    use_qk_l2norm_in_kernel=False,
                    use_cp=False,
                    backend="flashinfer",
                )
                torch.testing.assert_close(
                    output.float(), torch.stack(reference_outputs), atol=1e-2, rtol=5e-2
                )
                torch.testing.assert_close(pool[2], reference_state, atol=2e-2, rtol=5e-2)

    @torch.no_grad()
    def test_packed_prefill_matches_per_slot_results(self):
        conv_state = torch.randn_like(self.layer.state_pool.conv_state) * 0.01
        recurrent_state = torch.randn_like(self.layer.state_pool.recurrent_state) * 0.01
        self.layer.state_pool.conv_state.copy_(conv_state)
        self.layer.state_pool.recurrent_state.copy_(recurrent_state)
        inputs = torch.randn(66, 256, device="cuda", dtype=torch.bfloat16)
        actual = self.layer._forward_prefill_batch(
            inputs,
            (0, 1),
            (4, 3),
            (0, 2, 66),
            torch.tensor([0, 2, 66], device="cuda", dtype=torch.int32),
            torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            0,
            ((), ()),
            (0, 0, 0),
            None,
        )
        actual_state = self.layer.state_pool.recurrent_state.clone()
        actual_conv = self.layer.state_pool.conv_state.clone()
        self.layer.state_pool.conv_state.copy_(conv_state)
        self.layer.state_pool.recurrent_state.copy_(recurrent_state)
        parts = []
        for slot, start, end, prefix_len in ((0, 0, 2, 4), (1, 2, 66, 3)):
            parts.append(
                self.layer._forward_prefill_batch(
                    inputs[start:end],
                    (slot,),
                    (prefix_len,),
                    (0, end - start),
                    torch.tensor([0, end - start], device="cuda", dtype=torch.int32),
                    torch.tensor([slot], device="cuda", dtype=torch.int32),
                    0,
                    ((),),
                    (0, 0),
                    None,
                )
            )
        torch.testing.assert_close(actual, torch.cat(parts), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(
            actual_state, self.layer.state_pool.recurrent_state, atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(actual_conv, self.layer.state_pool.conv_state)

    @torch.no_grad()
    def test_intermediate_checkpoints_match_chunked_resume(self):
        inputs = torch.randn(192, 256, device="cuda", dtype=torch.bfloat16)
        offsets = torch.tensor([0, 192], device="cuda", dtype=torch.int32)
        slot_ids = torch.tensor([0], device="cuda", dtype=torch.int32)
        checkpoints = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
        full_output = self.layer._forward_prefill_batch(
            inputs,
            (0,),
            (0,),
            (0, 192),
            offsets,
            slot_ids,
            64,
            ((64, 128),),
            (0, 3),
            checkpoints,
        )
        pool = self.layer.state_pool
        checkpoint_states = {
            position: tuple(state.clone() for state in pool.checkpoint_state_slot(0, position))
            for position in (64, 128)
        }
        final_states = tuple(state.clone() for state in pool.slot(0))
        pool.conv_state.zero_()
        pool.recurrent_state.zero_()
        outputs = []
        for start in (0, 64, 128):
            outputs.append(
                self.layer._forward_prefill_batch(
                    inputs[start : start + 64],
                    (0,),
                    (start,),
                    (0, 64),
                    torch.tensor([0, 64], device="cuda", dtype=torch.int32),
                    slot_ids,
                    0,
                    ((),),
                    (0, 0),
                    None,
                )
            )
            expected = checkpoint_states.get(start + 64, final_states)
            for actual, reference in zip(pool.slot(0), expected):
                torch.testing.assert_close(actual, reference, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(full_output, torch.cat(outputs), atol=5e-2, rtol=5e-2)

    @torch.no_grad()
    def test_fp16_decode_matches_bf16_operand_reference_and_state(self):
        from causal_conv1d import causal_conv1d_update

        self.layer.to(torch.float16)
        self.layer.state_pool.allocate_state_cache(2)
        pool = self.layer.state_pool
        pool.conv_state.copy_(torch.randn_like(pool.conv_state) * 0.01)
        pool.recurrent_state.copy_(torch.randn_like(pool.recurrent_state) * 0.02)
        initial_conv = pool.conv_state.clone()
        initial_recurrent = pool.recurrent_state.clone()
        expected_recurrent = initial_recurrent.clone()
        inputs = torch.randn(2, 256, device="cuda", dtype=torch.float16) * 0.5
        slots = torch.tensor([1, 0], device="cuda", dtype=torch.int32)

        mixed, z, b, a = self.layer._project_inputs(inputs)
        mixed = causal_conv1d_update(
            mixed,
            pool.conv_state,
            self.layer.conv1d.weight.squeeze(1),
            None,
            activation="silu",
            conv_state_indices=slots,
        )
        expected_conv = pool.conv_state.clone()
        q, k, v = self.layer._split_qkv(mixed)
        # Independent FP32 recurrence using the BF16 operands consumed by the
        # FlashInfer 0.7.0 decode kernel (not FlashInfer's own output as oracle).
        q, k, v = (item.to(torch.bfloat16).float() for item in (q, k, v))
        repeats = self.layer.num_v_heads // self.layer.num_k_heads
        q = q.repeat_interleave(repeats, dim=1)
        k = k.repeat_interleave(repeats, dim=1)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        beta = b.float().sigmoid()
        decay = torch.exp(
            -self.layer.A_log.float().exp()
            * torch.nn.functional.softplus(a.float() + self.layer.dt_bias.float())
        )
        expected_core = []
        for row, slot in enumerate(slots.tolist()):
            state = expected_recurrent[slot] * decay[row, :, None, None]
            predicted = torch.einsum("hvk,hk->hv", state, k[row])
            delta = (v[row] - predicted) * beta[row, :, None]
            state = state + delta[:, :, None] * k[row, :, None, :]
            expected_recurrent[slot] = state
            expected_core.append(
                torch.einsum("hvk,hk->hv", state, q[row]) * self.layer.head_k_dim**-0.5
            )
        expected = self.layer._finalize_output(
            torch.stack(expected_core).to(torch.bfloat16).to(inputs.dtype), z
        )

        pool.conv_state.copy_(initial_conv)
        pool.recurrent_state.copy_(initial_recurrent)
        actual = self.layer._forward_decode_batch(inputs, slots)

        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=5e-2)
        torch.testing.assert_close(pool.conv_state, expected_conv)
        torch.testing.assert_close(pool.recurrent_state, expected_recurrent, atol=5e-3, rtol=5e-2)

    @torch.no_grad()
    def test_decode_tracks_slot_permutations_across_steps(self):
        conv_state = torch.randn_like(self.layer.state_pool.conv_state) * 0.01
        recurrent_state = torch.randn_like(self.layer.state_pool.recurrent_state) * 0.01
        self.layer.state_pool.conv_state.copy_(conv_state)
        self.layer.state_pool.recurrent_state.copy_(recurrent_state)

        permutations = ((1, 0), (0, 1), (1, 0))
        batches = [
            (
                torch.randn(2, 256, device="cuda", dtype=torch.bfloat16),
                torch.tensor(slots, device="cuda", dtype=torch.int32),
            )
            for slots in permutations
        ]
        # Compare a packed decode with independent single-slot updates.
        expected_outputs = []
        for inputs, slots in batches:
            outputs = [
                self.layer._forward_decode_batch(inputs[row : row + 1], slots[row : row + 1])
                for row in range(len(slots))
            ]
            expected_outputs.append(torch.cat(outputs))
        expected_conv = self.layer.state_pool.conv_state.clone()
        expected_state = self.layer.state_pool.recurrent_state.clone()

        self.layer.state_pool.conv_state.copy_(conv_state)
        self.layer.state_pool.recurrent_state.copy_(recurrent_state)
        for (inputs, slots), expected in zip(batches, expected_outputs):
            actual = self.layer._forward_decode_batch(inputs, slots)
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(
            self.layer.state_pool.conv_state, expected_conv, atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(
            self.layer.state_pool.recurrent_state, expected_state, atol=5e-2, rtol=5e-2
        )


if __name__ == "__main__":
    unittest.main()
