"""Numerical comparison of independent GDN block algebra with Transformers.

External oracle: installed Transformers 5.19.0; no checkpoint required.
"""

import unittest

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
)

from nanovllm.layers.gdn_chunk import chunk_gated_delta_rule


class HFChunkPrefillTest(unittest.TestCase):
    @torch.no_grad()
    def check_device(self, device, lengths, dtypes):
        torch.manual_seed(17)
        for length in lengths:
            for dtype in dtypes:
                with self.subTest(device=device, length=length, dtype=dtype):
                    query, key = [
                        torch.randn(1, length, 2, 128, dtype=dtype, device=device) for _ in range(2)
                    ]
                    value = torch.randn(1, length, 2, 96, dtype=dtype, device=device)
                    decay = -torch.rand(1, length, 2, device=device) * 0.2
                    beta = torch.rand(1, length, 2, device=device)
                    initial_state = torch.randn(1, 2, 128, 96, device=device)
                    expected, expected_state = torch_chunk_gated_delta_rule(
                        query,
                        key,
                        value,
                        g=decay,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    actual, actual_state = chunk_gated_delta_rule(
                        query,
                        key,
                        value,
                        decay,
                        beta,
                        initial_state,
                    )
                    # Different matrix factorization and operation ordering need tolerances.
                    # The FP64 recurrence test is the primary mathematical oracle.
                    tolerance = 2e-3 if dtype == torch.float16 else 5e-5
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(actual_state, expected_state, rtol=5e-5, atol=5e-5)

    def test_cpu_matches_hf(self):
        self.check_device("cpu", (1, 64, 65), (torch.float32, torch.float16))


if __name__ == "__main__":
    unittest.main()
