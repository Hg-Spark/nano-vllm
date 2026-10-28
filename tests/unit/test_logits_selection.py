"""CPU tests for runtime-owned prefill and sampling indices."""

import unittest
from unittest.mock import patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.utils.context import Context, use_context


class LogitsSelectionTest(unittest.TestCase):
    def test_prefill_staged_indices_match_sequence_selection(self):
        class FakeHead:
            lm_head = staticmethod(lambda hidden: hidden)

        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        context = Context(is_prefill=True, prefill_q_offsets=(0, 3, 5))
        with use_context(context):
            reference = Qwen3_5ForCausalLM.compute_logits(FakeHead(), hidden, sequence_indices=[1])
            staged = Qwen3_5ForCausalLM.compute_logits(
                FakeHead(), hidden, token_indices=torch.tensor([4], dtype=torch.int64)
            )
            torch.testing.assert_close(staged, reference)
            torch.testing.assert_close(staged, hidden[4:5])

    def test_sampler_accepts_staged_positions_without_allocating_indices(self):
        sampler = Sampler()
        logits = torch.tensor([[9.0, 0.0], [8.0, 1.0], [1.0, 9.0]])
        temperatures = torch.ones(3, dtype=torch.float32)
        indices = torch.tensor([1], dtype=torch.int64)
        with patch.object(
            sampler, "_sample", return_value=torch.tensor([1], dtype=torch.long)
        ) as sampled:
            result = sampler(logits, temperatures, sample_positions=indices)
        self.assertEqual(result.tolist(), [0, 1, 1])
        torch.testing.assert_close(sampled.call_args.args[0], logits.float()[1:2])
        torch.testing.assert_close(sampled.call_args.args[1], temperatures[1:2])


if __name__ == "__main__":
    unittest.main()
