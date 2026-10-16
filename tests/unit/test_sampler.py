import math
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class SamplingTest(unittest.TestCase):
    def test_mixed_temperature_preserves_greedy_and_sampling(self):
        logits = torch.tensor([[1.0, 4.0, 2.0], [5.0, 3.0, 6.0]])
        temperatures = torch.tensor([0.0, 1.0])
        with patch.object(Sampler, "_sample", return_value=torch.tensor([0])) as sample:
            tokens = Sampler()(logits, temperatures)
        torch.testing.assert_close(tokens, torch.tensor([1, 0]))
        sample.assert_called_once()
        torch.testing.assert_close(sample.call_args.args[0], logits[1:].float())
        torch.testing.assert_close(sample.call_args.args[1], temperatures[1:])

    def test_negative_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            SamplingParams(temperature=-0.1)

    def test_non_finite_temperature_is_rejected(self):
        for temperature in (math.nan, math.inf, -math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "finite"):
                    SamplingParams(temperature=temperature)

    def test_non_positive_max_tokens_is_rejected(self):
        with self.assertRaises(ValueError):
            SamplingParams(max_tokens=0)


if __name__ == "__main__":
    unittest.main()
