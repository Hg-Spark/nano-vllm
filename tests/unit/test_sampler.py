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
            tokens = Sampler()(logits, temperatures, sample_positions=[1])
        torch.testing.assert_close(tokens, torch.tensor([1, 0]))
        sample.assert_called_once()
        torch.testing.assert_close(sample.call_args.args[0], logits[1:].float())
        torch.testing.assert_close(sample.call_args.args[1], temperatures[1:])

    def test_invalid_temperature_is_rejected(self):
        for temperature in (-0.1, math.nan, math.inf, -math.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    SamplingParams(temperature=temperature)

    def test_invalid_max_tokens_is_rejected(self):
        for value in (0, -1, True, 1.5, float("nan"), "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SamplingParams(max_tokens=value)


if __name__ == "__main__":
    unittest.main()
