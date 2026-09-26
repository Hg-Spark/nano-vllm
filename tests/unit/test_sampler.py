import math
import unittest

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class SamplingTest(unittest.TestCase):

    def test_zero_temperature_uses_argmax(self):
        logits = torch.tensor([
            [1.0, 4.0, 2.0],
            [5.0, 3.0, 6.0],
        ])
        temperatures = torch.zeros(2)

        tokens = Sampler()(logits, temperatures)

        torch.testing.assert_close(
            tokens,
            torch.tensor([1, 2]),
        )

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
