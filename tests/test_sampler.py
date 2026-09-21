import unittest

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class GreedySamplingTest(unittest.TestCase):

    def test_zero_temperature_is_allowed(self):
        params = SamplingParams(temperature=0.0)
        self.assertEqual(params.temperature, 0.0)

    def test_zero_temperature_uses_argmax(self):
        logits = torch.tensor([
            [1.0, 4.0, 2.0],
            [5.0, 3.0, 6.0],
        ])
        temperatures = torch.zeros(2)

        tokens = Sampler()(logits, temperatures)

        torch.testing.assert_close(tokens, torch.tensor([1, 2]))


if __name__ == "__main__":
    unittest.main()
