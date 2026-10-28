import math
import unittest

from nanovllm.sampling_params import SamplingParams


class SamplingTest(unittest.TestCase):
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
