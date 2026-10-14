"""Layer diagnosis comparison checks."""

import contextlib
import io
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import diagnose_qwen3_5 as diagnose
from scripts._common import ValidationFailure


class Tensor:
    def __init__(self, values, shape=None):
        self.values = list(values)
        self.shape = shape if shape is not None else (1, len(self.values))
        self.ndim = len(self.shape)

    def numel(self):
        return len(self.values)

    def __sub__(self, other):
        return Tensor([a - b for a, b in zip(self.values, other.values)], self.shape)

    def abs(self):
        return Tensor([abs(x) for x in self.values], self.shape)

    def square(self):
        return Tensor([x * x for x in self.values], self.shape)

    def max(self):
        return Tensor([max(self.values)], ())

    def mean(self):
        return Tensor([sum(self.values) / len(self.values)], ())

    def sqrt(self):
        return Tensor([math.sqrt(x) for x in self.values], self.shape)

    def item(self):
        return self.values[0]


def tensor_runtime():
    return SimpleNamespace(
        Tensor=Tensor,
        isfinite=lambda t: SimpleNamespace(
            all=lambda: SimpleNamespace(item=lambda: all(math.isfinite(x) for x in t.values))
        ),
    )


class DiagnosisTest(unittest.TestCase):
    def test_compare_layers_detects_divergence_and_nonfinite_values(self):
        with (
            patch.dict(sys.modules, {"torch": tensor_runtime()}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(diagnose.compare_layers([Tensor([1, 2])], [Tensor([1, 2])], 0.02), -1)
            rows = []
            self.assertEqual(diagnose.compare_layers([Tensor([1])], [Tensor([2])], 0.02, rows), 0)
            self.assertEqual(rows[0]["relative_rms"], 1)
            with self.assertRaises(ValidationFailure):
                diagnose.compare_layers([Tensor([float("nan")])], [Tensor([1.0])], 0.02)


if __name__ == "__main__":
    unittest.main()
