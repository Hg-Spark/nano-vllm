"""Reject repeated non-aliased checkpoint weights without rejecting tied weights."""

import os
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.utils.loader import load_model


class OneWeightModel(torch.nn.Module):
    checkpoint_skip_prefixes = ()
    checkpoint_prefix_map = ()

    def __init__(self, *, tied=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(2))
        if tied:
            self.b = self.a

    def packed_checkpoint_shards(self):
        return {}

    @staticmethod
    def copy_checkpoint_weight(name, destination, source):
        destination.copy_(source)


class FakeSafetensors:
    def __init__(self, weights):
        self.weights = weights

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def keys(self):
        return self.weights.keys()

    def get_tensor(self, name):
        return self.weights[name]


class LoaderStrictTest(unittest.TestCase):
    def _load(self, weights, *, tied=False):
        model = OneWeightModel(tied=tied)
        with tempfile.TemporaryDirectory() as folder:
            for name in weights:
                with open(os.path.join(folder, name), "wb") as stream:
                    stream.write(b"placeholder")
            with patch(
                "nanovllm.utils.loader.safe_open",
                side_effect=lambda path, *_: FakeSafetensors(weights[os.path.basename(path)]),
            ):
                load_model(model, folder)
        return model

    def test_duplicate_non_shared_parameter_is_rejected(self):
        weights = {
            "1.safetensors": {"a": torch.ones(2)},
            "2.safetensors": {"a": torch.ones(2)},
        }
        with self.assertRaisesRegex(ValueError, "duplicate non-shared checkpoint weight"):
            self._load(weights)

    def test_identical_tied_aliases_are_valid(self):
        model = self._load(
            {
                "1.safetensors": {"a": torch.ones(2)},
                "2.safetensors": {"b": torch.ones(2)},
            },
            tied=True,
        )
        torch.testing.assert_close(model.a, torch.ones(2))

    def test_conflicting_tied_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting checkpoint weights"):
            self._load(
                {
                    "1.safetensors": {"a": torch.ones(2)},
                    "2.safetensors": {"b": torch.zeros(2)},
                },
                tied=True,
            )


if __name__ == "__main__":
    unittest.main()
