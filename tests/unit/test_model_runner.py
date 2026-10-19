"""ModelRunner CUDA Graph lifecycle and context contracts."""

import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams


@unittest.skipIf(torch is None, "requires torch")
class ModelRunnerTest(unittest.TestCase):
    def test_all_greedy_batch_avoids_temperature_upload(self):
        runner = object.__new__(ModelRunner)
        runner.sampler = Mock(side_effect=AssertionError("sampling is unnecessary"))
        runner.metadata_pool = Mock()
        seqs = [
            Sequence([1], SamplingParams(temperature=0.0)),
            Sequence([2], SamplingParams(temperature=0.0)),
        ]
        logits = torch.tensor([[0.0, 3.0], [7.0, 1.0]])
        result = runner._sample_logits(logits, seqs, [0, 1])
        self.assertEqual(result, [1, 0])
        runner.sampler.assert_not_called()
        runner.metadata_pool.to_cuda.assert_not_called()

    def test_model_runner_exit_releases_gpu_owners(self):
        runner = object.__new__(ModelRunner)
        runner._closed = False
        runner.model = object()
        runner.kv_cache = object()
        runner.sampler = object()
        bucket = Mock()
        metadata_pool = Mock()
        runner.decode_graph_buckets = {1: bucket}
        runner.metadata_pool = metadata_pool

        with (
            patch("nanovllm.engine.model_runner.torch.cuda.is_available", return_value=True),
            patch("nanovllm.engine.model_runner.torch.cuda.synchronize") as synchronize,
            patch("nanovllm.engine.model_runner.torch.cuda.empty_cache") as empty_cache,
        ):
            runner.exit()
            runner.exit()

        synchronize.assert_called_once()
        empty_cache.assert_called_once()
        bucket.release.assert_called_once()
        metadata_pool.clear.assert_called_once()
        self.assertIsNone(runner.model)


if __name__ == "__main__":
    unittest.main()
