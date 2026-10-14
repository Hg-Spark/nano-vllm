"""ModelRunner CUDA Graph lifecycle and context contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.context import Context


@unittest.skipIf(torch is None, "requires torch")
class ModelRunnerTest(unittest.TestCase):
    def test_initialization_failure_restores_torch_device_state(self):
        config = SimpleNamespace(kv_cache_block_size=16, dtype="float16")
        previous_device = torch.device("cuda:1")
        with (
            patch(
                "nanovllm.engine.model_runner.torch.cuda.current_device",
                return_value=1,
            ),
            patch("nanovllm.engine.model_runner.torch.cuda.set_device") as set_cuda_device,
            patch(
                "nanovllm.engine.model_runner.torch.get_default_device",
                return_value=previous_device,
            ),
            patch("nanovllm.engine.model_runner.torch.set_default_device") as set_default_device,
            patch(
                "nanovllm.engine.model_runner.torch.get_default_dtype",
                return_value=torch.float32,
            ),
            patch("nanovllm.engine.model_runner.torch.set_default_dtype") as set_default_dtype,
            patch("nanovllm.engine.model_runner.torch.cuda.is_available", return_value=False),
            patch(
                "nanovllm.engine.model_runner.Qwen3_5ForCausalLM",
                side_effect=RuntimeError("init failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "init failed"):
                ModelRunner(config)

        self.assertEqual([call.args[0] for call in set_cuda_device.call_args_list], [0, 1])
        self.assertEqual(
            [call.args[0] for call in set_default_device.call_args_list],
            ["cuda", previous_device],
        )
        self.assertEqual(
            [call.args[0] for call in set_default_dtype.call_args_list],
            [torch.float16, torch.float32],
        )

    def test_all_greedy_batch_avoids_temperature_upload(self):
        runner = object.__new__(ModelRunner)
        runner.sampler = Mock(side_effect=AssertionError("sampling is unnecessary"))
        seqs = [
            Sequence([1], SamplingParams(temperature=0.0)),
            Sequence([2], SamplingParams(temperature=0.0)),
        ]
        logits = torch.tensor([[0.0, 3.0], [7.0, 1.0]])
        with patch(
            "nanovllm.engine.model_runner.torch.tensor",
            side_effect=AssertionError("no temperature tensor needed"),
        ):
            result = runner._sample_logits(logits, seqs, [0, 1])
        self.assertEqual(result, [1, 0])
        runner.sampler.assert_not_called()

    def test_decode_graph_replay_failure_propagates(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = 4
        runner.model = object()
        runner.kv_cache = object()
        bucket = Mock()
        bucket.can_run.return_value = True
        bucket.replay.side_effect = RuntimeError("replay failed")
        runner.decode_graph_buckets = {1: bucket}

        with patch("nanovllm.engine.model_runner.build_decode_batch_layout", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "replay failed"):
                runner._run_decode_graph((Mock(),), [Sequence([1, 2])])
        bucket.prepare.assert_called_once()

    def test_configured_decode_graph_layout_mismatch_raises(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = 4
        bucket = Mock()
        bucket.can_run.return_value = False
        runner.decode_graph_buckets = {1: bucket}

        with patch("nanovllm.engine.model_runner.build_decode_batch_layout", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "configured decode graph bucket 1"):
                runner._run_decode_graph((Mock(),), [Sequence([1, 2])])
        bucket.prepare.assert_not_called()

    def test_paged_attention_requires_metadata(self):
        runner = object.__new__(ModelRunner)
        with self.assertRaisesRegex(RuntimeError, "regular attention requires paged KV"):
            runner._plan_attention(Context(is_prefill=True))

    def test_model_runner_exit_releases_gpu_owners(self):
        runner = object.__new__(ModelRunner)
        runner._closed = False
        runner.model = object()
        runner.kv_cache = object()
        runner.sampler = object()

        with (
            patch("nanovllm.engine.model_runner.torch.cuda.is_available", return_value=True),
            patch("nanovllm.engine.model_runner.torch.cuda.synchronize") as synchronize,
            patch("nanovllm.engine.model_runner.torch.cuda.empty_cache") as empty_cache,
        ):
            runner.exit()
            runner.exit()

        synchronize.assert_called_once()
        empty_cache.assert_called_once()
        self.assertIsNone(runner.model)


if __name__ == "__main__":
    unittest.main()
