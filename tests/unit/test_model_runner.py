"""ModelRunner CUDA Graph lifecycle and context contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from nanovllm.engine.decode_graph import DecodeGraphBucket
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.context import Context


@unittest.skipIf(torch is None, "requires torch")
class ModelRunnerTest(unittest.TestCase):
    def test_attention_plan_uses_cpu_metadata(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = 16
        runner.attention_spec = (8, 2, 128, {})
        runner.prefill_wrapper = Mock()
        runner.decode_wrapper = Mock()
        indices = object()
        layout = SimpleNamespace(
            q_offsets=(0, 3),
            paged_kv_indptr=(0, 1),
            paged_kv_last_page_len=(3,),
        )
        for prefill, wrapper in (
            (True, runner.prefill_wrapper),
            (False, runner.decode_wrapper),
        ):
            context = Context(
                is_prefill=prefill,
                qo_indptr=object() if prefill else None,
                paged_kv_indices=indices,
            )
            runner._plan_attention(SimpleNamespace(context=context, layout=layout))
            args = wrapper.plan.call_args.args
            cpu_args = (0, 1, 3) if prefill else (0, 2)
            for index in cpu_args:
                self.assertEqual(args[index].device.type, "cpu")
            self.assertIs(args[2 if prefill else 1], indices)

        bucket = object.__new__(DecodeGraphBucket)
        bucket.block_size = 16
        bucket.attention_spec = (8, 2, 128, {})
        bucket.wrapper = Mock()
        bucket.paged_kv_indices = torch.tensor([1], dtype=torch.int32)
        bucket._host_buffers = {
            "paged_kv_indptr": torch.tensor([0, 1], dtype=torch.int32),
            "paged_kv_last_page_len": torch.tensor([3], dtype=torch.int32),
        }
        bucket._plan(1)
        args = bucket.wrapper.plan.call_args.args
        self.assertIs(args[0], bucket._host_buffers["paged_kv_indptr"])
        self.assertIs(args[2], bucket._host_buffers["paged_kv_last_page_len"])

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

    def test_visual_feature_cache_uses_text_dtype(self):
        runner = object.__new__(ModelRunner)
        runner.model = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
            ),
            encode_image=Mock(return_value=torch.ones((2, 4), dtype=torch.float32)),
        )
        state = SimpleNamespace(
            pixel_values=Mock(),
            image_grid_thw=Mock(),
            image_start=1,
            image_end=3,
            visual_features=None,
        )
        seq = SimpleNamespace(image_state=state)
        features = runner._ensure_visual_features(seq)
        self.assertEqual(features.dtype, torch.float16)
        self.assertIs(features, state.visual_features)
        self.assertIsNone(state.pixel_values)

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
