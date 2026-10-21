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

        for q_offsets, indptr, last_page_len in (
            ((0, 3, 5), (0, 2, 3), (1, 2)),
            ((0, 1, 4), (0, 1, 4), (4, 3)),
        ):
            layout = SimpleNamespace(
                q_offsets=q_offsets,
                paged_kv_indptr=indptr,
                paged_kv_last_page_len=last_page_len,
            )
            pinned = {
                "prefill_q_offsets": q_offsets,
                "prefill_kv_indptr": indptr,
                "decode_kv_indptr": indptr,
                "prefill_page_len": last_page_len,
                "decode_page_len": last_page_len,
            }
            runner.metadata_pool = SimpleNamespace(
                host_view=lambda name, length: torch.tensor(
                    pinned[name][:length], dtype=torch.int32
                ),
                record_plan_transfer=Mock(),
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
                result = runner._plan_attention(SimpleNamespace(context=context, layout=layout))
                args = wrapper.plan.call_args.args
                offsets = (
                    ((0, q_offsets), (1, indptr), (3, last_page_len))
                    if prefill
                    else ((0, indptr), (2, last_page_len))
                )
                for index, expected in offsets:
                    self.assertEqual(args[index].dtype, torch.int32)
                    self.assertEqual(args[index].device.type, "cpu")
                    self.assertEqual(args[index].tolist(), list(expected))
                self.assertIs(args[2 if prefill else 1], indices)
                self.assertEqual(args[4 if prefill else 3 : 8 if prefill else 7], (8, 2, 128, 16))
                self.assertIs(result.attention_wrapper, wrapper)
                self.assertIs(result.paged_kv_indices, indices)

        # Exercise load -> plan -> event in a CPU-backed graph bucket, without
        # requiring CUDA or duplicating the FlashInfer GPU integration test.
        bucket = object.__new__(DecodeGraphBucket)
        bucket.batch_size = 2
        bucket.max_page_indices = 4
        bucket.block_size = 16
        bucket.attention_spec = (8, 2, 128, {})
        specs = {
            "input_ids": ((2,), torch.int64),
            "rope_positions": ((1, 2), torch.int64),
            "kv_positions": ((2,), torch.int32),
            "state_slot_ids": ((2,), torch.int32),
            "paged_kv_indptr": ((3,), torch.int32),
            "paged_kv_indices": ((4,), torch.int32),
            "paged_kv_last_page_len": ((2,), torch.int32),
        }
        for name, (shape, dtype) in specs.items():
            setattr(bucket, name, torch.zeros(shape, dtype=dtype))
        bucket._host_buffers = {name: torch.empty_like(getattr(bucket, name)) for name in specs}
        bucket._host_transfer_done = Mock()
        bucket._has_pending_host_transfer = False
        bucket.wrapper = Mock()
        steps = []

        def simulate_plan(*args, **kwargs):
            steps.append("plan")
            bucket.paged_kv_indptr.copy_(args[0])
            bucket.paged_kv_last_page_len.copy_(args[2])

        bucket.wrapper.plan.side_effect = simulate_plan
        bucket._host_transfer_done.synchronize.side_effect = lambda: steps.append("wait")
        bucket._host_transfer_done.record.side_effect = lambda _: steps.append("record")
        graph_layouts = (
            SimpleNamespace(
                input_ids=(1, 2),
                rope_positions=((0, 0),),
                kv_positions=(4, 1),
                state_slots=(0, 1),
                paged_kv_indptr=(0, 2, 3),
                paged_kv_indices=(7, 2, 9),
                paged_kv_last_page_len=(1, 2),
            ),
            SimpleNamespace(
                input_ids=(3, 4),
                rope_positions=((0, 0),),
                kv_positions=(8, 2),
                state_slots=(1, 0),
                paged_kv_indptr=(0, 1, 4),
                paged_kv_indices=(4, 6, 8, 5),
                paged_kv_last_page_len=(8, 3),
            ),
        )
        with patch("nanovllm.engine.decode_graph.torch.cuda.current_stream", return_value="stream"):
            for index, layout in enumerate(graph_layouts):
                bucket.prepare(layout)
                self.assertEqual(
                    steps,
                    ["plan", "record"]
                    if index == 0
                    else ["plan", "record", "wait", "plan", "record"],
                )
                self.assertEqual(bucket.paged_kv_indptr.tolist(), list(layout.paged_kv_indptr))
                self.assertEqual(
                    bucket.paged_kv_last_page_len.tolist(),
                    list(layout.paged_kv_last_page_len),
                )
                self.assertEqual(
                    bucket.wrapper.plan.call_args.args[1].tolist(),
                    list(layout.paged_kv_indices),
                )
        self.assertEqual(bucket._host_transfer_done.synchronize.call_count, 1)
        self.assertEqual(bucket._host_transfer_done.record.call_count, 2)

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
