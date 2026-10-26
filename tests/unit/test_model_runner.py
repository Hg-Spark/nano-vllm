"""ModelRunner CUDA Graph lifecycle and context contracts."""

import unittest
from contextlib import contextmanager
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


@unittest.skipIf(torch is None, "requires torch")
class ModelRunnerTest(unittest.TestCase):
    def test_decode_graph_waits_before_reusing_pinned_metadata(self):
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

        # Release must wait for the final transfer and be safe to repeat.
        transfer_done = bucket._host_transfer_done
        bucket.graph = Mock()
        graph = bucket.graph
        bucket.release()
        bucket.release()
        self.assertEqual(transfer_done.synchronize.call_count, 2)
        graph.reset.assert_called_once()
        self.assertFalse(bucket._has_pending_host_transfer)

    def test_decode_graph_reuses_unchanged_page_index_buffer(self):
        bucket = object.__new__(DecodeGraphBucket)
        bucket.batch_size = 1
        bucket.max_page_indices = 2
        bucket._has_pending_host_transfer = False
        bucket._page_table_key = None
        bucket._copy_layout = Mock()
        bucket._host_buffers = {
            "paged_kv_indptr": torch.empty(2, dtype=torch.int32),
            "paged_kv_last_page_len": torch.empty(1, dtype=torch.int32),
        }
        layout = SimpleNamespace(
            input_ids=(1,),
            rope_positions=((0,),),
            kv_positions=(7,),
            state_slots=(0,),
            paged_kv_indptr=(0, 1),
            paged_kv_indices=(9,),
            paged_kv_last_page_len=(8,),
            page_table_key=(16, ((42, 1, 1, 1),)),
        )
        bucket.load(layout)
        bucket.load(layout)
        uploads = [call.args[0] for call in bucket._copy_layout.call_args_list]
        self.assertEqual(uploads.count("paged_kv_indices"), 1)
        changed = SimpleNamespace(**vars(layout))
        changed.page_table_key = (16, ((42, 2, 1, 1),))
        bucket.load(changed)
        uploads = [call.args[0] for call in bucket._copy_layout.call_args_list]
        self.assertEqual(uploads.count("paged_kv_indices"), 2)

    def test_decode_graph_fallback_counters(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = 16
        runner.decode_graph_buckets = {}
        runner.decode_graph_fallbacks = {"missing_bucket": 0, "layout_exceeded": 0}
        chunks = (object(),)
        self.assertIsNone(runner._run_decode_graph(chunks, [], layout=None))
        self.assertEqual(runner.decode_graph_fallbacks["missing_bucket"], 1)

        runner.decode_graph_buckets = {1: SimpleNamespace(can_run=lambda _: False)}
        self.assertIsNone(runner._run_decode_graph(chunks, [], layout=object()))
        self.assertEqual(runner.decode_graph_fallbacks["layout_exceeded"], 1)

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

    def test_mixed_sampling_uses_pinned_metadata_indices(self):
        runner = object.__new__(ModelRunner)
        runner.sampler = Mock(return_value=torch.tensor([0, 1], dtype=torch.long))
        uploads = []

        @contextmanager
        def fake_batch():
            def upload(name, values, dtype):
                uploads.append((name, tuple(values)))
                return torch.tensor(
                    values, dtype=torch.float32 if dtype == "float32" else torch.int64
                )

            yield upload

        runner.metadata_pool = SimpleNamespace(batch=fake_batch)
        seqs = [
            Sequence([1], SamplingParams(temperature=0.0)),
            Sequence([2], SamplingParams(temperature=0.8)),
        ]
        result = runner._sample_logits(torch.tensor([[3.0, 2.0], [1.0, 3.0]]), seqs, [0, 1])
        self.assertEqual(result, [0, 1])
        self.assertEqual(
            uploads,
            [("sample_temperatures", (0.0, 0.8)), ("sample_positions", (1,))],
        )
        self.assertEqual(runner.sampler.call_args.kwargs["sample_positions"].tolist(), [1])

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
        self.assertIsNone(state.image_grid_thw)

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

    def test_model_runner_cleanup_retries_after_sync_and_release_failures(self):
        runner = object.__new__(ModelRunner)
        runner._closed = False
        runner._cleanup_complete = False
        runner.model = object()
        runner.kv_cache = object()
        runner.sampler = object()
        bucket = Mock()
        bucket.release.side_effect = [RuntimeError("bucket failed"), None]
        pool = Mock()
        runner.decode_graph_buckets = {1: bucket}
        runner.metadata_pool = pool

        with (
            patch("nanovllm.engine.model_runner.torch.cuda.is_available", return_value=True),
            patch(
                "nanovllm.engine.model_runner.torch.cuda.synchronize",
                side_effect=[RuntimeError("sync failed"), None],
            ) as synchronize,
            patch("nanovllm.engine.model_runner.torch.cuda.empty_cache") as empty_cache,
        ):
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                runner.exit()
            self.assertTrue(runner._closed)
            self.assertFalse(runner._cleanup_complete)
            self.assertEqual(list(runner.decode_graph_buckets), [1])
            self.assertIsNone(runner.model)
            pool.clear.assert_called_once()

            runner.exit()
            runner.exit()

        self.assertTrue(runner._cleanup_complete)
        self.assertEqual(runner.decode_graph_buckets, {})
        self.assertIsNone(runner.metadata_pool)
        self.assertEqual(bucket.release.call_count, 2)
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(empty_cache.call_count, 2)
        pool.clear.assert_called_once()

    def test_gdn_capture_preserves_prompt_and_rolls_completion_cpu_tensor(self):
        from nanovllm.engine.state_manager import GDNStateSnapshot

        class Layer:
            def __init__(self, value):
                self.conv = torch.full((3, 4), value, dtype=torch.bfloat16)
                self.recurrent = torch.full((2, 4), value, dtype=torch.float32)

            def slot(self, _slot_id):
                return self.conv, self.recurrent

        layers = [Layer(1), Layer(2)]
        runner = object.__new__(ModelRunner)
        runner.model = SimpleNamespace(state_cache_modules=lambda: layers)
        runner.block_size = 16
        seq = Sequence(list(range(65)))
        seq.state_slot = 0

        prompt = runner.capture_live_states([(seq, 64)])[seq.seq_id]
        self.assertIsInstance(prompt, GDNStateSnapshot)
        seq.prompt_snapshot = prompt
        prompt_before = tuple((conv.clone(), state.clone()) for conv, state in prompt.layers)

        for layer in layers:
            layer.conv.fill_(5)
            layer.recurrent.fill_(6)
        completion = runner.capture_live_states([(seq, 80)])[seq.seq_id]
        seq.completion_snapshot = completion
        completion_ptrs = tuple((c.data_ptr(), r.data_ptr()) for c, r in completion.layers)

        for layer in layers:
            layer.conv.fill_(7)
            layer.recurrent.fill_(8)
        updated = runner.capture_live_states([(seq, 96)])[seq.seq_id]

        self.assertEqual(updated.num_tokens, 96)
        self.assertEqual(
            tuple((c.data_ptr(), r.data_ptr()) for c, r in updated.layers), completion_ptrs
        )
        for (actual_conv, actual_state), (saved_conv, saved_state) in zip(
            prompt.layers, prompt_before
        ):
            torch.testing.assert_close(actual_conv, saved_conv)
            torch.testing.assert_close(actual_state, saved_state)
        for actual_conv, actual_state in updated.layers:
            self.assertTrue(torch.all(actual_conv == 7))
            self.assertTrue(torch.all(actual_state == 8))


if __name__ == "__main__":
    unittest.main()
