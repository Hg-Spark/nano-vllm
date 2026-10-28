"""CPU-only benchmark parity and comparable A/B result checks."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import bench_experiments as experiments
from benchmarks import capacity_search
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import make_running_sequence, make_scheduler


class BenchmarkExperimentsTest(unittest.TestCase):
    def test_exclusive_decode_first_scheduler_hook(self):
        for exclusive in (False, True):
            with self.subTest(exclusive=exclusive):
                scheduler = make_scheduler(
                    max_num_seqs=2, max_num_batched_tokens=8, max_prefix_cache_entries=0
                )
                make_running_sequence(scheduler, [1, 2])
                scheduler.add(Sequence([3, 4], SamplingParams(max_tokens=1)))
                if exclusive:
                    experiments.enable_exclusive_decode_first(scheduler)
                scheduled = scheduler.schedule()
                self.assertEqual(scheduled.decode_tokens, 1)
                self.assertEqual(scheduled.prefill_tokens, 0 if exclusive else 2)

        scheduler = make_scheduler(max_num_seqs=2, max_num_batched_tokens=8)
        experiments.enable_exclusive_decode_first(scheduler)
        scheduler.add(Sequence([3, 4], SamplingParams(max_tokens=1)))
        self.assertEqual(scheduler.schedule().prefill_tokens, 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_throughput_graph_replays_exclude_warmup(self):
        bucket = SimpleNamespace(replay_count=0)
        engine = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=128)),
            max_model_len=128,
            model_runner=SimpleNamespace(
                num_kv_cache_blocks=32,
                kv_cache=SimpleNamespace(numel=lambda: 32, element_size=lambda: 4),
                decode_graph_buckets={1: bucket},
                decode_graph_fallbacks={"missing_bucket": 0, "layout_exceeded": 0},
            ),
            scheduler=SimpleNamespace(schedule=lambda: None),
            _prepare_request=lambda tokens, params: Sequence(tokens, params),
            exit=lambda: None,
        )
        args = SimpleNamespace(
            model="stub",
            budget=16,
            mixed_prefill_limit=128,
            activation_reserve_mib=256,
            max_num_seqs=4,
            max_model_len=128,
            gpu_memory_utilization=0.8,
            kv_cache_dtype="auto",
            kv_cache_k_scale=1.0,
            kv_cache_v_scale=1.0,
            cache_entries=0,
            graph_sizes=(1,),
            min_graph_coverage=0.0,
            seed=0,
            scenario="throughput",
            schedule_mode="mixed",
            num_requests=4,
            input_len=2,
            output_len=2,
        )

        calls = 0

        def benchmark(_engine, _sequences):
            nonlocal calls
            calls += 1
            if calls == 1:
                bucket.replay_count += 1
            else:
                engine.model_runner.decode_graph_fallbacks["layout_exceeded"] += 1
            return {"decode_forward_calls": 1}

        with (
            patch("nanovllm.LLM", return_value=engine),
            patch("scripts.bench.run_benchmark", side_effect=benchmark),
            patch("torch.cuda.get_device_name", return_value="cpu-test"),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            with self.assertRaisesRegex(RuntimeError, "Graph coverage"):
                experiments.run(args)

            calls = 0

            def measured_benchmark(_engine, _sequences):
                nonlocal calls
                calls += 1
                if calls == 2:
                    bucket.replay_count += 1
                return {"decode_forward_calls": 1}

            with patch("scripts.bench.run_benchmark", side_effect=measured_benchmark):
                result = experiments.run(args)
            self.assertEqual(result["metrics"]["graph_replays"], 1)
            self.assertEqual(result["metrics"]["graph_coverage"], 1.0)
            self.assertEqual(result["metrics"]["eager_decode_forward_calls"], 0)

    def test_graph_usage_consistency(self):
        from benchmarks.bench_experiments import graph_counters, measured_graph_usage

        runner = SimpleNamespace(
            decode_graph_buckets={1: SimpleNamespace(replay_count=2)},
            decode_graph_fallbacks={"missing_bucket": 4, "layout_exceeded": 0},
        )
        before = graph_counters(runner)
        runner.decode_graph_buckets[1].replay_count += 3
        runner.decode_graph_fallbacks["missing_bucket"] += 2
        metrics = measured_graph_usage(runner, before, 5)
        self.assertEqual(metrics["graph_coverage"], 0.6)
        self.assertEqual(metrics["eager_decode_forward_calls"], 2)
        with self.assertRaisesRegex(RuntimeError, "accounting is inconsistent"):
            measured_graph_usage(runner, before, 4)

    def test_capacity_search_uses_fixed_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                model="stub",
                kv_cache_dtype="auto",
                kv_cache_k_scale=1.0,
                kv_cache_v_scale=1.0,
                budget=512,
                input_len=2048,
                output_len=128,
                max_model_len=4096,
                gpu_memory_utilization=0.8,
                seed=0,
                out_dir=Path(folder),
            )

            def over_capacity(command, **kwargs):
                self.assertEqual(command[command.index("--scenario") + 1], "decode")
                self.assertEqual(command[command.index("--measure-steps") + 1], "126")
                kwargs["stdout"].write("ValueError: fixed decode batch exceeds KV capacity")
                return SimpleNamespace(returncode=1)

            with patch("benchmarks.capacity_search.subprocess.run", side_effect=over_capacity):
                self.assertFalse(capacity_search.probe(args, 32))


if __name__ == "__main__":
    unittest.main()
