"""CPU-only benchmark parity and comparable A/B result checks."""

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

    def test_top4_diagnostic_rejects_performance_scenarios(self):
        def base(*extra):
            return [
                "model", "--scenario", "throughput", "--out", "/tmp/top4_probe.json",
                "--trace-top4", *extra,
            ]
        with self.assertRaises(SystemExit):
            experiments.arguments(base())
        with self.assertRaises(SystemExit):
            experiments.arguments([
                "model", "--scenario", "prefix", "--out", "/tmp/top4_probe.json",
                "--trace-top4", "--trace-max-tokens", "0",
            ])

    def test_graph_usage_excludes_warmup_and_rejects_inconsistent_counts(self):
        from benchmarks.bench_experiments import graph_counters, measured_graph_usage

        runner = SimpleNamespace(
            decode_graph_buckets={1: SimpleNamespace(replay_count=2)},
            decode_graph_fallbacks={"missing_bucket": 4, "layout_exceeded": 0},
        )
        # Counters already include warmup: only post-snapshot increments count.
        before = graph_counters(runner)
        runner.decode_graph_buckets[1].replay_count += 3
        runner.decode_graph_fallbacks["missing_bucket"] += 2
        metrics = measured_graph_usage(runner, before, 5)
        self.assertEqual(metrics["graph_replays"], 3)
        self.assertEqual(metrics["decode_forward_calls"], 5)
        self.assertEqual(metrics["graph_coverage"], 0.6)
        self.assertEqual(metrics["eager_decode_forward_calls"], 2)
        with self.assertRaisesRegex(RuntimeError, "accounting is inconsistent"):
            measured_graph_usage(runner, before, 4)

    def test_throughput_graph_coverage_requires_a_measured_replay(self):
        check = experiments.require_throughput_graph_coverage
        with self.assertRaisesRegex(RuntimeError, "Graph coverage"):
            check({"graph_replays": 0, "graph_coverage": 0.0}, 0.0)
        with self.assertRaisesRegex(RuntimeError, "Graph coverage 60.0%"):
            check({"graph_replays": 3, "graph_coverage": 0.6}, 0.95)
        check({"graph_replays": 3, "graph_coverage": 0.6}, 0.6)

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
