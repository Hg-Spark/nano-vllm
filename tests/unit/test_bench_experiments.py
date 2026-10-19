"""CPU-only benchmark parity and comparable A/B result checks."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import bench_experiments as experiments
from benchmarks import capacity_search
from benchmarks import summarize_bench as summary
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

    def test_throughput_graph_replays_exclude_warmup(self):
        bucket = SimpleNamespace(replay_count=0)
        engine = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=128)),
            max_model_len=128,
            model_runner=SimpleNamespace(
                num_kv_cache_blocks=32,
                kv_cache=SimpleNamespace(numel=lambda: 32, element_size=lambda: 4),
                decode_graph_buckets={1: bucket},
            ),
            scheduler=SimpleNamespace(schedule=lambda: None),
            _prepare_request=lambda tokens, params: Sequence(tokens, params),
            exit=lambda: None,
        )
        args = SimpleNamespace(
            model="stub",
            dtype="float16",
            budget=16,
            max_num_seqs=4,
            max_model_len=128,
            gpu_memory_utilization=0.8,
            kv_cache_dtype="auto",
            cache_entries=0,
            graph_sizes=(1,),
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
            return {}

        with (
            patch("nanovllm.LLM", return_value=engine),
            patch("scripts.bench.run_benchmark", side_effect=benchmark),
            patch("torch.cuda.get_device_name", return_value="cpu-test"),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            with self.assertRaisesRegex(RuntimeError, "no measured replay"):
                experiments.run(args)

    def test_capacity_search_uses_fixed_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                model="stub",
                dtype="float16",
                kv_cache_dtype="auto",
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

    def test_ab_parity_and_speedup(self):
        base = {
            "schema_version": 1,
            "experiment": "benchmark_decode",
            "execution_status": "completed",
            "config": {
                "scenario": "decode",
                "max_num_seqs": 2,
                "budget": 512,
                "graph_sizes": [],
                "cache_entries": 0,
                "schedule_mode": "mixed",
            },
            "result": {
                "gpu": "GPU",
                "compute_capability": [9, 0],
                "metrics": {
                    "output_throughput_tok_s": 100,
                    "tpot_seconds": {"mean": 0.01},
                },
                "details": {"token_hash": "same"},
            },
        }
        graph = json.loads(json.dumps(base))
        graph["config"]["graph_sizes"] = [2]
        graph["result"]["metrics"]["output_throughput_tok_s"] = 150
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "eager.json").write_text(json.dumps(base), encoding="utf-8")
            graph_file = Path(folder) / "graph.json"
            graph_file.write_text(json.dumps(graph), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                summary.main([folder])
            self.assertIn("Output throughput improvement: +50.00%", output.getvalue())
            self.assertIn("Successful paired comparisons: 1", output.getvalue())

            # Other experiments in the same folder must not invalidate A/B results.
            (Path(folder) / "parity.json").write_text(
                json.dumps({"experiment": "image_parity", "execution_status": "error"}),
                encoding="utf-8",
            )
            with redirect_stdout(StringIO()):
                summary.main([folder])

            graph["result"]["details"]["token_hash"] = "different"
            graph_file.write_text(json.dumps(graph), encoding="utf-8")
            with (
                redirect_stdout(StringIO()),
                self.assertRaisesRegex(SystemExit, "Output token hashes differ"),
            ):
                summary.main([folder])


if __name__ == "__main__":
    unittest.main()
