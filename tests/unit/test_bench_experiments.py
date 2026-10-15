"""CPU-only regression checks for benchmark result validity and timing."""

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks import bench_experiments as benchmark
from benchmarks import summarize_bench as summary
from scripts import _probes


class BenchmarkExperimentsTest(unittest.TestCase):
    def test_ab_parity_and_speedup(self):
        base = {
            "status": "completed",
            "gpu": "GPU",
            "git": {"commit": "abc", "dirty": False},
            "parameters": {
                "scenario": "decode",
                "max_num_seqs": 2,
                "budget": 512,
                "graph_sizes": [],
                "cache_entries": 0,
                "schedule_mode": "mixed",
            },
            "metrics": {
                "output_throughput_tok_s": 100,
                "tpot_seconds": {"mean": 0.01},
            },
            "details": {"token_hash": "same"},
        }
        graph = json.loads(json.dumps(base))
        graph["parameters"]["graph_sizes"] = [2]
        graph["metrics"]["output_throughput_tok_s"] = 150
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "eager.json").write_text(json.dumps(base), encoding="utf-8")
            graph_file = Path(folder) / "graph.json"
            graph_file.write_text(json.dumps(graph), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                summary.main([folder])
            self.assertIn("Output throughput improvement: +50.00%", output.getvalue())
            self.assertIn("Successful paired comparisons: 1", output.getvalue())

            graph["details"]["token_hash"] = "different"
            graph_file.write_text(json.dumps(graph), encoding="utf-8")
            with (
                redirect_stdout(StringIO()),
                self.assertRaisesRegex(SystemExit, "Output token hashes differ"),
            ):
                summary.main([folder])

    def test_preemption_must_be_observed_when_required(self):
        with self.assertRaisesRegex(RuntimeError, "no preemptions observed"):
            benchmark.validate_preemptions(0, required=True)
        benchmark.validate_preemptions(1, required=True)
        benchmark.validate_preemptions(0, required=False)

    def test_decode_event_record_precedes_sync(self):
        events = []

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self):
                events.append(self.name)

            def synchronize(self):
                events.append("event_sync")

            def elapsed_time(self, other):
                return 2.0

        names = iter(("start", "end"))
        cuda = SimpleNamespace(
            Event=lambda **kw: Event(next(names)),
            synchronize=Mock(side_effect=lambda: events.append("device_sync")),
            reset_peak_memory_stats=Mock(),
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
            mem_get_info=lambda: (0, 0),
        )
        seq = SimpleNamespace(completion_token_ids=[])

        def step():
            events.append("step")
            seq.completion_token_ids.append(1)
            return [], SimpleNamespace(prefill_tokens=0, decode_tokens=1)

        engine = SimpleNamespace(step=step, scheduler=SimpleNamespace(running=[seq]))
        args = SimpleNamespace(
            input_len=16,
            max_num_seqs=1,
            warmup_steps=0,
            measure_steps=2,
            graph_sizes=(),
        )
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
            patch.object(_probes, "prepare_fixed_decode_batch"),
            patch.object(_probes, "vocabulary_size", return_value=10),
        ):
            metrics, _ = benchmark.run_fixed_decode(engine, args, lambda: 0)

        self.assertEqual(
            events,
            ["device_sync"] + ["start", "step", "end", "event_sync"] * 2,
        )
        self.assertEqual(metrics["cuda_event_step_window_seconds"]["mean"], 0.002)


if __name__ == "__main__":
    unittest.main()
