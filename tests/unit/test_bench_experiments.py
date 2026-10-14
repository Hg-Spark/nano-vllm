"""CPU-only contracts for the focused hybrid benchmark experiments."""

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
    def test_graph_sizes_and_output_parity(self):
        self.assertEqual(benchmark.parse_graph_sizes("4,1,2,2"), (1, 2, 4))
        self.assertEqual(benchmark.parse_graph_sizes(""), ())
        with self.assertRaises(ValueError):
            benchmark.parse_graph_sizes("0")
        self.assertNotEqual(
            benchmark.tokens_hash([SimpleNamespace(completion_token_ids=[1, 2])]),
            benchmark.tokens_hash([SimpleNamespace(completion_token_ids=[1, 3])]),
        )

    def test_summary_rejects_output_mismatch_before_speedups(self):
        base = {
            "status": "completed",
            "gpu": "GPU",
            "compute_capability": [9, 0],
            "git": {"commit": "abc", "dirty": False},
            "script_sha256": "same",
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
            "details": {"token_hash": "original"},
        }
        changed = json.loads(json.dumps(base))
        changed["parameters"]["graph_sizes"] = [2]
        changed["details"]["token_hash"] = "different"
        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in (("base.json", base), ("graph.json", changed)):
                (Path(tmp) / name).write_text(json.dumps(payload), encoding="utf-8")
            with (
                redirect_stdout(StringIO()),
                self.assertRaisesRegex(SystemExit, "Output token hashes differ"),
            ):
                summary.main([tmp])

    def test_fixed_decode_cuda_events_do_not_sync_before_end_record(self):
        events = []

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self):
                events.append(self.name + "_record")

            def synchronize(self):
                events.append(self.name + "_sync")

            def elapsed_time(self, other):
                return 2.0

        event_number = iter(("start", "end"))
        cuda = SimpleNamespace(
            Event=lambda **kwargs: Event(next(event_number)),
            synchronize=Mock(side_effect=lambda: events.append("device_sync")),
            reset_peak_memory_stats=Mock(),
            max_memory_allocated=lambda: 1024,
            max_memory_reserved=lambda: 1024,
            mem_get_info=lambda: (1024, 2048),
        )
        sequences = [SimpleNamespace(completion_token_ids=[]) for _ in range(2)]

        def step():
            events.append("engine_step")
            for seq in sequences:
                seq.completion_token_ids.append(1)
            return [], SimpleNamespace(prefill_tokens=0, decode_tokens=2)

        engine = SimpleNamespace(step=step, scheduler=SimpleNamespace(running=sequences))
        args = SimpleNamespace(
            input_len=16,
            max_num_seqs=2,
            warmup_steps=1,
            measure_steps=3,
            graph_sizes=(),
        )
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
            patch.object(_probes, "prepare_fixed_decode_batch"),
            patch.object(_probes, "vocabulary_size", return_value=10),
        ):
            metrics, _ = benchmark.run_fixed_decode(engine, args, lambda: 0)

        self.assertEqual(cuda.synchronize.call_count, 1)
        self.assertEqual(events.count("start_record"), 3)
        self.assertEqual(events.count("end_record"), 3)
        self.assertEqual(events.count("end_sync"), 3)
        self.assertEqual(metrics["cuda_event_step_window_seconds"]["mean"], 0.002)
        for idx, event in enumerate(events):
            if event == "end_record":
                self.assertEqual(events[idx - 1], "engine_step")


if __name__ == "__main__":
    unittest.main()
