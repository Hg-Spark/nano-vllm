"""CPU-only benchmark parity and comparable A/B result checks."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from benchmarks import summarize_bench as summary


class BenchmarkExperimentsTest(unittest.TestCase):
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
