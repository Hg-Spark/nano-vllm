"""CPU-only benchmark summaries, workload comparability, and A/B reporting."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from benchmarks import summarize_bench


class BenchmarkSummaryTest(unittest.TestCase):
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
                "mixed_prefill_limit": 128,
                "activation_reserve_mib": 256,
                "kv_cache_dtype": "fp8",
                "kv_cache_k_scale": 1.0,
                "kv_cache_v_scale": 1.0,
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
        # Runtime versions are part of benchmark comparability, not A/B toggles.
        env = {
            **base["result"],
            "config": base["config"],
            "torch_version": "2.14.0",
            "torch_cuda_version": "13.0",
        }
        self.assertNotEqual(
            summarize_bench.workload_key(env),
            summarize_bench.workload_key({**env, "torch_version": "2.15.0"}),
        )
        self.assertNotEqual(
            summarize_bench.workload_key(env),
            summarize_bench.workload_key({**env, "torch_cuda_version": "13.1"}),
        )

        for field, value in (
            ("mixed_prefill_limit", 0),
            ("activation_reserve_mib", 512),
            ("kv_cache_k_scale", 0.5),
            ("kv_cache_v_scale", 2.0),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(env))
                changed["config"][field] = value
                self.assertNotEqual(
                    summarize_bench.workload_key(env), summarize_bench.workload_key(changed)
                )

        graph = json.loads(json.dumps(base))
        graph["config"]["graph_sizes"] = [2]
        graph["result"]["metrics"]["output_throughput_tok_s"] = 150
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "eager.json").write_text(json.dumps(base), encoding="utf-8")
            graph_file = Path(folder) / "graph.json"
            graph_file.write_text(json.dumps(graph), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                summarize_bench.main([folder, "--min-repeats", "1"])
            self.assertIn("Output throughput improvement: +50.00%", output.getvalue())
            self.assertIn("Successful paired comparisons: 1", output.getvalue())

            # Other experiments in the same folder must not invalidate A/B results.
            (Path(folder) / "parity.json").write_text(
                json.dumps({"experiment": "image_parity", "execution_status": "error"}),
                encoding="utf-8",
            )
            with redirect_stdout(StringIO()):
                summarize_bench.main([folder, "--min-repeats", "1"])

            # An incomplete metric or zero baseline must print n/a, not fail formatting.
            base["result"]["metrics"]["output_throughput_tok_s"] = 0
            base["result"]["metrics"].pop("tpot_seconds")
            (Path(folder) / "eager.json").write_text(json.dumps(base), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                summarize_bench.main([folder, "--min-repeats", "1"])
            self.assertIn("Output throughput improvement: n/a", output.getvalue())
            self.assertIn("tpot_seconds reduction: n/a", output.getvalue())

    def test_prefix_comparison_reports_snapshots_and_peak_memory(self):
        base_config = {
            "scenario": "prefix",
            "model": "model",
            "dtype": "bfloat16",
            "graph_sizes": [],
            "cache_entries": 0,
            "schedule_mode": "mixed",
            "max_num_seqs": 4,
            "budget": 512,
            "seed": 0,
            "shared_prefix_len": 512,
        }

        def result(cache_entries, restored, hit_rate, snapshot_count, memory):
            config = dict(base_config, cache_entries=cache_entries)
            return {
                "experiment": "benchmark_prefix",
                "execution_status": "completed",
                "config": config,
                "result": {
                    "gpu": "test GPU",
                    "compute_capability": [9, 0],
                    "metrics": {
                        "output_throughput_tok_s": 100 if not cache_entries else 120,
                        "ttft_seconds": {"mean": 0.1 if not cache_entries else 0.08},
                        "tpot_seconds": {"mean": 0.02},
                        "peak_cuda_memory_gib": memory,
                    },
                    "details": {
                        "token_hash": "identical",
                        "total_restored_prefix_tokens": restored,
                        "mean_restored_prefix_tokens": restored / 4,
                        "restored_prompt_token_ratio": restored / (4 * 1280),
                        "prefix_hit_rate": hit_rate,
                        "published_snapshots_during_target": snapshot_count,
                        "evicted_snapshots_during_target": 1 if cache_entries else 0,
                        "retained_snapshots_after_target": 1 if cache_entries else 0,
                        "peak_snapshot_bytes_including_setup": 1024**2 if cache_entries else 0,
                        "peak_staging_snapshot_bytes_including_setup": (
                            4 * 1024**2 if cache_entries else 0
                        ),
                        "peak_total_snapshot_bytes_including_setup": (
                            5 * 1024**2 if cache_entries else 0
                        ),
                    },
                },
            }

        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "off.json").write_text(
                json.dumps(result(0, 0, 0.0, 0, 8.0)), encoding="utf-8"
            )
            Path(folder, "on.json").write_text(
                json.dumps(result(32, 1024, 1.0, 2, 7.0)), encoding="utf-8"
            )
            out = StringIO()
            with redirect_stdout(out):
                summarize_bench.main([folder, "--min-repeats", "1"])
            rendered = out.getvalue()
            self.assertIn("Peak CUDA allocated: 8.000 -> 7.000 GiB", rendered)
            self.assertIn(
                "Prefix restored tokens: 1024, mean/request: 256, "
                "hit rate: 1.00, restored/input: 20.0%",
                rendered,
            )
            self.assertIn("Peak CPU GDN snapshot tensors:", rendered)
            self.assertIn("Resident: 1.00 MiB", rendered)
            self.assertIn("Staging: 4.00 MiB", rendered)
            self.assertIn("Total: 5.00 MiB", rendered)
            self.assertIn("published_snapshots_during_target: 2", rendered)
            self.assertIn("retained_snapshots_after_target: 1", rendered)

            # Old prefix JSON is intentionally unsupported: all three
            # distinct CPU snapshot peaks must be recorded, not inferred.
            legacy = result(32, 1024, 1.0, 2, 7.0)
            del legacy["result"]["details"]["peak_total_snapshot_bytes_including_setup"]
            Path(folder, "on.json").write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "missing required prefix snapshot metrics"):
                with redirect_stdout(StringIO()):
                    summarize_bench.main([folder, "--min-repeats", "1"])

    def test_small_interference_p99_is_not_reported(self):
        def result(mode):
            return {
                "experiment": "benchmark_interference",
                "execution_status": "completed",
                "config": {
                    "scenario": "interference",
                    "model": "model",
                    "dtype": "bfloat16",
                    "max_num_seqs": 4,
                    "budget": 512,
                    "graph_sizes": [],
                    "cache_entries": 0,
                    "schedule_mode": mode,
                },
                "result": {
                    "gpu": "test GPU",
                    "compute_capability": [9, 0],
                    "metrics": {
                        "output_throughput_tok_s": 100,
                        "tpot_seconds": {"mean": 0.01},
                        "long_prompt_interference": {
                            "long_ttft_seconds": 0.2,
                            "short_token_gap_after_seconds": {"mean": 0.02, "p99": None},
                            "short_token_gap_after_samples": 20,
                        },
                    },
                    "details": {"token_hash": "same"},
                },
            }

        with tempfile.TemporaryDirectory() as folder:
            for mode in ("mixed", "exclusive_decode_first"):
                Path(folder, mode + ".json").write_text(json.dumps(result(mode)), encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                summarize_bench.main([folder, "--min-repeats", "1"])
            self.assertIn("p99=n/a ms", out.getvalue())
            self.assertIn("Synthetic P99 ITL comparison withheld", out.getvalue())

    def test_interference_reuses_exclusive_baseline_across_caps(self):
        def record(mode, cap, budget=512, token_hash="matching"):
            return {
                "experiment": "benchmark_interference",
                "execution_status": "completed",
                "config": {
                    "scenario": "interference",
                    "model": "model",
                    "dtype": "bfloat16",
                    "max_num_seqs": 4,
                    "budget": budget,
                    "mixed_prefill_limit": cap,
                    "graph_sizes": [],
                    "cache_entries": 0,
                    "schedule_mode": mode,
                    "seed": 0,
                },
                "result": {
                    "gpu": "test GPU",
                    "compute_capability": [9, 0],
                    "metrics": {
                        "output_throughput_tok_s": 100 if mode == "exclusive_decode_first" else 120,
                        "tpot_seconds": {"mean": 0.01},
                        "long_prompt_interference": {
                            "long_ttft_seconds": 0.2,
                            "short_token_gap_after_seconds": {"mean": 0.02, "p99": None},
                            "short_token_gap_after_samples": 20,
                        },
                    },
                    "details": {"token_hash": token_hash},
                },
            }

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "exclusive.json").write_text(
                json.dumps(record("exclusive_decode_first", -1)), encoding="utf-8"
            )
            # Budget sweep treatment and three independent mixed-prefill limits.
            for cap in (-1, 64, 128, 256):
                (root / f"mixed_{cap}.json").write_text(
                    json.dumps(record("mixed", cap)), encoding="utf-8"
                )
            out = StringIO()
            with redirect_stdout(out):
                summarize_bench.main([folder, "--min-repeats", "1"])
            self.assertIn("Successful paired comparisons: 4", out.getvalue())
            self.assertIn("cap=64", out.getvalue())
            self.assertIn("cap=256", out.getvalue())

            # A reused baseline must not silently cross incompatible prompt/output tokens.
            (root / "mixed_64.json").write_text(
                json.dumps(record("mixed", 64, token_hash="wrong")), encoding="utf-8"
            )
            with self.assertRaisesRegex(SystemExit, "PARITY FAILURE"):
                with redirect_stdout(StringIO()):
                    summarize_bench.main([folder, "--min-repeats", "1"])

            # An unrelated budget cannot use the shared exclusive baseline.
            (root / "mixed_64.json").write_text(
                json.dumps(record("mixed", 64, budget=256)), encoding="utf-8"
            )
            with self.assertRaisesRegex(SystemExit, "Some runs could not be compared"):
                with redirect_stdout(StringIO()):
                    summarize_bench.main([folder, "--min-repeats", "1"])


if __name__ == "__main__":
    unittest.main()
