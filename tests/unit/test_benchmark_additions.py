"""CPU-only regression checks for checkpoint instrumentation and A/B comparability."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from benchmarks.bench_experiments import checkpoint_schedule_stats
from benchmarks import summarize_bench
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import make_scheduler


class BenchmarkAdditionsTest(unittest.TestCase):
    def test_offgrid_checkpoint_recovery_statistics_from_real_scheduler(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=64,
            max_num_seqs=2,
            num_blocks=256,
            block_size=16,
        )
        running = Sequence([1, 2], SamplingParams(max_tokens=100, ignore_eos=True))
        scheduler.block_manager.ensure_capacity(running, len(running))
        scheduler.state_manager.allocate(running)
        running.committed_tokens = len(running) - 1
        scheduler.running.append(running)
        pending = Sequence(list(range(3500)), SamplingParams(max_tokens=1))
        scheduler.add(pending)

        first = scheduler.schedule()
        self.assertEqual(
            (first.prefill_chunks[0].start, first.prefill_chunks[0].end), (0, 63)
        )
        self.assertEqual(
            checkpoint_schedule_stats(first.prefill_chunks, block_size=16),
            {
                "prefill_chunks": 1,
                "checkpoint_chunks": 0,
                "checkpoint_positions": 0,
                "offgrid_prefill_chunks": 0,
                "offgrid_realignments": 0,
            },
        )
        scheduler.postprocess(first.decode_chunks, [7], False)
        scheduler.postprocess(first.prefill_chunks, [None], True)

        second = scheduler.schedule()
        self.assertEqual(
            (second.prefill_chunks[0].start, second.prefill_chunks[0].end), (63, 64)
        )
        self.assertEqual(second.prefill_chunks[0].checkpoint_positions, (64,))
        self.assertEqual(
            checkpoint_schedule_stats(second.prefill_chunks, block_size=16),
            {
                "prefill_chunks": 1,
                "checkpoint_chunks": 1,
                "checkpoint_positions": 1,
                "offgrid_prefill_chunks": 1,
                "offgrid_realignments": 1,
            },
        )

    def test_checkpoint_stats_require_joint_kv_and_gdn_grid(self):
        chunks = (
            SimpleNamespace(start=63, end=64, checkpoint_positions=(64,)),
            SimpleNamespace(start=127, end=128, checkpoint_positions=(128,)),
            SimpleNamespace(start=128, end=255, checkpoint_positions=()),
        )
        stats = checkpoint_schedule_stats(chunks, block_size=128)
        self.assertEqual(stats["checkpoint_positions"], 2)
        self.assertEqual(stats["offgrid_prefill_chunks"], 2)
        self.assertEqual(stats["offgrid_realignments"], 1)

    def test_workload_key_includes_prefill_and_memory_budget_controls(self):
        sample = {
            "gpu": "test GPU",
            "compute_capability": [9, 0],
            "config": {
                "scenario": "interference",
                "graph_sizes": [],
                "cache_entries": 0,
                "schedule_mode": "mixed",
                "mixed_prefill_limit": 128,
                "activation_reserve_mib": 256,
            },
        }
        original = summarize_bench.workload_key(sample)
        for option, changed in (("mixed_prefill_limit", 0), ("activation_reserve_mib", 512)):
            with self.subTest(option=option):
                different = json.loads(json.dumps(sample))
                different["config"][option] = changed
                self.assertNotEqual(original, summarize_bench.workload_key(different))

    def test_prefix_comparison_reports_checkpoint_and_peak_memory(self):
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
        }

        def result(cache_entries, restored, hit_rate, checkpoint_count, memory):
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
                        "prefix_hit_rate": hit_rate,
                        "target_scheduled_checkpoint_stats": {
                            "checkpoint_positions": checkpoint_count,
                        },
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
            self.assertIn("Prefix restored tokens: 1024, hit rate: 1.00", rendered)
            self.assertIn("Target scheduled checkpoints", rendered)


if __name__ == "__main__":
    unittest.main()
