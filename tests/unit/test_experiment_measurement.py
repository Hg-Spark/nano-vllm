"""Benchmark timing, measurement units, and token-parity contracts."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import bench
from scripts._common import ValidationFailure, compare_tokens, require_greedy_parity


class MeasurementTest(unittest.TestCase):
    def test_benchmark_observation_boundaries_and_units(self):
        seq = SimpleNamespace(
            seq_id=1,
            num_completion_tokens=0,
            num_prompt_tokens=2,
            is_finished=False,
        )
        queued = []

        def step():
            prefill = seq.num_completion_tokens == 0
            seq.num_completion_tokens = 1 if prefill else 3
            seq.is_finished = not prefill
            return [], SimpleNamespace(
                prefill_tokens=2 if prefill else 0,
                decode_tokens=0 if prefill else 2,
                prefill_seconds=1.0 if prefill else 0.0,
                decode_seconds=0.0 if prefill else 1.0,
                snapshot_capture_seconds=0.125 if prefill else 0.25,
            )

        engine = SimpleNamespace(
            is_finished=lambda: not queued or seq.is_finished,
            step=step,
            scheduler=SimpleNamespace(add=lambda item: queued.append(item)),
        )
        cuda = SimpleNamespace(
            synchronize=Mock(),
            reset_peak_memory_stats=Mock(),
            max_memory_allocated=lambda: 1024**3,
            max_memory_reserved=lambda: 2 * 1024**3,
            mem_get_info=lambda: (3 * 1024**3, 10 * 1024**3),
        )
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
            patch.object(bench.time, "perf_counter", side_effect=[0, 1, 2, 3]),
        ):
            result = bench.run_benchmark(engine, [seq])
        self.assertEqual(result["ttft_seconds"]["mean"], 1)
        self.assertEqual(result["e2e_latency_seconds"]["mean"], 2)
        self.assertEqual(result["tpot_seconds"]["mean"], 0.5)
        self.assertEqual(result["prefill_host_call_tok_s"], 2)
        self.assertEqual(result["decode_host_call_tok_s"], 2)
        self.assertAlmostEqual(result["snapshot_capture_seconds"], 0.375)
        self.assertEqual(result["decode_forward_calls"], 1)
        self.assertEqual(result["ttft_samples"], 1)
        self.assertIsNone(result["ttft_seconds"]["p99"])
        self.assertEqual(cuda.synchronize.call_count, 3)

    def test_long_prompt_interference_records_arrival_relative_latency(self):
        short = SimpleNamespace(
            seq_id=1, num_completion_tokens=0, num_prompt_tokens=8, is_finished=False
        )
        long = SimpleNamespace(
            seq_id=2, num_completion_tokens=0, num_prompt_tokens=128, is_finished=False
        )
        queued = []

        def step():
            if short.num_completion_tokens == 0:
                short.num_completion_tokens = 1
                return [], SimpleNamespace(
                    prefill_tokens=8,
                    decode_tokens=0,
                    prefill_seconds=1,
                    decode_seconds=0,
                    snapshot_capture_seconds=0.0,
                )
            if short.num_completion_tokens < 4:
                short.num_completion_tokens += 1
                short.is_finished = short.num_completion_tokens == 4
                return [], SimpleNamespace(
                    prefill_tokens=0,
                    decode_tokens=1,
                    prefill_seconds=0,
                    decode_seconds=1,
                    snapshot_capture_seconds=0.0,
                )
            if long in queued and not long.is_finished:
                long.num_completion_tokens = 1
                long.is_finished = True
                return [], SimpleNamespace(
                    prefill_tokens=128,
                    decode_tokens=0,
                    prefill_seconds=1,
                    decode_seconds=0,
                    snapshot_capture_seconds=0.0,
                )
            raise AssertionError("benchmark called step without work")

        engine = SimpleNamespace(
            scheduler=SimpleNamespace(add=lambda seq: queued.append(seq)),
            is_finished=lambda: not queued or all(seq.is_finished for seq in queued),
            step=step,
        )
        cuda = SimpleNamespace(
            synchronize=Mock(),
            reset_peak_memory_stats=Mock(),
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
            mem_get_info=lambda: (0, 0),
        )
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
            patch.object(bench.time, "perf_counter", side_effect=range(100)),
        ):
            result = bench.run_benchmark(
                engine, [short], late_sequence=long, late_after_decode_steps=1
            )

        self.assertEqual(result["requests"], 2)
        interference = result["long_prompt_interference"]
        self.assertEqual(interference["injected_after_decode_steps"], 1)
        self.assertGreater(interference["long_ttft_seconds"], 0)
        self.assertGreater(interference["short_token_gap_after_seconds"]["mean"], 0)
        self.assertIsNone(interference["short_token_gap_after_seconds"]["p99"])
        self.assertLess(interference["short_token_gap_after_samples"], 1000)

    def test_require_greedy_parity_rejects_short_generations(self):
        with self.assertRaisesRegex(ValidationFailure, "expected 4 greedy tokens"):
            require_greedy_parity([1, 2], [1, 2], 4, "eager")
        self.assertTrue(compare_tokens([1, 2], [1, 2])["matched"])
        self.assertEqual(compare_tokens([1, 2], [1, 3])["first_mismatch"], 1)
        self.assertEqual(compare_tokens([1], [1, 2])["first_mismatch"], 1)

    def test_prefix_resume_rejects_invalid_greedy_outputs(self):
        from scripts.verify_prefix_resume import validate_prefix_resume_result

        for actual, expected, target_len, message in (
            ([], [], 16, "expected 16 greedy tokens"),
            ([10, 20], [10, 30], 2, "token 1"),
        ):
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaisesRegex(ValidationFailure, message):
                    validate_prefix_resume_result(
                        actual, expected, target_len, {"num_tokens": 64}, 64
                    )

    def test_logits_comparison_detects_numeric_divergence(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")

        from scripts.verify_qwen3_5 import compare_logits

        hf = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(compare_logits([hf, hf], [hf, hf], 0.05)["max_relative_rms"], 0)
        with self.assertRaisesRegex(ValidationFailure, "logits at step 1"):
            compare_logits([hf, hf], [hf, hf + 1], 0.05)
        with self.assertRaisesRegex(ValidationFailure, "non-finite"):
            compare_logits([hf], [torch.tensor([1.0, float("nan"), 3.0])], 0.05)

    def test_statistics_interpolation(self):
        summary = bench.summarize([0, 10])
        self.assertEqual(summary["p50"], 5)
        self.assertEqual(summary["p99"], 9.9)
        self.assertEqual(bench.summarize([7]), {"mean": 7, "p50": 7, "p99": 7})
        self.assertIsNone(bench.summarize([0, 10], p99_min_samples=1000)["p99"])
        self.assertEqual(bench.summarize([1.0] * 1000, p99_min_samples=1000)["p99"], 1.0)


if __name__ == "__main__":
    unittest.main()
