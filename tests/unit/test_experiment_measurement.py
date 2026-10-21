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
                    prefill_tokens=8, decode_tokens=0, prefill_seconds=1, decode_seconds=0
                )
            if short.num_completion_tokens < 4:
                short.num_completion_tokens += 1
                short.is_finished = short.num_completion_tokens == 4
                return [], SimpleNamespace(
                    prefill_tokens=0, decode_tokens=1, prefill_seconds=0, decode_seconds=1
                )
            if long in queued and not long.is_finished:
                long.num_completion_tokens = 1
                long.is_finished = True
                return [], SimpleNamespace(
                    prefill_tokens=128, decode_tokens=0, prefill_seconds=1, decode_seconds=0
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
        self.assertGreater(interference["short_token_gap_after_seconds"]["p99"], 0)

    def test_require_greedy_parity_rejects_short_generations(self):
        with self.assertRaisesRegex(ValidationFailure, "expected 4 greedy tokens"):
            require_greedy_parity([1, 2], [1, 2], 4, "eager")
        self.assertTrue(compare_tokens([1, 2], [1, 2])["matched"])
        self.assertEqual(compare_tokens([1, 2], [1, 3])["first_mismatch"], 1)
        self.assertEqual(compare_tokens([1], [1, 2])["first_mismatch"], 1)

    def test_prefix_resume_rejects_matched_empty_outputs(self):
        from nanovllm.engine import cache_runtime
        from scripts.verify_prefix_resume import run_prefix_resume_probe

        entry = SimpleNamespace(num_tokens=64, block_ids=(0, 1, 2, 3), state_snapshot=object())
        cache = SimpleNamespace(snapshot_nbytes=16, find=lambda *args, **kwargs: entry)
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3])

        class StubEngine:
            def __init__(self, *args, max_prefix_cache_entries=0, **kwargs):
                self.cached = max_prefix_cache_entries > 0
                self.calls = 0
                self.scheduler = SimpleNamespace(block_manager=SimpleNamespace(prefix_cache=cache))

            def generate(self, *args, **kwargs):
                self.calls += 1
                if self.cached and self.calls == 2:
                    seq = SimpleNamespace(committed_tokens=64, block_table=list(entry.block_ids))
                    cache_runtime.restore_gdn_state(None, seq, entry.state_snapshot)
                return [{"token_ids": []}]

            def exit(self):
                pass

        with (
            patch("nanovllm.LLM", StubEngine),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch.object(cache_runtime, "restore_gdn_state"),
            patch("torch.cuda.empty_cache"),
            self.assertRaisesRegex(ValidationFailure, "expected 16 greedy tokens"),
        ):
            run_prefix_resume_probe("stub-model", 16, 1024, 16, "float16")

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


if __name__ == "__main__":
    unittest.main()
