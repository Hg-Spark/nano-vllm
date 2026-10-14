"""Benchmark, profile, parity, and resource-cleanup experiment contracts."""

import sys
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import bench
from scripts import profile_decode as decode_profile
from scripts import verify_qwen3_5 as parity
from scripts._common import ValidationFailure, require_greedy_parity


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
        self.assertAlmostEqual(
            result["request_throughput_rps"],
            result["requests"] / result["elapsed_seconds"],
        )
        interference = result["long_prompt_interference"]
        self.assertEqual(interference["injected_after_decode_steps"], 1)
        self.assertGreater(interference["long_ttft_seconds"], 0)
        self.assertGreater(interference["short_token_gap_after_seconds"]["p99"], 0)

    def test_require_greedy_parity_rejects_short_generations(self):
        with self.assertRaisesRegex(ValidationFailure, "expected 4 greedy tokens"):
            require_greedy_parity([1, 2], [1, 2], 4, "eager")

    def test_dense_ffn_is_in_decode_hotspot_candidates(self):
        hotspots = decode_profile._profiled_hotspots(
            {"nanovllm::ffn_layer": 1250.0, "nanovllm::gdn_layer": 300.0}
        )
        self.assertEqual(hotspots["ffn"], 1250.0)
        self.assertEqual(max(hotspots, key=hotspots.get), "ffn")

    def test_statistics_interpolation(self):
        summary = bench.summarize([0, 10])
        self.assertEqual(summary["p50"], 5)
        self.assertEqual(summary["p99"], 9.9)
        self.assertEqual(bench.summarize([7]), {"mean": 7, "p50": 7, "p99": 7})

    def test_hf_runner_releases_model_reference_before_empty_cache(self):
        class DummyObj:
            def __getitem__(self, key):
                return SimpleNamespace(tolist=lambda: [7, 8])

        refs = {}

        def make_model(*args, **kwargs):
            m = DummyObj()
            m.generation_config = SimpleNamespace(eos_token_id=None)
            m.cuda = lambda: m
            m.eval = lambda: m

            def gen(**kw):
                out = DummyObj()
                refs["output"] = weakref.ref(out)
                return out

            m.generate = gen
            refs["model"] = weakref.ref(m)
            return m

        def check_empty_cache():
            self.assertIsNone(refs["model"]())
            self.assertIsNone(refs["output"]())

        tok = SimpleNamespace(
            from_pretrained=lambda *a, **kw: lambda prompt, return_tensors: SimpleNamespace(
                input_ids=SimpleNamespace(cuda=lambda: SimpleNamespace(size=lambda d: 2))
            )
        )
        torch_mod = SimpleNamespace(
            float16="fp16",
            inference_mode=nullcontext,
            cuda=SimpleNamespace(empty_cache=Mock(side_effect=check_empty_cache)),
        )
        tf_mod = SimpleNamespace(
            AutoTokenizer=tok,
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=make_model),
        )
        with patch.dict(sys.modules, {"torch": torch_mod, "transformers": tf_mod}):
            self.assertEqual(parity.run_hf("model", "hi", 2, "float16"), [7, 8])
        torch_mod.cuda.empty_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
