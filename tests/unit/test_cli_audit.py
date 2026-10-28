"""CLI validation must reject bad numeric options without loading models."""

import tempfile
import unittest
from pathlib import Path

from benchmarks.bench_experiments import arguments as benchmark_args
from scripts.chat import parse_args as chat_args
from scripts.verify_qwen3_5 import parse_args as parity_args


class CLIAuditTest(unittest.TestCase):
    def test_chat_invalid_gpu_utilization_and_batch_budget(self):
        for flag, value in (
            ("--gpu-memory-utilization", "0"),
            ("--gpu-memory-utilization", "1.2"),
            ("--max-num-batched-tokens", "0"),
            ("--max-model-len", "-1"),
        ):
            with self.subTest(flag=flag, value=value):
                with self.assertRaises(SystemExit):
                    chat_args(["unused-model", flag, value])

    def test_parity_accepts_borderline_generation_count_for_runtime_check(self):
        args = parity_args(
            [
                "unused-model",
                "--max-model-len",
                "16",
                "--max-new-tokens",
                "16",
            ]
        )
        self.assertEqual(args.max_new_tokens, args.max_model_len)

    def test_benchmark_rejects_nonpositive_batch_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            result = str(Path(folder) / "result.json")
            for flag, value in (
                ("--max-num-seqs", "0"),
                ("--budget", "0"),
                ("--gpu-memory-utilization", "0"),
                ("--gpu-memory-utilization", "1.2"),
            ):
                with self.subTest(flag=flag, value=value):
                    with self.assertRaises(SystemExit):
                        benchmark_args(
                            [
                                "unused-model",
                                "--scenario",
                                "throughput",
                                "--out",
                                result,
                                flag,
                                value,
                            ]
                        )


if __name__ == "__main__":
    unittest.main()
