"""Contracts for fixed BF16 and KV-only quantization benchmark settings."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import run_tests
from nanovllm.config import Config
from scripts import verify_qwen3_5, verify_prefix_resume


class FixedDtypeConfigTest(unittest.TestCase):
    def test_model_dtype_is_implementation_detail(self):
        self.assertNotIn("dtype", inspect.signature(Config).parameters)
        args = verify_qwen3_5.parse_args(["/tmp/example-model"])
        self.assertFalse(hasattr(args, "dtype"))

    def test_correctness_cli_exposes_kv_precision_not_model_precision(self):
        for parser in (verify_qwen3_5.parse_args, verify_prefix_resume.parse_args):
            args = parser(
                [
                    "/tmp/example-model",
                    "--kv-cache-dtype",
                    "fp8_e4m3",
                    "--kv-cache-k-scale",
                    "0.25",
                    "--kv-cache-v-scale",
                    "0.5",
                ]
            )
            self.assertFalse(hasattr(args, "dtype"))
            self.assertEqual(args.kv_cache_dtype, "fp8_e4m3")
            self.assertEqual(args.kv_cache_k_scale, 0.25)
            self.assertEqual(args.kv_cache_v_scale, 0.5)

    def test_config_dispatch_preserves_all_stage_kv_options(self):
        config = {
            "model": "/tmp/example-model",
            "output_dir": "/tmp/results",
            "gpu": "0",
            "stage": "check",
            "benchmark": {"mode": "smoke", "repeats": 3},
            "kv_cache": {"dtype": "fp8_e4m3", "k_scale": 0.25, "v_scale": 0.5},
            "correctness": {
                "max_jobs": 1,
                "nvcc_threads": 1,
                "max_new_tokens": 8,
                "max_model_len": 512,
                "prompt": "Hello",
                "prefix_block_size": 16,
            },
            "capacity": {
                "max_batch": 2,
                "budget": 32,
                "input_len": 64,
                "output_len": 16,
                "max_model_len": 128,
                "gpu_memory_utilization": 0.75,
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for stage in ("check", "bench", "capacity"):
                calls = []

                def record(name, command, destination, env):
                    calls.append((name, command, env))

                with patch("benchmarks.run_tests.run_logged", side_effect=record):
                    config["stage"] = stage
                    self.assertEqual(run_tests.execute(config, output), 0)

                self.assertEqual(calls[0][0], "nvidia-smi")
                self.assertEqual(calls[0][2]["REPEATS"], "3")
                self.assertEqual(calls[0][2]["NANOVLLM_KV_CACHE_DTYPE"], "fp8_e4m3")
                self.assertEqual(calls[0][2]["NANOVLLM_KV_CACHE_K_SCALE"], "0.25")
                self.assertEqual(calls[0][2]["NANOVLLM_KV_CACHE_V_SCALE"], "0.5")
                if stage == "check":
                    targets = [
                        entry for entry in calls if entry[0] in ("hf-parity", "prefix-parity")
                    ]
                elif stage == "capacity":
                    targets = [entry for entry in calls if entry[0] == "capacity"]
                else:
                    targets = []
                    self.assertEqual(calls[-1][1][-1], "smoke")
                for _, command, _ in targets:
                    self.assertNotIn("--dtype", command)
                    self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "fp8_e4m3")
                    self.assertEqual(command[command.index("--kv-cache-k-scale") + 1], "0.25")
                    self.assertEqual(command[command.index("--kv-cache-v-scale") + 1], "0.5")


if __name__ == "__main__":
    unittest.main()
