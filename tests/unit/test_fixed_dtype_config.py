"""Contracts for fixed BF16 and KV-only quantization benchmark settings."""

import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from benchmarks import run_tests
from nanovllm.config import Config
from scripts import verify_prefix_resume, verify_qwen3_5


class FixedDtypeConfigTest(unittest.TestCase):
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
                    expected = {
                        "hf-parity",
                        "prefix-prompt-parity",
                        "prefix-decode-parity",
                    }
                    targets = [entry for entry in calls if entry[0] in expected]
                    self.assertEqual({name for name, _, _ in targets}, expected)
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

    def test_kv_page_size_128_is_rejected_by_cli_and_runtime(self):
        for block_size in (16, 32, 64):
            args = verify_prefix_resume.parse_args(
                ["/tmp/example-model", "--prefix-block-size", str(block_size)]
            )
            self.assertEqual(args.prefix_block_size, block_size)

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                verify_prefix_resume.parse_args(
                    ["/tmp/example-model", "--prefix-block-size", "128"]
                )
        self.assertEqual(error.exception.code, 2)

        with tempfile.TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config._load_hf_config",
                side_effect=AssertionError("unsupported page size should fail before model load"),
            ) as model_load:
                with self.assertRaisesRegex(ValueError, "kv_cache_block_size must be one of"):
                    Config(model_dir, kv_cache_block_size=128)
                model_load.assert_not_called()

    def test_bench_mode_rejects_duplicate_tail_or_long_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "tests.toml")
            for mode in ("tail", "long"):
                path.write_text(
                    chr(10).join(
                        (
                            'model = "model"',
                            'output_dir = "results"',
                            'stage = "all"',
                            "[benchmark]",
                            f'mode = "{mode}"',
                            "",
                        )
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "benchmark.mode must be smoke/full"):
                    run_tests.load_config(path, root=Path(folder))
                self.assertEqual(
                    run_tests.load_config(path, root=Path(folder), stage=mode)["stage"], mode
                )

    def test_prefix_stage_covers_supported_kv_page_sizes(self):
        config = {
            "model": "/tmp/example-model",
            "gpu": "0",
            "stage": "prefix",
            "benchmark": {"mode": "smoke", "repeats": 1},
            "correctness": {"max_jobs": 1, "nvcc_threads": 1},
        }
        commands = []
        with tempfile.TemporaryDirectory() as folder:
            with patch(
                "benchmarks.run_tests.run_logged",
                side_effect=lambda name, command, output, env: commands.append((name, command)),
            ):
                self.assertEqual(run_tests.execute(config, Path(folder)), 0)

        self.assertEqual(
            [name for name, _ in commands],
            ["nvidia-smi", "prefix-resume-b16", "prefix-resume-b32", "prefix-resume-b64", "prefix"],
        )
        for block_size in (16, 32, 64):
            command = next(cmd for name, cmd in commands if name == f"prefix-resume-b{block_size}")
            self.assertEqual(command[command.index("--prefix-block-size") + 1], str(block_size))


if __name__ == "__main__":
    unittest.main()
