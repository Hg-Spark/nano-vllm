"""Concise CLI parsing validation without GPU runtime imports."""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from benchmarks import bench_experiments as experiments
from scripts import (
    bench,
    chat,
)
from scripts import (
    diagnose_qwen3_5 as diagnose,
)
from scripts import (
    profile_decode as profile,
)
from scripts import (
    verify_prefix_resume as prefix,
)
from scripts import (
    verify_qwen3_5 as parity,
)
from scripts import (
    verify_qwen3_5_multimodal as image,
)


class CliTest(unittest.TestCase):
    def test_dtype_must_be_explicit_and_valid(self):
        for script, argv in (
            (chat, ["model"]),
            (bench, ["model"]),
            (profile, ["model"]),
            (diagnose, ["model"]),
            (parity, ["model"]),
            (prefix, ["model"]),
            (image, ["model", "image"]),
        ):
            with self.subTest(script=script.__name__):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    script.parse_args(argv)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    script.parse_args([*argv, "--dtype", "auto"])
                self.assertEqual(script.parse_args([*argv, "--dtype", "float16"]).dtype, "float16")

    def test_invalid_numeric_bounds_fail_fast(self):
        for script, argv in (
            (bench, ["model", "--max-num-seqs", "0"]),
            (profile, ["model", "--warmup-steps", "-1"]),
            (diagnose, ["model", "--relative-rms-threshold", "nan"]),
            (parity, ["model", "--max-new-tokens", "0"]),
            (chat, ["model", "--temperature", "-1"]),
        ):
            with (
                self.subTest(script=script.__name__),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                script.parse_args([*argv, "--dtype", "float16"])

    def test_graph_bucket_argument(self):
        self.assertEqual(experiments.parse_graph_sizes("4,1,2,2"), (1, 2, 4))
        with self.assertRaises(ValueError):
            experiments.parse_graph_sizes("0")

    def test_environment_model_and_explicit_override(self):
        with patch.dict(os.environ, {"NANOVLLM_MODEL": "~/environment-model"}):
            self.assertEqual(
                chat.parse_args(["--dtype", "float16"]).model,
                os.path.expanduser("~/environment-model"),
            )
            self.assertEqual(chat.parse_args(["explicit", "--dtype", "float16"]).model, "explicit")


if __name__ == "__main__":
    unittest.main()
