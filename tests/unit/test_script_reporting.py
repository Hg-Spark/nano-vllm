"""Result file reporting checks."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts._common import ValidationFailure, execute, write_result


class ReportingTest(unittest.TestCase):
    def test_execute_writes_completed_and_failed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            args = SimpleNamespace(result_file=str(path), model="checkpoint")
            self.assertEqual(execute("bench", args, lambda _: {"ok": 1}), 0)
            record = json.loads(path.read_text())
            self.assertEqual(record["execution_status"], "completed")
            self.assertEqual(record["result"], {"ok": 1})

            fail_action = Mock(side_effect=ValidationFailure("mismatch", {"tokens": [1]}))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(execute("parity", args, fail_action), 1)
            self.assertEqual(json.loads(path.read_text())["execution_status"], "validation_failed")

    def test_failed_replace_or_nonfinite_json_preserves_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_result(path, {"ok": True})
            with self.assertRaises(ValueError):
                write_result(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})

            with patch("scripts._common.os.replace", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    write_result(path, {"new": 1})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
