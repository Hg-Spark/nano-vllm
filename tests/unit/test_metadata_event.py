"""Pinned metadata has a valid transfer event even without a prior H2D upload."""

import unittest
from unittest.mock import Mock, patch

try:
    import torch  # noqa: F401 -- optional dependency check
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.engine.batch_common import PinnedMetadataPool


class PinnedMetadataEventTest(unittest.TestCase):
    def test_plan_transfer_initializes_event_without_to_cuda(self):
        pool = PinnedMetadataPool()
        event = Mock()
        with (
            patch("torch.cuda.Event", return_value=event) as make_event,
            patch("torch.cuda.current_stream", return_value="stream"),
        ):
            pool.record_plan_transfer()
            make_event.assert_called_once_with()
            event.record.assert_called_once_with("stream")
            self.assertTrue(pool._pending)
            pool.clear()
            event.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
