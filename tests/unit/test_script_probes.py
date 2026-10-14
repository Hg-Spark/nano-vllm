"""Script adapters preserve the original operation order without CUDA imports."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import _probes


class ProbeTest(unittest.TestCase):
    def test_preparation_and_enqueue_require_idle_engine(self):
        engine = SimpleNamespace(is_finished=lambda: False)
        for operation in (_probes.prepare_requests, _probes.enqueue_sequences):
            with self.subTest(operation=operation), self.assertRaisesRegex(RuntimeError, "idle"):
                if operation is _probes.prepare_requests:
                    operation(engine, [], None)
                else:
                    operation(engine, [])

    def test_fixed_batch_prefills_independently_then_restores_order(self):
        events = []
        requests = []
        scheduler = SimpleNamespace(waiting=deque(), running=deque())

        def add_request(prompt, sampling):
            self.assertFalse(scheduler.running)
            seq = object()
            requests.append(seq)
            events.append(("add", seq))
            scheduler.waiting.append(seq)

        def schedule():
            seq = scheduler.waiting.popleft()
            events.append(("schedule", seq))
            return SimpleNamespace(decode_chunks=(), prefill_chunks=(seq,))

        def run_batch(chunks, prefill):
            self.assertTrue(prefill)
            events.append(("run", chunks[0]))
            scheduler.running.append(chunks[0])

        scheduler.schedule = schedule
        engine = SimpleNamespace(
            is_finished=lambda: True,
            scheduler=scheduler,
            add_request=add_request,
            _run_batch=run_batch,
        )
        _probes.prepare_fixed_decode_batch(engine, [1], object(), 3)
        self.assertEqual(len(requests), 3)
        self.assertEqual(list(scheduler.running), requests)
        self.assertEqual([event[0] for event in events], ["add", "schedule", "run"] * 3)

    def test_fixed_batch_rejects_decode_in_setup_without_running_it(self):
        scheduler = SimpleNamespace(
            waiting=[1],
            running=deque(),
            schedule=Mock(
                return_value=SimpleNamespace(decode_chunks=("decode",), prefill_chunks=())
            ),
        )
        engine = SimpleNamespace(
            is_finished=lambda: True, scheduler=scheduler, add_request=Mock(), _run_batch=Mock()
        )
        with self.assertRaisesRegex(RuntimeError, "decode work"):
            _probes.prepare_fixed_decode_batch(engine, [1], object(), 1)
        engine._run_batch.assert_not_called()

    def test_prefix_metadata_does_not_expose_mutable_entry(self):
        entry = SimpleNamespace(
            num_tokens=16,
            state_snapshot=SimpleNamespace(
                layers=[[SimpleNamespace(numel=lambda: 5, element_size=lambda: 2)]]
            ),
        )
        finder = Mock(return_value=entry)
        engine = SimpleNamespace(
            scheduler=SimpleNamespace(block_manager=SimpleNamespace(find_prefix=finder))
        )
        with patch.dict(
            "sys.modules",
            {"nanovllm.engine.sequence": SimpleNamespace(Sequence=Mock())},
        ):
            result = _probes.find_prefix_metadata(engine, [1, 2])
            self.assertEqual(result, {"num_tokens": 16, "snapshot_bytes": 10})
            finder.return_value = None
            self.assertIsNone(_probes.find_prefix_metadata(engine, [1, 2]))


if __name__ == "__main__":
    unittest.main()
