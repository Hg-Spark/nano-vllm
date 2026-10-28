"""Correctness and lifetime tests for packed GDN checkpoint restoration."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from nanovllm.engine.cache_runtime import GDNRestoreStaging, restore_pending_states
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.state_manager import GDNStateSnapshot


@unittest.skipIf(torch is None, "requires torch")
class GDNRestoreStagingTest(unittest.TestCase):
    class Layer:
        def __init__(self, device):
            self.conv_state = torch.zeros(2, 4, 3, device=device, dtype=torch.bfloat16)
            self.recurrent_state = torch.zeros(
                2, 2, 4, 4, device=device, dtype=torch.float32
            )

        def slot(self, slot_id):
            return self.conv_state[slot_id], self.recurrent_state[slot_id]

        def restore_state_slot(self, slot_id, snapshot):
            conv, recurrent = self.slot(slot_id)
            conv.copy_(snapshot[0])
            recurrent.copy_(snapshot[1])

    @staticmethod
    def pending_sequence(slot, layers, offset):
        seq = Sequence([1] * 64)
        seq.committed_tokens = 64
        seq.state_slot = slot
        states = tuple(
            (
                torch.full((4, 3), float(offset + layer), dtype=torch.bfloat16),
                torch.full((2, 4, 4), float(offset + layer), dtype=torch.float32),
            )
            for layer in range(layers)
        )
        seq.pending_state_snapshot = GDNStateSnapshot(64, states)
        return seq

    def test_cpu_fallback_preserves_exact_state(self):
        from types import SimpleNamespace

        layers = [self.Layer("cpu") for _ in range(4)]
        model = SimpleNamespace(state_cache_modules=lambda: layers)
        seq = self.pending_sequence(1, len(layers), 5)
        stage = GDNRestoreStaging()
        restore_pending_states(model, [seq], stage)
        self.assertIsNone(seq.pending_state_snapshot)
        for i, layer in enumerate(layers):
            torch.testing.assert_close(
                layer.conv_state[1], torch.full((4, 3), 5 + i, dtype=torch.bfloat16)
            )
            torch.testing.assert_close(
                layer.recurrent_state[1], torch.full((2, 4, 4), 5 + i)
            )

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_pinned_batch_restores_multiple_slots_and_reuses_safely(self):
        from types import SimpleNamespace

        layers = [self.Layer("cuda") for _ in range(18)]
        model = SimpleNamespace(state_cache_modules=lambda: layers)
        staging = GDNRestoreStaging(max_staging_bytes=8 * 1024**2)
        try:
            first = self.pending_sequence(0, len(layers), 10)
            second = self.pending_sequence(1, len(layers), 20)
            restore_pending_states(model, [first, second], staging)
            self.assertTrue(staging._conv_host.is_pinned())
            self.assertTrue(staging._recurrent_host.is_pinned())
            self.assertIsNone(first.pending_state_snapshot)
            self.assertIsNone(second.pending_state_snapshot)
            torch.cuda.synchronize()
            for index, layer in enumerate(layers):
                for slot, offset in ((0, 10), (1, 20)):
                    torch.testing.assert_close(
                        layer.conv_state[slot].cpu(),
                        torch.full((4, 3), offset + index, dtype=torch.bfloat16),
                    )
                    torch.testing.assert_close(
                        layer.recurrent_state[slot].cpu(),
                        torch.full((2, 4, 4), offset + index),
                    )

            third = self.pending_sequence(0, len(layers), 40)
            restore_pending_states(model, [third], staging)
            torch.cuda.synchronize()
            self.assertEqual(layers[0].conv_state[0, 0, 0].item(), 40)
            self.assertEqual(layers[0].conv_state[1, 0, 0].item(), 20)
        finally:
            staging.clear()

    def test_wrong_checkpoint_boundary_keeps_snapshot_pending(self):
        from types import SimpleNamespace

        layers = [self.Layer("cpu")]
        seq = self.pending_sequence(0, 1, 1)
        seq.committed_tokens = 63
        with self.assertRaisesRegex(RuntimeError, "restore boundary mismatch"):
            restore_pending_states(
                SimpleNamespace(state_cache_modules=lambda: layers),
                [seq],
                GDNRestoreStaging(),
            )
        self.assertIsNotNone(seq.pending_state_snapshot)


if __name__ == "__main__":
    unittest.main()
