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
            self.recurrent_state = torch.zeros(2, 2, 4, 4, device=device, dtype=torch.float32)

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

    def assert_restored_slots(self, layers, slot_offsets):
        for index, layer in enumerate(layers):
            for slot, offset in slot_offsets.items():
                value = float(offset + index)
                torch.testing.assert_close(
                    layer.conv_state[slot].cpu(),
                    torch.full((4, 3), value, dtype=torch.bfloat16),
                )
                torch.testing.assert_close(
                    layer.recurrent_state[slot].cpu(),
                    torch.full((2, 4, 4), value, dtype=torch.float32),
                )

    def test_cpu_fallback_preserves_exact_state(self):
        from types import SimpleNamespace

        layers = [self.Layer("cpu") for _ in range(4)]
        model = SimpleNamespace(state_cache_modules=lambda: layers)
        seq = self.pending_sequence(1, len(layers), 5)
        stage = GDNRestoreStaging()
        restore_pending_states(model, [seq], stage)
        self.assertIsNone(seq.pending_state_snapshot)
        self.assert_restored_slots(layers, {1: 5})

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
            self.assert_restored_slots(layers, {0: 10, 1: 20})

            third = self.pending_sequence(0, len(layers), 40)
            restore_pending_states(model, [third], staging)
            self.assertIsNone(third.pending_state_snapshot)
            torch.cuda.synchronize()
            self.assert_restored_slots(layers, {0: 40, 1: 20})
        finally:
            staging.clear()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_pinned_staging_reuse_without_sync_across_groups(self):
        from types import SimpleNamespace

        layers = [self.Layer("cuda") for _ in range(18)]
        model = SimpleNamespace(state_cache_modules=lambda: layers)
        first = self.pending_sequence(0, len(layers), 10)
        second = self.pending_sequence(1, len(layers), 20)
        third = self.pending_sequence(0, len(layers), 40)
        snapshot_bytes = sum(
            conv.numel() * conv.element_size() + recurrent.numel() * recurrent.element_size()
            for conv, recurrent in first.pending_state_snapshot.layers
        )
        # Only one snapshot fits: the first restore must split into two groups.
        staging = GDNRestoreStaging(max_staging_bytes=snapshot_bytes)
        try:
            restore_pending_states(model, [first, second], staging)
            # No synchronization between restores; pinned storage must be safe to reuse.
            restore_pending_states(model, [third], staging)
            for seq in (first, second, third):
                self.assertIsNone(seq.pending_state_snapshot)
            self.assertEqual(staging._conv_host.shape[0], len(layers))
            self.assertTrue(staging._conv_host.is_pinned())
            self.assertTrue(staging._recurrent_host.is_pinned())
            torch.cuda.synchronize()
            self.assert_restored_slots(layers, {0: 40, 1: 20})
        finally:
            staging.clear()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_failed_gpu_scatter_keeps_staging_guarded_and_snapshot_pending(self):
        from types import SimpleNamespace

        layers = [self.Layer("cuda") for _ in range(2)]
        model = SimpleNamespace(state_cache_modules=lambda: layers)
        seq = self.pending_sequence(0, len(layers), 10)
        staging = GDNRestoreStaging()
        original_restore = layers[0].restore_state_slot

        def fail_scatter(_slot, _snapshot):
            raise RuntimeError("synthetic scatter failure")

        try:
            layers[0].restore_state_slot = fail_scatter
            with self.assertRaisesRegex(RuntimeError, "synthetic scatter failure"):
                restore_pending_states(model, [seq], staging)
            self.assertTrue(staging._pending)
            self.assertIsNotNone(seq.pending_state_snapshot)

            layers[0].restore_state_slot = original_restore
            # Retry without synchronization; staging must wait for the old transfer.
            restore_pending_states(model, [seq], staging)
            self.assertIsNone(seq.pending_state_snapshot)
            torch.cuda.synchronize()
            self.assert_restored_slots(layers, {0: 10})
        finally:
            layers[0].restore_state_slot = original_restore
            staging.clear()

    def test_invalid_restore_layout_is_rejected_before_any_copy(self):
        from types import SimpleNamespace

        layers = [self.Layer("cpu") for _ in range(2)]
        model = SimpleNamespace(state_cache_modules=lambda: layers)
        for fault in (
            "conv_shape",
            "conv_dtype",
            "recurrent_shape",
            "recurrent_dtype",
            "slot_out_of_range",
        ):
            with self.subTest(fault=fault):
                valid = self.pending_sequence(0, len(layers), 5)
                invalid = self.pending_sequence(1, len(layers), 10)
                states = list(invalid.pending_state_snapshot.layers)
                conv, recurrent = states[1]
                if fault == "conv_shape":
                    conv = conv[:, :-1]
                elif fault == "conv_dtype":
                    conv = conv.float()
                elif fault == "recurrent_shape":
                    recurrent = recurrent[..., :-1]
                elif fault == "recurrent_dtype":
                    recurrent = recurrent.to(torch.int64)
                else:
                    invalid.state_slot = 2
                states[1] = (conv, recurrent)
                invalid.pending_state_snapshot = GDNStateSnapshot(64, tuple(states))
                with self.assertRaisesRegex(
                    RuntimeError,
                    "state slot out of range"
                    if fault == "slot_out_of_range"
                    else "snapshot shape/dtype",
                ):
                    restore_pending_states(model, [valid, invalid], GDNRestoreStaging())
                self.assertIsNotNone(valid.pending_state_snapshot)
                self.assertIsNotNone(invalid.pending_state_snapshot)
                for layer in layers:
                    torch.testing.assert_close(layer.conv_state, torch.zeros_like(layer.conv_state))
                    torch.testing.assert_close(
                        layer.recurrent_state, torch.zeros_like(layer.recurrent_state)
                    )

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
