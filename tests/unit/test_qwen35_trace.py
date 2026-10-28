"""CPU-only checks for the Qwen3.5 teacher-forced parity reporter."""

import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise unittest.SkipTest("PyTorch is not installed") from exc

from scripts.trace_qwen3_5 import _hf_states, _LayerCapture, _metric, compare_traces


def test_metric_reports_exact_relative_error_and_shape():
    ref = torch.tensor([2.0, -2.0])
    same = _metric(ref, ref.clone())
    assert same["matched_shape"]
    assert same["relative_rmse"] == 0.0
    assert same["max_abs"] == 0.0

    changed = _metric(ref, torch.tensor([3.0, -2.0]))
    assert changed["relative_rmse"] > 0
    assert changed["max_abs"] == 1.0

    different_shape = _metric(ref, torch.zeros(1, 2))
    assert not different_shape["matched_shape"]


def test_report_first_mismatch_and_layer_before_logits():
    hf = []
    nano = []
    for step in range(3):
        hf.append(
            {
                "logits": torch.tensor([2.0, 1.0]),
                "layers": {
                    "embedding": torch.zeros(2),
                    "layer.0": torch.ones(2),
                    "norm": torch.ones(2),
                },
                "states": {},
            }
        )
        nano.append(
            {
                "logits": torch.tensor([1.0, 3.0]) if step == 2 else torch.tensor([2.0, 1.0]),
                "layers": {
                    "embedding": torch.zeros(2),
                    "layer.0": torch.ones(2) + (0.2 if step == 2 else 0),
                    "norm": torch.ones(2),
                },
                "states": {},
            }
        )
    report = compare_traces([0, 0, 0], hf, nano, 0.05, 2)
    assert report["first_token_mismatch"] == 2
    assert report["first_error_above_threshold"]["step"] == 2
    assert report["first_error_above_threshold"]["site"] == "layer.0"
    assert report["per_step"][0]["phase"] == "prefill"
    assert report["per_step"][1]["phase"] == "decode"


class FakeNanoLayer(nn.Module):
    def forward(self, hidden):
        return hidden * 2, hidden


class FakeFusedNorm(nn.Module):
    def forward_with_residual(self, hidden, residual):
        return hidden + residual


class FakeNanoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([FakeNanoLayer()])
        self.norm = FakeFusedNorm()

    def forward(self, ids):
        embedding = self.embed_tokens(ids)
        hidden, residual = self.layers[0](embedding)
        return self.norm.forward_with_residual(hidden, residual)


def test_nano_hook_reconstructs_residual_stream_before_mutation():
    model = FakeNanoBackbone()
    capture = _LayerCapture(model, nano=True)
    try:
        final = model(torch.tensor([1, 2, 3]))
        states = capture.snapshot()
        assert list(states) == ["embedding", "layer.0", "norm"]
        torch.testing.assert_close(states["layer.0"], final[-1].float())
        torch.testing.assert_close(states["norm"], final[-1].float())
    finally:
        capture.close()
    assert not model.layers[0]._forward_hooks
    assert "forward_with_residual" not in vars(model.norm)


def test_hf_gdn_state_collection_uses_layer_cache_and_valid_history():
    conv = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    recurrent = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    layer = SimpleNamespace(
        block_type="linear_attention",
        linear_attn=SimpleNamespace(conv_kernel_size=4),
    )
    cache = SimpleNamespace(
        layers=[SimpleNamespace(conv_states={0: conv}, recurrent_states={0: recurrent})]
    )
    states = _hf_states(cache, SimpleNamespace(layers=[layer]))
    torch.testing.assert_close(states["gdn.0.conv"], conv[0, :, -3:])
    torch.testing.assert_close(states["gdn.0.recurrent"], recurrent[0].transpose(-1, -2))
