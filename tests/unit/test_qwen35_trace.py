"""CPU-only checks for the Qwen3.5 teacher-forced parity reporter."""

import torch
from torch import nn

from scripts.trace_qwen3_5 import _LayerCapture, _metric, compare_traces


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
        hf.append({
            "logits": torch.tensor([2.0, 1.0]),
            "layers": {
                "embedding": torch.zeros(2),
                "layer.0": torch.ones(2),
                "norm": torch.ones(2),
            },
            "states": {},
        })
        nano.append({
            "logits": torch.tensor([1.0, 3.0]) if step == 2 else torch.tensor([2.0, 1.0]),
            "layers": {
                "embedding": torch.zeros(2),
                "layer.0": torch.ones(2) + (0.2 if step == 2 else 0),
                "norm": torch.ones(2),
            },
            "states": {},
        })
    report = compare_traces([0, 0, 0], hf, nano, 0.05, 2)
    assert report["first_token_mismatch"] == 2
    assert report["first_error_above_threshold"]["step"] == 2
    assert report["first_error_above_threshold"]["site"] == "layer.0"
    assert report["per_step"][0]["phase"] == "prefill"
    assert report["per_step"][1]["phase"] == "decode"


class FakeNanoLayer(nn.Module):
    def forward(self, hidden):
        return hidden * 2, hidden


class FakeNanoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([FakeNanoLayer()])
        self.norm = nn.Identity()

    def forward(self, ids):
        embedding = self.embed_tokens(ids)
        hidden, residual = self.layers[0](embedding)
        return self.norm(hidden + residual)


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
