import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.models.qwen3_5_moe import initialize_moe_runtime


class MoeRuntimeTest(unittest.TestCase):
    def make_model(self):
        return SimpleNamespace(
            parameters=lambda: iter([torch.zeros(1, dtype=torch.float16)]),
            bind_moe_runtime=Mock(),
        )

    def make_config(self):
        return SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=64,
            text_config=SimpleNamespace(
                hidden_size=8, moe_intermediate_size=4, num_experts=4, num_experts_per_tok=2
            ),
        )

    def test_unsupported_gpu_fails_without_fallback(self):
        model = self.make_model()
        with (
            patch(
                "nanovllm.models.qwen3_5_moe.torch.cuda.get_device_capability",
                return_value=(7, 5),
            ),
            self.assertRaisesRegex(RuntimeError, "unsupported on SM75"),
        ):
            initialize_moe_runtime(model, self.make_config())
        model.bind_moe_runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
