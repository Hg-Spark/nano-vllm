import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"requires torch: {exc}")

from nanovllm.engine.moe_runtime import initialize_moe_runtime


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
                "nanovllm.engine.moe_runtime.torch.cuda.get_device_capability",
                return_value=(7, 5),
            ),
            self.assertRaisesRegex(RuntimeError, "unsupported on SM75"),
        ):
            initialize_moe_runtime(model, self.make_config())
        model.bind_moe_runtime.assert_not_called()

    def test_kernel_unavailable_is_reported_without_fallback(self):
        model = self.make_model()
        with (
            patch(
                "nanovllm.engine.moe_runtime.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ),
            patch(
                "nanovllm.engine.moe_runtime.cutlass_fused_moe_workspace_size",
                side_effect=NotImplementedError("unsupported shape"),
            ),
            self.assertRaisesRegex(RuntimeError, "unsupported shape"),
        ):
            initialize_moe_runtime(model, self.make_config())
        model.bind_moe_runtime.assert_not_called()

    def test_workspace_covers_prefill_and_decode(self):
        model = self.make_model()
        config = self.make_config()
        real_zeros = torch.zeros

        def fake_empty(shape, *, dtype, device):
            return real_zeros(shape, dtype=dtype)

        with (
            patch(
                "nanovllm.engine.moe_runtime.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ),
            patch(
                "nanovllm.engine.moe_runtime.cutlass_fused_moe_workspace_size",
                return_value=4096,
            ) as size,
            patch("nanovllm.engine.moe_runtime.torch.empty", side_effect=fake_empty),
        ):
            workspace, output = initialize_moe_runtime(model, config)
        self.assertEqual(size.call_args.args[:5], (64, 8, 4, 4, 2))
        self.assertEqual(workspace.shape, (4096,))
        self.assertEqual(output.shape, (64, 8))
        model.bind_moe_runtime.assert_called_once_with(workspace, output)


if __name__ == "__main__":
    unittest.main()
