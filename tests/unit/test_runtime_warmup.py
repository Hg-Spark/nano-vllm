import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nanovllm.engine.runtime_warmup import warmup_runtime


class RuntimeWarmupTest(unittest.TestCase):

    def test_runtime_warmup_profiles_decode_vision_and_full_prefill_budget(self):
        model = object()
        config = SimpleNamespace(
            max_num_batched_tokens=10,
            max_model_len=4,
            max_num_seqs=3,
        )
        buckets = {1: object()}
        run_batch = Mock()

        with (
            patch(
                "nanovllm.engine.runtime_warmup._warmup_decode_moe"
            ) as decode,
            patch(
                "nanovllm.engine.runtime_warmup._warmup_vision"
            ) as vision,
            patch(
                "nanovllm.engine.runtime_warmup.torch.cuda.empty_cache"
            ),
            patch(
                "nanovllm.engine.runtime_warmup.torch.cuda.reset_peak_memory_stats"
            ),
        ):
            warmup_runtime(model, config, buckets, run_batch)

        decode.assert_called_once_with(model, config, buckets)
        vision.assert_called_once_with(model, config)
        chunks, is_prefill = run_batch.call_args.args
        self.assertTrue(is_prefill)
        self.assertEqual(
            [chunk.num_tokens for chunk in chunks],
            [4, 4, 2],
        )


if __name__ == "__main__":
    unittest.main()
