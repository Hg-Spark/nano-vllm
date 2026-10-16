import unittest
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.schedule import BatchResult
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from tests.scheduler_helpers import make_running_sequence, make_scheduler


def make_engine_scheduler():
    return make_scheduler(max_num_seqs=2, block_size=4, eos_token_ids=(99,))


def make_engine_stub() -> LLMEngine:
    engine = object.__new__(LLMEngine)
    engine._closed = False
    return engine


class FailingDecodeRunner:
    def run(self, chunks, is_prefill):
        if is_prefill:
            raise AssertionError("prefill must not execute after decode failure")
        raise RuntimeError("injected decode failure")


class DecodeThenFailPrefillRunner:
    def run(self, chunks, is_prefill):
        if is_prefill:
            raise RuntimeError("injected prefill failure")
        return BatchResult([99 for _ in chunks], {})


class LLMEngineStepTest(unittest.TestCase):
    def test_add_request_returns_id_used_by_scheduler(self):
        engine = make_engine_stub()
        engine.scheduler = make_engine_scheduler()
        engine.max_model_len = 16
        engine.max_kv_tokens = 16

        seq_id = engine.add_request([1, 2], SamplingParams(max_tokens=2))

        self.assertEqual(len(engine.scheduler.waiting), 1)
        self.assertEqual(engine.scheduler.waiting[0].seq_id, seq_id)

    def test_decode_failure_disables_engine(self):
        scheduler = make_engine_scheduler()
        make_running_sequence(scheduler, [1, 2])
        scheduler.add(Sequence([3, 4]))
        engine = make_engine_stub()
        engine.scheduler = scheduler
        engine.model_runner = FailingDecodeRunner()

        with self.assertRaisesRegex(RuntimeError, "injected decode failure"):
            engine.step()

        self.assertTrue(engine._closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.step()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.add_request([1], SamplingParams())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.generate([[1]], SamplingParams(), use_tqdm=False)

    def test_prefill_failure_disables_engine_after_committed_decode(self):
        scheduler = make_engine_scheduler()
        running = make_running_sequence(scheduler, [1, 2])
        waiting = Sequence([3, 4], SamplingParams(max_tokens=2))
        scheduler.add(waiting)
        engine = make_engine_stub()
        engine.scheduler = scheduler
        engine.model_runner = DecodeThenFailPrefillRunner()

        with self.assertRaisesRegex(RuntimeError, "injected prefill failure"):
            engine.step()

        self.assertTrue(running.is_finished)
        self.assertTrue(engine._closed)
        self.assertIn(waiting, scheduler.running)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.step()

    def test_generate_validates_all_prompts_before_enqueueing_any(self):
        engine = make_engine_stub()
        engine.scheduler = make_engine_scheduler()
        engine.max_model_len = 4
        engine.max_kv_tokens = 4
        engine.tokenizer = SimpleNamespace(encode=lambda text: [1] * len(text))

        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            engine.generate(
                ["ok", "too_long"],
                SamplingParams(max_tokens=1),
                use_tqdm=False,
            )
        self.assertTrue(engine.is_finished())
        self.assertFalse(engine.scheduler.waiting)


if __name__ == "__main__":
    unittest.main()
