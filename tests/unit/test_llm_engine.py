import unittest
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.schedule import BatchResult
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from scripts._common import prepare_fixed_decode_batch
from tests.scheduler_helpers import make_running_sequence, make_scheduler


def make_engine_scheduler():
    return make_scheduler(max_num_seqs=2, block_size=4, eos_token_ids=(99,))


def make_engine_stub() -> LLMEngine:
    engine = object.__new__(LLMEngine)
    engine._closed = False
    engine._atexit_handler = None
    return engine


class FailingDecodeRunner:
    def run(self, chunks, is_prefill):
        if is_prefill:
            raise AssertionError("prefill must not execute after decode failure")
        raise RuntimeError("injected decode failure")

    def exit(self):
        pass


class DecodeThenFailPrefillRunner:
    def run(self, chunks, is_prefill):
        if is_prefill:
            raise RuntimeError("injected prefill failure")
        return BatchResult([99 for _ in chunks], {})

    def exit(self):
        pass


class LLMEngineStepTest(unittest.TestCase):
    def test_add_request_returns_id_used_by_scheduler(self):
        engine = make_engine_stub()
        engine.scheduler = make_engine_scheduler()
        engine.max_model_len = 16
        engine.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=128))
        engine.max_kv_tokens = 16

        seq_id = engine.add_request([1, 2], SamplingParams(max_tokens=2))

        self.assertEqual(len(engine.scheduler.waiting), 1)
        self.assertEqual(engine.scheduler.waiting[0].seq_id, seq_id)

    def test_decode_failure_disables_engine(self):
        scheduler = make_scheduler(max_num_seqs=2, num_blocks=32, block_size=4, eos_token_ids=(99,))
        running = make_running_sequence(scheduler, list(range(65)))
        scheduler.block_manager.prefix_cache.publish_many(running, (GDNStateSnapshot(64, ()),))
        image = SimpleNamespace(visual_features=object(), image_start=0)
        scheduler.add(Sequence([3, 4], SamplingParams(max_tokens=2), image_state=image))
        engine = make_engine_stub()
        engine.scheduler = scheduler
        engine.model_runner = FailingDecodeRunner()

        with self.assertRaisesRegex(RuntimeError, "injected decode failure"):
            engine.step()

        self.assertTrue(engine._closed)
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(len(scheduler.block_manager.prefix_cache), 0)
        self.assertEqual(
            len(scheduler.block_manager.free_block_ids), scheduler.block_manager.num_blocks
        )
        self.assertIsNone(image.visual_features)
        engine.exit()  # Repeated cleanup is safe.
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.step()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.add_request([1], SamplingParams())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.generate([[1]], SamplingParams(), use_tqdm=False)

    def test_interrupted_step_disables_engine_before_commit(self):
        scheduler = make_engine_scheduler()
        make_running_sequence(scheduler, [1, 2])
        executed = []

        def interrupted_run(chunks, is_prefill):
            executed.extend((chunk.start, chunk.end) for chunk in chunks)
            raise KeyboardInterrupt()

        engine = make_engine_stub()
        engine.scheduler = scheduler
        engine.model_runner = SimpleNamespace(run=interrupted_run, exit=lambda: None)

        with self.assertRaises(KeyboardInterrupt):
            engine.step()
        self.assertEqual(executed, [(1, 2)])
        self.assertTrue(engine._closed)
        self.assertTrue(scheduler.is_finished())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.step()

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
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(waiting.state_slot, -1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.step()

    def test_fixed_decode_rejects_overcommitted_kv(self):
        scheduler = make_scheduler(max_num_seqs=2, num_blocks=3, block_size=4)
        engine = SimpleNamespace(
            scheduler=scheduler,
            is_finished=scheduler.is_finished,
            add_request=lambda prompt, params: scheduler.add(Sequence(prompt, params)),
        )
        with self.assertRaisesRegex(ValueError, "exceeds KV capacity"):
            prepare_fixed_decode_batch(engine, [1, 2, 3, 4, 5], SamplingParams(max_tokens=2), 2)

    def test_input_token_ids_rejected_before_enqueue(self):
        engine = make_engine_stub()
        engine.scheduler = make_engine_scheduler()
        engine.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=128))
        engine.max_model_len = 16
        engine.max_kv_tokens = 16
        for invalid_ids in ([128], [True]):
            with self.assertRaisesRegex(ValueError, "prompt token IDs"):
                engine.add_request(invalid_ids, SamplingParams(max_tokens=1))
        self.assertFalse(engine.scheduler.waiting)
        # SamplingParams already tests the invalid-value matrix; the engine
        # only needs to prove it revalidates parameters modified after creation.
        params = SamplingParams(max_tokens=1)
        params.max_tokens = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            engine.add_request([1, 2], params)
        params = SamplingParams()
        params.temperature = float("nan")
        with self.assertRaisesRegex(ValueError, "temperature"):
            engine.add_request([1, 2], params)
        params = SamplingParams()
        params.ignore_eos = "true"
        with self.assertRaisesRegex(ValueError, "ignore_eos"):
            engine.add_request([1, 2], params)
        self.assertFalse(engine.scheduler.waiting)
        seq = engine._prepare_request((1, 2), SamplingParams(max_tokens=1))
        seq.append_token(3)
        self.assertEqual(seq.token_ids, [1, 2, 3])

    def test_fixed_decode_parks_completed_prefills(self):
        scheduler = make_scheduler(
            max_num_seqs=3,
            max_num_batched_tokens=4,
            num_blocks=8,
            block_size=4,
            max_prefix_cache_entries=0,
        )
        decode_seen = []

        def run_batch(chunks, is_prefill):
            if not is_prefill:
                decode_seen.append(chunks)
            sampled = [7 if chunk.end == len(chunk.seq) else None for chunk in chunks]
            scheduler.postprocess(chunks, sampled, True)
            return [], 0.0

        engine = SimpleNamespace(
            scheduler=scheduler,
            is_finished=scheduler.is_finished,
            add_request=lambda prompt, params: scheduler.add(Sequence(prompt, params)),
            _run_batch=run_batch,
        )
        prepare_fixed_decode_batch(engine, list(range(6)), SamplingParams(max_tokens=2), 3)
        self.assertEqual(len(scheduler.running), 3)
        self.assertEqual(decode_seen, [])

    def test_generate_validates_all_prompts_before_enqueueing_any(self):
        engine = make_engine_stub()
        engine.scheduler = make_engine_scheduler()
        engine.max_model_len = 4
        engine.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=128))
        engine.max_kv_tokens = 4
        engine.tokenizer = SimpleNamespace(encode=lambda text: [1] * len(text))

        with self.assertRaisesRegex(TypeError, "expects a list"):
            engine.generate("ok", SamplingParams(max_tokens=1), use_tqdm=False)
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
