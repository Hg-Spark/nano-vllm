import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nanovllm.engine.batch import PreparedBatch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import BatchResult, ModelRunner
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_manager import GDNStateSnapshot
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import Context, get_context


def make_scheduler():
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos_token_ids=(99,),
        kvcache_block_size=4,
        max_prefix_cache_entries=16,
    )
    return Scheduler(config, 16)


def make_running_sequence(scheduler, token_ids):
    seq = Sequence(token_ids)
    scheduler.block_manager.ensure_capacity(seq, len(seq))
    scheduler.state_manager.allocate(seq)
    seq.committed_tokens = len(seq) - 1
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return seq


class CollectingScheduler:

    def __init__(self):
        self.seqs = []

    def is_finished(self):
        return not self.seqs

    def add(self, seq):
        self.seqs.append(seq)


class FailingModelRunner:

    def run(self, seqs, is_prefill):
        if is_prefill:
            raise AssertionError(
                "prefill must not execute after decode failure"
            )
        raise RuntimeError("injected decode failure")


class SamplingModelRunner:

    def run(self, chunks, is_prefill):
        return BatchResult([99 for _ in chunks], {})


class LLMEngineStepTest(unittest.TestCase):

    def test_exit_unregisters_callback_and_is_idempotent(self):
        engine = object.__new__(LLMEngine)
        engine._closed = False
        engine._atexit_handler = Mock()
        engine.model_runner = Mock()

        with patch(
            "nanovllm.engine.llm_engine.atexit.unregister"
        ) as unregister:
            engine.exit()
            engine.exit()

        unregister.assert_called_once()
        engine.model_runner.exit.assert_called_once()
        self.assertTrue(engine._closed)
        self.assertIsNone(engine._atexit_handler)

    def test_model_runner_exit_releases_gpu_owners(self):
        runner = object.__new__(ModelRunner)
        runner._closed = False
        runner.model = object()
        runner.kv_cache = object()
        runner.sampler = object()

        with (
            patch(
                "nanovllm.engine.model_runner.torch.cuda.synchronize"
            ) as synchronize,
            patch(
                "nanovllm.engine.model_runner.torch.cuda.empty_cache"
            ) as empty_cache,
        ):
            runner.exit()
            runner.exit()

        synchronize.assert_called_once()
        empty_cache.assert_called_once()
        self.assertIsNone(runner.model)
        self.assertIsNone(runner.kv_cache)
        self.assertIsNone(runner.sampler)

    def test_model_runner_scopes_context_and_returns_requested_snapshot(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = 4
        context = Context(is_prefill=True)
        previous_context = get_context()
        seq = Sequence([0, 1, 2, 3, 4, 5])
        chunk = ScheduledChunk(seq, 0, 4, capture_snapshot=True)
        snapshot = GDNStateSnapshot(num_tokens=4, layers=())

        def forward(*_):
            self.assertIs(get_context(), context)
            return object()

        runner.model = Mock(side_effect=forward)
        with (
            patch(
                "nanovllm.engine.model_runner.prepare_prefill",
                return_value=PreparedBatch(object(), object(), context),
            ),
            patch(
                "nanovllm.engine.model_runner.capture_gdn_state",
                return_value=snapshot,
            ) as capture,
        ):
            result = runner.run((chunk,), True)

        self.assertEqual(result.token_ids, [None])
        self.assertEqual(result.prefix_snapshots, {seq.seq_id: snapshot})
        capture.assert_called_once_with(runner.model, chunk)
        self.assertIs(get_context(), previous_context)

    def test_add_request_accepts_exact_context_boundary(self):
        engine = object.__new__(LLMEngine)
        engine.max_model_len = 4
        engine.max_kv_tokens = 4
        engine.scheduler = CollectingScheduler()

        engine.add_request(
            [1, 2],
            SamplingParams(max_tokens=2),
        )

        self.assertEqual(len(engine.scheduler.seqs), 1)

    def test_add_request_rejects_prompt_over_context_limit(self):
        engine = object.__new__(LLMEngine)
        engine.max_model_len = 3
        engine.max_kv_tokens = 3
        engine.scheduler = CollectingScheduler()

        with self.assertRaisesRegex(
            ValueError,
            "prompt length exceeds max_model_len",
        ):
            engine.add_request(
                [1, 2, 3, 4],
                SamplingParams(max_tokens=1),
            )

        self.assertFalse(engine.scheduler.seqs)

    def test_add_request_rejects_completion_over_context_limit(self):
        engine = object.__new__(LLMEngine)
        engine.max_model_len = 4
        engine.max_kv_tokens = 4
        engine.scheduler = CollectingScheduler()

        with self.assertRaisesRegex(
            ValueError,
            r"prompt \+ completion exceeds max_model_len",
        ):
            engine.add_request(
                [1, 2, 3],
                SamplingParams(max_tokens=2),
            )

        self.assertFalse(engine.scheduler.seqs)

    def test_add_request_rejects_physical_kv_capacity_overflow(self):
        engine = object.__new__(LLMEngine)
        engine.max_model_len = 8
        engine.max_kv_tokens = 4
        engine.scheduler = CollectingScheduler()

        with self.assertRaisesRegex(
            ValueError,
            "physical KV cache capacity",
        ):
            engine.add_request(
                [1, 2, 3, 4],
                SamplingParams(max_tokens=2),
            )

        self.assertFalse(engine.scheduler.seqs)

    def test_generate_requires_idle_engine(self):
        engine = object.__new__(LLMEngine)
        engine.max_model_len = 8
        engine.max_kv_tokens = 8
        engine.scheduler = CollectingScheduler()
        engine.add_request(
            [1, 2],
            SamplingParams(max_tokens=1),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "generate requires an idle engine",
        ):
            engine.generate(
                [[3, 4]],
                SamplingParams(max_tokens=1),
                use_tqdm=False,
            )

        self.assertEqual(len(engine.scheduler.seqs), 1)

    def test_generate_validation_failure_does_not_enqueue_partial_batch(self):
        engine = object.__new__(LLMEngine)
        engine.max_model_len = 4
        engine.max_kv_tokens = 4
        engine.scheduler = CollectingScheduler()

        with self.assertRaisesRegex(
            ValueError,
            r"prompt \+ completion exceeds max_model_len",
        ):
            engine.generate(
                [[1, 2], [3, 4, 5, 6]],
                SamplingParams(max_tokens=1),
                use_tqdm=False,
            )

        self.assertFalse(engine.scheduler.seqs)

    def test_decode_failure_rolls_back_unexecuted_prefill_reservation(self):
        scheduler = make_scheduler()
        decode_seq = make_running_sequence(
            scheduler,
            [1, 2],
        )
        prefill_seq = Sequence([3, 4, 5])
        scheduler.add(prefill_seq)

        engine = object.__new__(LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = FailingModelRunner()

        with self.assertRaisesRegex(
            RuntimeError,
            "injected decode failure",
        ):
            engine.step()

        self.assertFalse(scheduler.running)
        self.assertEqual(
            set(scheduler.waiting),
            {decode_seq, prefill_seq},
        )

        for seq in (decode_seq, prefill_seq):
            self.assertEqual(seq.status, SequenceStatus.WAITING)
            self.assertEqual(seq.committed_tokens, 0)
            self.assertEqual(seq.state_slot, -1)
            self.assertFalse(seq.block_table)

    def test_step_commits_runner_result_from_scheduled_chunks(self):
        scheduler = make_scheduler()
        seq = Sequence([3, 4], SamplingParams(max_tokens=1))
        scheduler.add(seq)

        engine = object.__new__(LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = SamplingModelRunner()

        outputs, stats = engine.step()

        self.assertEqual(outputs, [(seq.seq_id, [99])])
        self.assertEqual(stats.prefill_tokens, 2)
        self.assertTrue(scheduler.is_finished())


if __name__ == "__main__":
    unittest.main()
