import unittest
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


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
    seq.committed_tokens = len(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return seq


class CollectingScheduler:

    def __init__(self):
        self.seqs = []

    def add(self, seq):
        self.seqs.append(seq)


class FailingModelRunner:

    def run(self, seqs, is_prefill):
        if is_prefill:
            raise AssertionError(
                "prefill must not execute after decode failure"
            )
        raise RuntimeError("injected decode failure")


class LLMEngineStepTest(unittest.TestCase):

    def test_add_request_accepts_exact_context_boundary(self):
        engine = object.__new__(LLMEngine)
        engine.max_model_len = 4
        engine.scheduler = CollectingScheduler()

        engine.add_request(
            [1, 2],
            SamplingParams(max_tokens=2),
        )

        self.assertEqual(len(engine.scheduler.seqs), 1)

    def test_add_request_rejects_prompt_over_context_limit(self):
        engine = object.__new__(LLMEngine)
        engine.max_model_len = 3
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
            self.assertEqual(seq.num_scheduled_tokens, 0)
            self.assertEqual(seq.committed_tokens, 0)
            self.assertEqual(seq.state_slot, -1)
            self.assertFalse(seq.block_table)


if __name__ == "__main__":
    unittest.main()
