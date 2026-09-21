import unittest
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler():
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos_token_ids=(99,),
        kvcache_block_size=4,
        num_kvcache_blocks=16,
        max_num_state_slots=2,
    )
    Sequence.block_size = config.kvcache_block_size
    return Scheduler(config)


def make_running_sequence(scheduler, token_ids):
    seq = Sequence(token_ids)
    scheduler.block_manager.ensure_capacity(seq, len(seq))
    scheduler.state_manager.allocate(seq)
    seq.num_cached_tokens = len(seq)
    seq.num_state_tokens = len(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return seq


class FailingModelRunner:

    def run(self, seqs, is_prefill):
        if is_prefill:
            raise AssertionError(
                "prefill must not execute after decode failure"
            )
        raise RuntimeError("injected decode failure")


class LLMEngineStepTest(unittest.TestCase):

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
            self.assertEqual(seq.num_cached_tokens, 0)
            self.assertEqual(seq.num_state_tokens, 0)
            self.assertEqual(seq.state_slot, -1)
            self.assertFalse(seq.block_table)


if __name__ == "__main__":
    unittest.main()
