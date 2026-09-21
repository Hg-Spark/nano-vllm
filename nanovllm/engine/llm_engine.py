import atexit
from dataclasses import dataclass, fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


@dataclass(slots=True)
class StepStats:
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0


def _normalize_eos_token_ids(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(token_id) for token_id in value)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {
            field.name
            for field in fields(Config)
        }
        config_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in config_fields
        }
        config = Config(model, **config_kwargs)

        self.model_runner = ModelRunner(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            use_fast=True,
        )

        try:
            generation_config = GenerationConfig.from_pretrained(
                config.model
            )
            eos_token_ids = _normalize_eos_token_ids(
                generation_config.eos_token_id
            )
        except (OSError, ValueError):
            eos_token_ids = ()

        if not eos_token_ids:
            eos_token_ids = _normalize_eos_token_ids(
                self.tokenizer.eos_token_id
            )
        if not eos_token_ids:
            eos_token_ids = _normalize_eos_token_ids(
                getattr(
                    config.text_config,
                    "eos_token_id",
                    None,
                )
            )
        config.eos_token_ids = eos_token_ids

        self.scheduler = Scheduler(config)
        self._closed = False
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.model_runner.exit()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add(
            Sequence(prompt, sampling_params)
        )

    def _run_batch(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> tuple[list[tuple[int, list[int]]], float]:
        if not seqs:
            return [], 0.0

        start = perf_counter()
        prefix_snapshots: dict[int, object] = {}
        try:
            token_ids = self.model_runner.run(
                seqs,
                is_prefill,
            )
            if is_prefill:
                for seq in seqs:
                    if not self.scheduler.should_snapshot_prefix_after_step(
                        seq
                    ):
                        continue
                    target_tokens = (
                        seq.committed_tokens
                        + seq.num_scheduled_tokens
                    )
                    prefix_snapshots[seq.seq_id] = (
                        self.model_runner.capture_gdn_state(
                            seq,
                            target_tokens,
                        )
                    )
        except Exception:
            # KV/GDN tensors may have been mutated before logical commit.
            # Drop both request-owned histories; any previously published
            # joint-prefix entry remains independently pinned and valid.
            self.scheduler.recover_failed_step(seqs)
            raise
        elapsed = perf_counter() - start
        self.scheduler.postprocess(
            seqs,
            token_ids,
            is_prefill,
            prefix_snapshots,
        )
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        return outputs, elapsed

    def step(self):
        scheduled = self.scheduler.schedule()
        stats = StepStats(
            prefill_tokens=scheduled.prefill_tokens,
            decode_tokens=scheduled.decode_tokens,
        )

        try:
            decode_outputs, stats.decode_seconds = self._run_batch(
                scheduled.decode_seqs,
                False,
            )
        except Exception:
            # Prefill reservations for this scheduler step have not executed
            # yet. Drop them too, otherwise a final prompt chunk may remain
            # marked RUNNING with KV/state capacity that never received data.
            if scheduled.prefill_seqs:
                self.scheduler.recover_failed_step(
                    scheduled.prefill_seqs
                )
            raise

        prefill_outputs, stats.prefill_seconds = self._run_batch(
            scheduled.prefill_seqs,
            True,
        )
        return decode_outputs + prefill_outputs, stats

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [
                sampling_params
            ] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError(
                "sampling_params length must match prompts"
            )

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = 0.0
        decode_throughput = 0.0
        while not self.is_finished():
            output, stats = self.step()
            if stats.prefill_tokens and stats.prefill_seconds:
                prefill_throughput = (
                    stats.prefill_tokens
                    / stats.prefill_seconds
                )
            if stats.decode_tokens and stats.decode_seconds:
                decode_throughput = (
                    stats.decode_tokens
                    / stats.decode_seconds
                )

            pbar.set_postfix({
                "Prefill": (
                    f"{int(prefill_throughput)}tok/s"
                ),
                "Decode": (
                    f"{int(decode_throughput)}tok/s"
                ),
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        pbar.close()
        token_outputs = [
            outputs[seq_id]
            for seq_id in sorted(outputs)
        ]
        return [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
            }
            for token_ids in token_outputs
        ]
