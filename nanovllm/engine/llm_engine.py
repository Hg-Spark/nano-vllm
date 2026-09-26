import atexit
from dataclasses import dataclass, fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer, GenerationConfig

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.schedule import ScheduledChunk
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
            if field.init
        }
        unknown = set(kwargs) - config_fields
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unsupported runtime options: {names}")
        config = Config(model, **kwargs)
        self.max_model_len = config.max_model_len
        self.model_runner = None
        self._closed = False
        self._atexit_handler = None

        try:
            self.model_runner = ModelRunner(config)
            self.max_kv_tokens = (
                self.model_runner.num_kvcache_blocks
                * config.kvcache_block_size
            )
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

            self.scheduler = Scheduler(
                config,
                self.model_runner.num_kvcache_blocks,
            )
        except Exception:
            if self.model_runner is not None:
                self.model_runner.exit()
                self.model_runner = None
            raise

        self._atexit_handler = self.exit
        atexit.register(self._atexit_handler)

    def exit(self):
        if self._closed:
            return
        self._closed = True

        handler = self._atexit_handler
        self._atexit_handler = None
        if handler is not None:
            atexit.unregister(handler)

        if self.model_runner is not None:
            self.model_runner.exit()
            self.model_runner = None

    def _prepare_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        prompt_len = len(prompt)
        if prompt_len > self.max_model_len:
            raise ValueError(
                "prompt length exceeds max_model_len: "
                f"prompt={prompt_len}, max_model_len={self.max_model_len}"
            )
        requested_total = prompt_len + sampling_params.max_tokens
        if requested_total > self.max_model_len:
            raise ValueError(
                "requested prompt + completion exceeds max_model_len: "
                f"prompt={prompt_len}, max_tokens={sampling_params.max_tokens}, "
                f"max_model_len={self.max_model_len}"
            )

        required_kv_tokens = (
            prompt_len
            + max(0, sampling_params.max_tokens - 1)
        )
        if required_kv_tokens > self.max_kv_tokens:
            raise ValueError(
                "request exceeds physical KV cache capacity: "
                f"required={required_kv_tokens}, "
                f"capacity={self.max_kv_tokens}"
            )

        return Sequence(prompt, sampling_params)

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ):
        self.scheduler.add(
            self._prepare_request(prompt, sampling_params)
        )

    def _run_batch(
        self,
        chunks: tuple[ScheduledChunk, ...],
        is_prefill: bool,
    ) -> tuple[list[tuple[int, list[int]]], float]:
        if not chunks:
            return [], 0.0

        start = perf_counter()
        try:
            result = self.model_runner.run(
                chunks,
                is_prefill,
            )
        except Exception:
            # KV/GDN tensors may have been mutated before logical commit.
            # Drop both request-owned histories; any previously published
            # joint-prefix entry remains independently pinned and valid.
            self.scheduler.recover_failed_step(chunks)
            raise
        elapsed = perf_counter() - start
        try:
            self.scheduler.postprocess(
                chunks,
                result.token_ids,
                is_prefill,
                result.prefix_snapshots,
            )
        except Exception:
            # postprocess validates and publishes reusable prefixes before
            # logical commit. Request-owned physical state can therefore be
            # discarded and replayed safely if publication/commit fails.
            self.scheduler.recover_failed_step(chunks)
            raise
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in (chunk.seq for chunk in chunks)
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
                scheduled.decode_chunks,
                False,
            )
        except Exception:
            # Prefill reservations for this scheduler step have not executed
            # yet. Drop them too, otherwise a final prompt chunk may remain
            # marked RUNNING with KV/state capacity that never received data.
            if scheduled.prefill_chunks:
                self.scheduler.recover_failed_step(
                    scheduled.prefill_chunks
                )
            raise

        prefill_outputs, stats.prefill_seconds = self._run_batch(
            scheduled.prefill_chunks,
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
        if not self.scheduler.is_finished():
            raise RuntimeError(
                "generate requires an idle engine; "
                "use add_request()/step() for continuous batching"
            )

        if not isinstance(sampling_params, list):
            sampling_params = [
                sampling_params
            ] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError(
                "sampling_params length must match prompts"
            )

        seqs = [
            self._prepare_request(prompt, sp)
            for prompt, sp in zip(prompts, sampling_params)
        ]
        for seq in seqs:
            self.scheduler.add(seq)

        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
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
