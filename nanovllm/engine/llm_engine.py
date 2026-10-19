import atexit
import logging
import weakref
from dataclasses import dataclass
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.schedule import ScheduledChunk
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.multimodal.request import ImagePrompt, image_state_from_processor_output
from nanovllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


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
    def __init__(self, model, *, dtype: str, **kwargs) -> None:
        from transformers import AutoTokenizer, GenerationConfig

        from nanovllm.engine.model_runner import ModelRunner

        config = Config(model, dtype=dtype, **kwargs)
        self.config = config
        self.max_model_len = config.max_model_len
        self.model_runner = None
        self._closed = False
        self._atexit_handler = None
        self.processor = None

        try:
            self.model_runner = ModelRunner(config)
            self.max_kv_tokens = self.model_runner.num_kv_cache_blocks * config.kv_cache_block_size
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

            try:
                generation_config = GenerationConfig.from_pretrained(config.model)
                eos_token_ids = _normalize_eos_token_ids(generation_config.eos_token_id)
            except (OSError, ValueError):
                eos_token_ids = ()

            if not eos_token_ids:
                eos_token_ids = _normalize_eos_token_ids(self.tokenizer.eos_token_id) or (
                    _normalize_eos_token_ids(getattr(config.text_config, "eos_token_id", None))
                )
            config.eos_token_ids = eos_token_ids
            self.scheduler = Scheduler(
                config,
                self.model_runner.num_kv_cache_blocks,
                snapshot_nbytes=self.model_runner.snapshot_nbytes,
            )
        except Exception:
            if self.model_runner is not None:
                self.model_runner.exit()
                self.model_runner = None
            raise

        engine_ref = weakref.ref(self)

        def cleanup() -> None:
            engine = engine_ref()
            if engine is not None:
                engine.exit()

        self._atexit_handler = cleanup
        atexit.register(cleanup)

    def exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._atexit_handler is not None:
            atexit.unregister(self._atexit_handler)
            self._atexit_handler = None
        runner = self.model_runner
        self.model_runner = None
        try:
            if runner is not None:
                runner.exit()
        finally:
            self.scheduler.clear()

    def _get_processor(self):
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.config.model)
        return self.processor

    def _prepare_request(
        self,
        prompt: str | list[int] | ImagePrompt,
        sampling_params: SamplingParams,
    ) -> Sequence:
        sampling_params.validate()
        image_state = None
        if isinstance(prompt, ImagePrompt):
            if self.config.max_visual_tokens_per_request == 0:
                raise ValueError("image requests require max_visual_tokens_per_request > 0 or None")
            processed = self._get_processor()(
                text=prompt.text, images=prompt.image, return_tensors="pt"
            )
            prompt, image_state = image_state_from_processor_output(
                processed, self.config.root_config
            )
            visual_limit = self.config.max_visual_tokens_per_request
            if (
                visual_limit is not None
                and image_state.image_end - image_state.image_start > visual_limit
            ):
                raise ValueError(
                    "image has more visual tokens than max_visual_tokens_per_request: "
                    f"{image_state.image_end - image_state.image_start} > {visual_limit}"
                )
        elif isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        else:
            vocab_size = self.config.text_config.vocab_size
            if any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < vocab_size
                for token_id in prompt
            ):
                raise ValueError(f"prompt token IDs must be integers in [0, {vocab_size})")
        prompt_len = len(prompt)
        if prompt_len + sampling_params.max_tokens > self.max_model_len:
            raise ValueError(
                f"requested prompt ({prompt_len}) + max_tokens ({sampling_params.max_tokens}) "
                f"exceeds max_model_len ({self.max_model_len})"
            )
        required_kv_tokens = prompt_len + max(0, sampling_params.max_tokens - 1)
        if required_kv_tokens > self.max_kv_tokens:
            raise ValueError(
                f"request exceeds physical KV cache capacity: "
                f"required={required_kv_tokens}, capacity={self.max_kv_tokens}"
            )
        return Sequence(prompt, sampling_params, image_state=image_state)

    def _ensure_usable(self) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")

    def add_request(
        self,
        prompt: str | list[int] | ImagePrompt,
        sampling_params: SamplingParams,
    ) -> int:
        self._ensure_usable()
        seq = self._prepare_request(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def _run_batch(
        self,
        chunks: tuple[ScheduledChunk, ...],
        is_prefill: bool,
    ) -> tuple[list[tuple[int, list[int]]], float]:
        if not chunks:
            return [], 0.0
        start = perf_counter()
        result = self.model_runner.run(chunks, is_prefill)
        elapsed = perf_counter() - start
        self.scheduler.postprocess(
            chunks,
            result.token_ids,
            is_prefill,
            result.prefix_snapshots,
        )
        outputs = [
            (chunk.seq.seq_id, chunk.seq.completion_token_ids)
            for chunk in chunks
            if chunk.seq.is_finished
        ]
        return outputs, elapsed

    def step(self) -> tuple[list[tuple[int, list[int]]], StepStats]:
        """Execute Decode then Prefill; an execution failure makes the engine unusable."""
        self._ensure_usable()
        completed = False
        try:
            scheduled = self.scheduler.schedule()
            stats = StepStats(
                prefill_tokens=scheduled.prefill_tokens,
                decode_tokens=scheduled.decode_tokens,
            )
            decode_outputs, stats.decode_seconds = self._run_batch(scheduled.decode_chunks, False)
            prefill_outputs, stats.prefill_seconds = self._run_batch(scheduled.prefill_chunks, True)
            completed = True
            return decode_outputs + prefill_outputs, stats
        finally:
            if not completed:
                try:
                    self.exit()
                except Exception:
                    logger.exception("engine cleanup failed after an execution error")

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[ImagePrompt],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        self._ensure_usable()
        if not self.is_finished():
            raise RuntimeError(
                "generate requires an idle engine; use add_request()/step() for continuous batching"
            )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params length must match prompts")

        seqs = [self._prepare_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        for seq in seqs:
            self.scheduler.add(seq)

        if use_tqdm:
            from tqdm.auto import tqdm

            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        else:
            pbar = None
        outputs: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            output, stats = self.step()
            if stats.prefill_tokens and stats.prefill_seconds:
                prefill_throughput = stats.prefill_tokens / stats.prefill_seconds
            if stats.decode_tokens and stats.decode_seconds:
                decode_throughput = stats.decode_tokens / stats.decode_seconds
            if pbar is not None:
                pbar.set_postfix(
                    Prefill=f"{int(prefill_throughput)}tok/s",
                    Decode=f"{int(decode_throughput)}tok/s",
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()
        return [
            {
                "text": self.tokenizer.decode(outputs[seq_id], skip_special_tokens=True),
                "token_ids": outputs[seq_id],
            }
            for seq_id in sorted(outputs)
        ]
