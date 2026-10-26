"""CLI helpers, comparisons and optional result records."""

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


def add_model_argument(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    if required:
        parser.add_argument("model", help="local checkpoint directory")
    else:
        parser.add_argument(
            "model",
            nargs="?",
            default=os.environ.get("NANOVLLM_MODEL"),
            help="local checkpoint directory (or set NANOVLLM_MODEL)",
        )


def add_kv_cache_arguments(parser: argparse.ArgumentParser) -> None:
    """Optional KV storage precision; model compute precision remains BF16."""
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8_e4m3"), default="auto")
    parser.add_argument("--kv-cache-k-scale", type=float, default=1.0)
    parser.add_argument("--kv-cache-v-scale", type=float, default=1.0)


def kv_cache_kwargs(args) -> dict[str, str | float]:
    return {
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_cache_k_scale": args.kv_cache_k_scale,
        "kv_cache_v_scale": args.kv_cache_v_scale,
    }


def add_hf_experts_backend_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hf-experts-backend",
        choices=("eager", "batched_mm", "grouped_mm"),
        default="eager",
        help="explicit Transformers expert backend; eager is the reference baseline",
    )


def hf_model_options(backend: str | None) -> dict[str, str]:
    return {"experts_implementation": backend} if backend is not None else {}


def release_hf_memory() -> None:
    """Free HF model allocations after the caller drops its tensor references."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run_hf_reference(model_cls, model_path, experts_backend, evaluate):
    """Run a reference probe and release HF GPU allocations before the next engine."""
    import torch

    model = None
    try:
        model = (
            model_cls.from_pretrained(
                model_path, dtype=torch.bfloat16, **hf_model_options(experts_backend)
            )
            .cuda()
            .eval()
        )
        return evaluate(model)
    finally:
        model = None
        release_hf_memory()


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (l_tok, r_tok) in enumerate(zip(left, right)):
        if l_tok != r_tok:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def compare_tokens(reference: list[int], actual: list[int]) -> dict:
    index = first_mismatch(reference, actual)
    return {
        "matched": index is None,
        "reference_length": len(reference),
        "actual_length": len(actual),
        "first_mismatch": index,
        "reference_token": reference[index]
        if index is not None and index < len(reference)
        else None,
        "actual_token": actual[index] if index is not None and index < len(actual) else None,
    }


def mismatch_message(
    comparison: dict,
    *,
    failure_prefix="FAIL: first mismatch at completion token",
    reference_name="hf",
    actual_name="nano",
) -> str:
    return (
        f"{failure_prefix} {comparison['first_mismatch']}: "
        f"{reference_name}={comparison['reference_token']}, "
        f"{actual_name}={comparison['actual_token']}"
    )


class ValidationFailure(Exception):
    def __init__(self, message: str, result: dict) -> None:
        super().__init__(message)
        self.result = result


def require_greedy_parity(
    hf_tokens: list[int],
    nano_tokens: list[int],
    expected_tokens: int,
    hf_experts_backend: str | None,
    *,
    mismatch_prefix: str = "FAIL: first mismatch at completion token",
    **metadata,
) -> dict:
    comparison = compare_tokens(hf_tokens, nano_tokens)
    result = {
        "hf_tokens": hf_tokens,
        "nano_tokens": nano_tokens,
        "comparison": comparison,
        "expected_tokens": expected_tokens,
        "hf_experts_backend": hf_experts_backend,
        "full_length": len(hf_tokens) == expected_tokens and len(nano_tokens) == expected_tokens,
        **metadata,
    }
    if not result["full_length"]:
        raise ValidationFailure(
            f"FAIL: expected {expected_tokens} greedy tokens, got HF={len(hf_tokens)}, Nano={len(nano_tokens)}",
            result,
        )
    if not comparison["matched"]:
        raise ValidationFailure(
            mismatch_message(comparison, failure_prefix=mismatch_prefix),
            result,
        )
    return result


def validate_positive(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *names: str
) -> None:
    for name in names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def validate_finite(
    parser: argparse.ArgumentParser,
    value: float,
    name: str,
    *,
    minimum: float = 0.0,
    inclusive: bool = True,
) -> None:
    if not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
        relation = "at least" if inclusive else "greater than"
        parser.error(f"--{name} must be finite and {relation} {minimum:g}")


def validate_gpu_utilization(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    value = args.gpu_memory_utilization
    validate_finite(parser, value, "gpu-memory-utilization", minimum=0.0, inclusive=False)
    if value > 1.0:
        parser.error("--gpu-memory-utilization must not exceed 1")


def normalize_paths(args: argparse.Namespace, *names: str) -> argparse.Namespace:
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, os.path.expanduser(value))
    return args


def add_result_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--result-file", help="save a structured experiment result JSON")


def run_nano_greedy(engine, prompt, max_new_tokens: int) -> list[int]:
    from nanovllm import SamplingParams

    result = engine.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True),
        use_tqdm=False,
    )
    return result[0]["token_ids"]


def validate_model_family(model_path: str) -> bool:
    from transformers import AutoConfig

    from nanovllm.config import is_qwen3_5_moe

    return is_qwen3_5_moe(AutoConfig.from_pretrained(model_path))


def write_result(path: str | Path, record: dict) -> None:
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(payload + "\n", encoding="utf-8")


def cuda_memory_stats() -> dict[str, float]:
    import torch

    return {
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "cuda_free_gib_after": torch.cuda.mem_get_info()[0] / 1024**3,
    }


def kv_cache_stats(engine) -> dict[str, float | int]:
    cache = engine.model_runner.kv_cache
    return {
        "kv_cache_blocks": engine.model_runner.num_kv_cache_blocks,
        "kv_cache_gib": cache.numel() * cache.element_size() / 1024**3,
    }


def execute_fixed_decode_step(engine, batch_size: int, phase: str) -> None:
    _, stats = engine.step()
    if stats.prefill_tokens or stats.decode_tokens != batch_size:
        raise RuntimeError(
            f"{phase} batch drift: prefill={stats.prefill_tokens}, "
            f"decode={stats.decode_tokens}, expected={batch_size}"
        )


def prepare_fixed_decode_batch(engine, prompt, sampling, batch_size):
    """Build a fixed active decode batch without decoding earlier requests."""
    if not engine.is_finished():
        raise RuntimeError("fixed decode setup requires an idle engine")
    scheduler = engine.scheduler
    if not 0 < batch_size <= min(scheduler.max_num_seqs, scheduler.max_num_batched_tokens):
        raise ValueError("fixed decode batch exceeds scheduler limits")
    if sampling.max_tokens < 2:
        raise ValueError("fixed decode requires at least two generated tokens")

    required = scheduler.cache_manager.block_manager.required_blocks(
        len(prompt) + sampling.max_tokens - 1
    )
    if batch_size * required > scheduler.cache_manager.block_manager.num_blocks:
        raise ValueError("fixed decode batch exceeds KV capacity")
    for _ in range(batch_size):
        engine.add_request(prompt, sampling)

    parked = []
    try:
        while scheduler.waiting or scheduler.prefilling:
            scheduled = scheduler.schedule()
            if not scheduled.prefill_chunks:
                raise RuntimeError("fixed decode setup cannot advance prefill")
            engine._run_batch(scheduled.prefill_chunks, True)
            # Completed prefills must not consume decode budget during setup.
            parked.extend(scheduler.running)
            scheduler.running.clear()
    finally:
        scheduler.running.extend(parked)
    if len(scheduler.running) != batch_size:
        raise RuntimeError("fixed decode setup expected a full prefilled batch")


def execute(
    experiment: str,
    args: argparse.Namespace,
    run: Callable[[argparse.Namespace], dict],
) -> int:
    path = getattr(args, "result_file", None)

    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    record = {
        "schema_version": 1,
        "experiment": experiment,
        "started_at": now(),
        "finished_at": None,
        "execution_status": "running",
        "config": vars(args).copy(),
        "result": None,
        "error": None,
    }
    try:
        record.update(execution_status="completed", result=run(args))
        code = 0
    except ValidationFailure as exc:
        record.update(
            execution_status="validation_failed",
            result=exc.result,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        print(str(exc), file=sys.stderr)
        code = 1
    except KeyboardInterrupt as exc:
        record.update(
            execution_status="interrupted",
            error={"type": type(exc).__name__, "message": "interrupted"},
        )
        code = 130
    except Exception as exc:
        record.update(
            execution_status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        record["finished_at"] = now()
        if path:
            write_result(path, record)
    return code
