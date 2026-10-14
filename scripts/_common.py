"""CLI helpers, comparisons and optional result records."""

import argparse
import json
import math
import os
import sys
import tempfile
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


def add_dtype_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        required=True,
        help="model compute dtype (must be selected explicitly)",
    )


def add_hf_experts_backend_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hf-experts-backend",
        choices=("eager", "batched_mm", "grouped_mm"),
        default="eager",
        help="explicit Transformers expert backend; eager is the reference baseline",
    )


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

    from nanovllm.models.family import is_qwen3_5_moe

    return is_qwen3_5_moe(AutoConfig.from_pretrained(model_path))


def write_result(path: str | Path, record: dict) -> None:
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def execute(
    experiment: str,
    args: argparse.Namespace,
    run: Callable[[argparse.Namespace], dict],
) -> int:
    path = getattr(args, "result_file", None)
    now = lambda: datetime.now(timezone.utc).isoformat()
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
