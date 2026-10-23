"""Standalone joint-prefix continuation comparison."""

import argparse
import gc
from math import lcm
from unittest.mock import patch

from scripts._common import (
    ValidationFailure,
    add_kv_cache_arguments,
    add_model_argument,
    add_result_argument,
    compare_tokens,
    execute,
    kv_cache_kwargs,
    mismatch_message,
    normalize_paths,
    validate_model_family,
    validate_positive,
)


def _build_prefix_probe_prompts(tokenizer, shared_prefix_len: int) -> tuple[list[int], list[int]]:
    seed = tokenizer.encode(
        "Hybrid inference keeps KV and recurrent state aligned. ", add_special_tokens=False
    )
    repeats = (shared_prefix_len + len(seed) - 1) // len(seed)
    shared = (seed * repeats)[:shared_prefix_len]
    suffix_a = tokenizer.encode(" Reference branch A.", add_special_tokens=False)
    suffix_b = tokenizer.encode(" Reference branch B.", add_special_tokens=False)
    return shared + suffix_a, shared + suffix_b


def validate_prefix_resume_result(
    baseline_tokens: list[int],
    resumed_tokens: list[int],
    expected_tokens: int,
    metadata: dict,
    restore_boundary: int,
) -> dict:
    """Validate a completed continuation without mocking an engine or cache."""
    comparison = compare_tokens(baseline_tokens, resumed_tokens)
    full_length = len(baseline_tokens) == len(resumed_tokens) == expected_tokens
    result = {
        "baseline_tokens": baseline_tokens,
        "resumed_tokens": resumed_tokens,
        "comparison": comparison,
        "expected_tokens": expected_tokens,
        "full_length": full_length,
        "prefix": metadata,
        "restore_boundary": restore_boundary,
    }
    if not full_length:
        raise ValidationFailure(
            f"FAIL: expected {expected_tokens} greedy tokens, "
            f"got baseline={len(baseline_tokens)}, resumed={len(resumed_tokens)}",
            result,
        )
    if not comparison["matched"]:
        raise ValidationFailure(
            mismatch_message(
                comparison,
                failure_prefix="FAIL: prefix resume changed greedy decode at token",
                reference_name="baseline",
                actual_name="resumed",
            ),
            result,
        )
    return result


def run_prefix_resume_probe(
    model_path: str,
    max_new_tokens: int,
    max_model_len: int,
    block_size: int,
    kv_options: dict,
    shared_prefix_len: int | None = None,
    prefill_budget: int | None = None,
) -> dict:
    import torch
    from transformers import AutoTokenizer

    from nanovllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    shared_prefix_len = shared_prefix_len or lcm(64, block_size)
    prefill_budget = prefill_budget or block_size
    joint_step = lcm(64, block_size)
    if shared_prefix_len < joint_step or shared_prefix_len % joint_step:
        raise ValueError(f"shared prefix must align to the {joint_step}-token joint boundary")
    if prefill_budget <= 0:
        raise ValueError("prefill budget must be positive")
    prompt_a, prompt_b = _build_prefix_probe_prompts(tokenizer, shared_prefix_len)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)

    baseline = LLM(
        model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=prefill_budget,
        max_num_seqs=1,
        kv_cache_block_size=block_size,
        **kv_options,
        max_prefix_cache_entries=0,
    )
    try:
        baseline_tokens = baseline.generate([prompt_b], params, use_tqdm=False)[0]["token_ids"]
    finally:
        baseline.exit()
    gc.collect()
    torch.cuda.empty_cache()

    cached = LLM(
        model_path,
        **kv_options,
        max_model_len=max_model_len,
        max_num_batched_tokens=prefill_budget,
        max_num_seqs=1,
        kv_cache_block_size=block_size,
        max_prefix_cache_entries=4,
    )
    try:
        cached.generate(
            [prompt_a],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        from nanovllm.engine.sequence import Sequence

        prefix_cache = cached.scheduler.cache_manager.prefix_cache
        entry = prefix_cache.find(Sequence(prompt_b), max_tokens=len(prompt_b) - 1)
        metadata = (
            None
            if entry is None
            else {"num_tokens": entry.num_tokens, "snapshot_bytes": prefix_cache.snapshot_nbytes}
        )
        if metadata is None or metadata["num_tokens"] != shared_prefix_len:
            raise ValidationFailure(
                "FAIL: prefix-resume probe did not publish expected prefix", {"prefix": metadata}
            )
        from nanovllm.engine import cache_runtime

        original_restore = cache_runtime.GDNRestoreStaging.restore
        restored = []

        def observe_restore(staging, model, pending):
            for seq, snapshot in pending:
                restored.append(
                    (
                        seq.committed_tokens,
                        tuple(seq.block_table[: len(entry.block_ids)]),
                        snapshot is entry.state_snapshot,
                    )
                )
            return original_restore(staging, model, pending)

        with patch.object(cache_runtime.GDNRestoreStaging, "restore", observe_restore):
            resumed_tokens = cached.generate([prompt_b], params, use_tqdm=False)[0]["token_ids"]
        if restored != [(shared_prefix_len, entry.block_ids, True)]:
            raise ValidationFailure(
                "FAIL: prefix-resume did not restore the cached KV/GDN boundary",
                {"prefix": metadata, "restored": restored},
            )
    finally:
        cached.exit()

    result = validate_prefix_resume_result(
        baseline_tokens, resumed_tokens, max_new_tokens, metadata, restored[0][0]
    )
    result["shared_prefix_len"] = shared_prefix_len
    result["prefill_budget"] = prefill_budget
    result["kv_block_size"] = block_size
    print("PASS: prefix resume preserves greedy token sequence")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare uncached and prefix-restored greedy generation."
    )
    add_model_argument(parser, required=True)
    add_kv_cache_arguments(parser)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prefix-block-size", type=int, choices=(16, 32, 64, 128), default=16)
    parser.add_argument(
        "--shared-prefix-len",
        type=int,
        default=0,
        help="zero uses the first joint KV/GDN checkpoint; otherwise a multiple of lcm(64, block size)",
    )
    parser.add_argument(
        "--prefill-budget",
        type=int,
        default=0,
        help="tokens per scheduling step; 0 uses the KV block size (65 can exercise off-grid recovery)",
    )
    add_result_argument(parser)
    args = parser.parse_args(argv)
    validate_positive(parser, args, "max_new_tokens")
    if args.shared_prefix_len < 0 or args.prefill_budget < 0:
        parser.error("--shared-prefix-len and --prefill-budget must be non-negative")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must leave room for the prefix prompt")
    return normalize_paths(args, "model", "result_file")


def run(args):
    validate_model_family(args.model)
    return run_prefix_resume_probe(
        args.model,
        args.max_new_tokens,
        args.max_model_len,
        args.prefix_block_size,
        kv_cache_kwargs(args),
        shared_prefix_len=(args.shared_prefix_len or None),
        prefill_budget=(args.prefill_budget or None),
    )


def main(argv=None):
    return execute("prefix", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
