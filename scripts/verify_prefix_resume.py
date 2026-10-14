"""Standalone joint-prefix continuation comparison."""

import argparse
import gc

from scripts._common import (
    ValidationFailure,
    add_dtype_argument,
    add_model_argument,
    add_result_argument,
    compare_tokens,
    execute,
    mismatch_message,
    normalize_paths,
    validate_model_family,
    validate_positive,
)
from scripts._probes import find_prefix_metadata


def _build_prefix_probe_prompts(tokenizer, block_size: int) -> tuple[list[int], list[int]]:
    seed = tokenizer.encode(
        "Hybrid inference keeps KV and recurrent state aligned. ", add_special_tokens=False
    )
    repeats = (block_size + len(seed) - 1) // len(seed)
    shared = (seed * repeats)[:block_size]
    suffix_a = tokenizer.encode(" Reference branch A.", add_special_tokens=False)
    suffix_b = tokenizer.encode(" Reference branch B.", add_special_tokens=False)
    return shared + suffix_a, shared + suffix_b


def run_prefix_resume_probe(
    model_path: str, max_new_tokens: int, max_model_len: int, block_size: int, dtype: str
) -> dict:
    import torch
    from transformers import AutoTokenizer

    from nanovllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    prompt_a, prompt_b = _build_prefix_probe_prompts(tokenizer, block_size)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)

    baseline = LLM(
        model_path,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_batched_tokens=block_size,
        max_num_seqs=1,
        kv_cache_block_size=block_size,
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
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_batched_tokens=block_size,
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
        metadata = find_prefix_metadata(cached, prompt_b)
        if metadata is None or metadata["num_tokens"] != block_size:
            raise ValidationFailure(
                "FAIL: prefix-resume probe did not publish expected prefix", {"prefix": metadata}
            )
        resumed_tokens = cached.generate([prompt_b], params, use_tqdm=False)[0]["token_ids"]
    finally:
        cached.exit()

    comparison = compare_tokens(baseline_tokens, resumed_tokens)
    result = {
        "baseline_tokens": baseline_tokens,
        "resumed_tokens": resumed_tokens,
        "comparison": comparison,
        "prefix": metadata,
    }
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
    print("PASS: prefix resume preserves greedy token sequence")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare uncached and prefix-restored greedy generation."
    )
    add_model_argument(parser, required=True)
    add_dtype_argument(parser)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prefix-block-size", type=int, choices=(16, 32, 64, 128), default=16)
    add_result_argument(parser)
    args = parser.parse_args(argv)
    validate_positive(parser, args, "max_new_tokens", "max_model_len")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must leave room for the prefix prompt")
    return normalize_paths(args, "model", "result_file")


def run(args):
    validate_model_family(args.model)
    return run_prefix_resume_probe(
        args.model, args.max_new_tokens, args.max_model_len, args.prefix_block_size, args.dtype
    )


def main(argv=None):
    return execute("prefix", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
