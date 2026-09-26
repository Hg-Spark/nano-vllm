import argparse
import gc
import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoTokenizer,
)

from nanovllm import LLM, SamplingParams


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    common = min(len(left), len(right))
    mismatch = next(
        (idx for idx in range(common) if left[idx] != right[idx]),
        None,
    )
    if mismatch is not None:
        return mismatch
    if len(left) != len(right):
        return common
    return None


def validate_model_family(model_path: str) -> None:
    config = AutoConfig.from_pretrained(model_path)
    text_config = getattr(config, "text_config", config)
    if (
        getattr(config, "model_type", None) != "qwen3_5_moe"
        and getattr(text_config, "model_type", None)
        != "qwen3_5_moe_text"
    ):
        raise ValueError(
            "verification requires a Qwen3.5-MoE checkpoint"
        )


def run_hf(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.cuda()

    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ).cuda()
    model.generation_config.eos_token_id = None

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    completion = output_ids[
        0,
        input_ids.size(1):,
    ].tolist()

    del output_ids, input_ids, model
    gc.collect()
    torch.cuda.empty_cache()
    return completion


def run_nanovllm(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    max_model_len: int,
) -> list[int]:
    llm = LLM(
        model_path,
        max_model_len=max_model_len,
        max_num_seqs=1,
    )
    try:
        output = llm.generate(
            [prompt],
            SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        return output[0]["token_ids"]
    finally:
        llm.exit()


def _build_prefix_probe_prompts(
    tokenizer,
    block_size: int,
) -> tuple[list[int], list[int]]:
    seed = tokenizer.encode(
        "Hybrid inference keeps KV and recurrent state aligned. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer produced an empty prefix-probe seed")
    repeats = (block_size + len(seed) - 1) // len(seed)
    shared = (seed * repeats)[:block_size]
    suffix_a = tokenizer.encode(
        " Reference branch A.",
        add_special_tokens=False,
    )
    suffix_b = tokenizer.encode(
        " Reference branch B.",
        add_special_tokens=False,
    )
    if not suffix_a or not suffix_b or suffix_a == suffix_b:
        raise RuntimeError("prefix-probe suffixes must be non-empty and distinct")
    return shared + suffix_a, shared + suffix_b


def _snapshot_nbytes(snapshot) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in snapshot.layers
        for tensor in layer
    )


def run_prefix_resume_probe(
    model_path: str,
    max_new_tokens: int,
    max_model_len: int,
    block_size: int,
) -> None:
    if block_size <= 0 or block_size % 256 != 0:
        raise ValueError("prefix block size must be a positive multiple of 256")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    prompt_a, prompt_b = _build_prefix_probe_prompts(
        tokenizer,
        block_size,
    )
    if len(prompt_b) + max_new_tokens > max_model_len:
        raise ValueError(
            "prefix-resume probe exceeds max_model_len: "
            f"prompt={len(prompt_b)}, output={max_new_tokens}, "
            f"max={max_model_len}"
        )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        ignore_eos=True,
    )

    baseline = LLM(
        model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=block_size,
        max_num_seqs=1,
        kvcache_block_size=block_size,
        max_prefix_cache_entries=0,
    )
    try:
        baseline_tokens = baseline.generate(
            [prompt_b],
            params,
            use_tqdm=False,
        )[0]["token_ids"]
    finally:
        baseline.exit()
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    cached = LLM(
        model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=block_size,
        max_num_seqs=1,
        kvcache_block_size=block_size,
        max_prefix_cache_entries=4,
    )
    try:
        cached.generate(
            [prompt_a],
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        entry = cached.scheduler.prefix_runtime.cache.longest_match(
            prompt_b,
            max_tokens=len(prompt_b) - 1,
        )
        if entry is None:
            raise SystemExit(
                "FAIL: prefix-resume probe did not publish a reusable prefix"
            )
        if entry.num_tokens != block_size:
            raise SystemExit(
                "FAIL: unexpected reusable prefix boundary: "
                f"expected={block_size}, actual={entry.num_tokens}"
            )
        snapshot_bytes = _snapshot_nbytes(entry.state_snapshot)
        resumed_tokens = cached.generate(
            [prompt_b],
            params,
            use_tqdm=False,
        )[0]["token_ids"]
    finally:
        cached.exit()

    mismatch = first_mismatch(baseline_tokens, resumed_tokens)
    print(
        "Prefix resume: "
        f"boundary={block_size} tokens, "
        f"snapshot={snapshot_bytes / (1024 ** 2):.2f} MiB"
    )
    print(f"Uncached tokens: {baseline_tokens}")
    print(f"Resumed tokens:  {resumed_tokens}")
    if mismatch is not None:
        baseline_token = (
            baseline_tokens[mismatch]
            if mismatch < len(baseline_tokens)
            else None
        )
        resumed_token = (
            resumed_tokens[mismatch]
            if mismatch < len(resumed_tokens)
            else None
        )
        raise SystemExit(
            "FAIL: BF16 prefix resume changed greedy decode at token "
            f"{mismatch}: baseline={baseline_token}, resumed={resumed_token}"
        )
    print("PASS: BF16 prefix resume preserves greedy token sequence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "--prompt",
        default=(
            "Explain why KV cache is useful in "
            "autoregressive inference."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--check-prefix-resume",
        action="store_true",
        help="compare uncached decode with BF16 GDN prefix restore",
    )
    parser.add_argument(
        "--prefix-block-size",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    validate_model_family(model_path)

    hf_tokens = run_hf(
        model_path,
        args.prompt,
        args.max_new_tokens,
    )
    nano_tokens = run_nanovllm(
        model_path,
        args.prompt,
        args.max_new_tokens,
        args.max_model_len,
    )

    mismatch = first_mismatch(hf_tokens, nano_tokens)

    print(f"HF tokens:   {hf_tokens}")
    print(f"Nano tokens: {nano_tokens}")
    if mismatch is None:
        print("PASS: greedy token sequences match exactly")
        if args.check_prefix_resume:
            run_prefix_resume_probe(
                model_path,
                args.max_new_tokens,
                args.max_model_len,
                args.prefix_block_size,
            )
        return
    hf_token = (
        hf_tokens[mismatch]
        if mismatch < len(hf_tokens)
        else None
    )
    nano_token = (
        nano_tokens[mismatch]
        if mismatch < len(nano_tokens)
        else None
    )
    raise SystemExit(
        "FAIL: first mismatch at completion token "
        f"{mismatch}: hf={hf_token}, nano={nano_token}"
    )


if __name__ == "__main__":
    main()
