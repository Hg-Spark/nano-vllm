import argparse
from contextlib import contextmanager

from scripts._common import (
    ValidationFailure,
    add_dtype_argument,
    add_hf_experts_backend_argument,
    add_model_argument,
    add_result_argument,
    execute,
    normalize_paths,
    require_greedy_parity,
    run_hf_reference,
    run_nano_greedy,
    validate_finite,
    validate_model_family,
    validate_positive,
)


@contextmanager
def _capture_last_logits(model):
    logits = []

    def capture(_module, _inputs, output):
        logits.append(output.detach().reshape(-1, output.size(-1))[-1].float().cpu())

    handle = model.lm_head.register_forward_hook(capture)
    try:
        yield logits
    finally:
        handle.remove()


def compare_logits(hf_logits, nano_logits, threshold: float) -> dict:
    import torch

    result = {"relative_rms_threshold": threshold, "per_step": []}
    if not hf_logits or len(hf_logits) != len(nano_logits):
        raise ValidationFailure(
            f"FAIL: logits step count differs: HF={len(hf_logits)}, Nano={len(nano_logits)}",
            result,
        )

    for step, (hf, nano) in enumerate(zip(hf_logits, nano_logits)):
        if hf.ndim != 1 or hf.shape != nano.shape or hf.numel() == 0:
            raise ValidationFailure(f"FAIL: logits shape mismatch at step {step}", result)
        if not torch.isfinite(hf).all().item() or not torch.isfinite(nano).all().item():
            raise ValidationFailure(f"FAIL: non-finite logits at step {step}", result)

        diff = nano - hf
        relative_rms = (
            diff.square().mean().sqrt() / hf.square().mean().sqrt().clamp_min(1e-12)
        ).item()
        max_abs = diff.abs().max().item()
        result["per_step"].append({"step": step, "relative_rms": relative_rms, "max_abs": max_abs})
        if relative_rms > threshold:
            raise ValidationFailure(
                f"FAIL: logits at step {step}: relative RMS={relative_rms:.4g} "
                f"exceeds {threshold:g} (max_abs={max_abs:.4g})",
                result,
            )

    result["max_relative_rms"] = max(step["relative_rms"] for step in result["per_step"])
    result["max_abs"] = max(step["max_abs"] for step in result["per_step"])
    return result


def run_hf(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    dtype: str,
    hf_experts_backend: str | None = "eager",
) -> tuple[list[int], list]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def generate(model):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        model.generation_config.eos_token_id = None
        with torch.inference_mode(), _capture_last_logits(model) as logits:
            output_ids = model.generate(
                input_ids=input_ids, do_sample=False, max_new_tokens=max_new_tokens
            )
        return output_ids[0, input_ids.size(1) :].tolist(), logits

    return run_hf_reference(AutoModelForCausalLM, model_path, dtype, hf_experts_backend, generate)


def run_nanovllm(
    model_path: str, prompt: str, max_new_tokens: int, max_model_len: int, dtype: str
) -> tuple[list[int], list]:
    from nanovllm import LLM

    llm = LLM(model_path, dtype=dtype, max_model_len=max_model_len, max_num_seqs=1)
    try:
        with _capture_last_logits(llm.model_runner.model) as logits:
            tokens = run_nano_greedy(llm, prompt, max_new_tokens)
        return tokens, logits
    finally:
        llm.exit()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_model_argument(parser, required=True)
    add_dtype_argument(parser)
    add_hf_experts_backend_argument(parser)
    parser.add_argument(
        "--prompt", default="Explain why KV cache is useful in autoregressive inference."
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--logits-relative-rms-threshold", type=float, default=0.05)
    add_result_argument(parser)
    args = parser.parse_args(argv)
    validate_positive(parser, args, "max_new_tokens")
    validate_finite(
        parser,
        args.logits_relative_rms_threshold,
        "logits-relative-rms-threshold",
        minimum=0.0,
        inclusive=False,
    )
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must leave room for the prompt")
    return normalize_paths(args, "model", "result_file")


def run(args):
    is_moe = validate_model_family(args.model)
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.model)
    text_cfg = getattr(cfg, "text_config", cfg)
    effective_len = min(args.max_model_len, text_cfg.max_position_embeddings)
    prompt_len = len(AutoTokenizer.from_pretrained(args.model, use_fast=True).encode(args.prompt))
    if prompt_len + args.max_new_tokens > effective_len:
        raise ValueError(
            f"prompt ({prompt_len}) + max_new_tokens ({args.max_new_tokens}) "
            f"exceeds effective max_model_len ({effective_len})"
        )
    hf_backend = args.hf_experts_backend if is_moe else None
    hf_tokens, hf_logits = run_hf(
        args.model, args.prompt, args.max_new_tokens, args.dtype, hf_backend
    )
    nano_tokens, nano_logits = run_nanovllm(
        args.model, args.prompt, args.max_new_tokens, args.max_model_len, args.dtype
    )
    print(f"HF tokens:   {hf_tokens}\nNano tokens: {nano_tokens}")
    result = require_greedy_parity(hf_tokens, nano_tokens, args.max_new_tokens, hf_backend)
    result["logits_comparison"] = compare_logits(
        hf_logits, nano_logits, args.logits_relative_rms_threshold
    )
    print("PASS: greedy token sequences match exactly")
    print(
        f"PASS: logits relative RMS <= {args.logits_relative_rms_threshold:g} "
        f"(worst={result['logits_comparison']['max_relative_rms']:.4g})"
    )
    return result


def main(argv=None):
    return execute("text_parity", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
