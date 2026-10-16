import argparse

from scripts._common import (
    add_dtype_argument,
    add_hf_experts_backend_argument,
    add_model_argument,
    add_result_argument,
    execute,
    normalize_paths,
    require_greedy_parity,
    run_hf_reference,
    run_nano_greedy,
    validate_model_family,
    validate_positive,
)


def run_hf(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    dtype: str,
    hf_experts_backend: str | None = "eager",
) -> list[int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def generate(model):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        model.generation_config.eos_token_id = None
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids, do_sample=False, max_new_tokens=max_new_tokens
            )
        return output_ids[0, input_ids.size(1) :].tolist()

    return run_hf_reference(AutoModelForCausalLM, model_path, dtype, hf_experts_backend, generate)


def run_nanovllm(
    model_path: str, prompt: str, max_new_tokens: int, max_model_len: int, dtype: str
) -> list[int]:
    from nanovllm import LLM

    llm = LLM(model_path, dtype=dtype, max_model_len=max_model_len, max_num_seqs=1)
    try:
        return run_nano_greedy(llm, prompt, max_new_tokens)
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
    add_result_argument(parser)
    args = parser.parse_args(argv)
    validate_positive(parser, args, "max_new_tokens")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must leave room for the prompt")
    return normalize_paths(args, "model", "result_file")


def run(args):
    is_moe = validate_model_family(args.model)
    hf_backend = args.hf_experts_backend if is_moe else None
    hf_tokens = run_hf(args.model, args.prompt, args.max_new_tokens, args.dtype, hf_backend)
    nano_tokens = run_nanovllm(
        args.model, args.prompt, args.max_new_tokens, args.max_model_len, args.dtype
    )
    print(f"HF tokens:   {hf_tokens}\nNano tokens: {nano_tokens}")
    result = require_greedy_parity(hf_tokens, nano_tokens, args.max_new_tokens, hf_backend)
    print("PASS: greedy token sequences match exactly")
    return result


def main(argv=None):
    return execute("text_parity", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
