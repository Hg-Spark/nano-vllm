from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm import ImagePrompt

from scripts._common import (
    add_hf_experts_backend_argument,
    add_kv_cache_arguments,
    add_model_argument,
    add_result_argument,
    execute,
    kv_cache_kwargs,
    normalize_paths,
    require_greedy_parity,
    run_hf_reference,
    run_nano_greedy,
    validate_model_family,
    validate_positive,
)


def build_messages(image, prompt: str):
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }
    ]


def run_hf(
    model_path,
    processor,
    image,
    prompt,
    max_new_tokens,
    hf_experts_backend: str | None = "eager",
) -> list[int]:
    import torch
    from transformers import AutoModelForImageTextToText

    def generate(model):
        inputs = processor.apply_chat_template(
            build_messages(image, prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_len = inputs.input_ids.shape[1]
        inputs = inputs.to("cuda")
        model.generation_config.eos_token_id = None
        with torch.inference_mode():
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        return output_ids[0, input_len:].tolist()

    return run_hf_reference(
        AutoModelForImageTextToText, model_path, hf_experts_backend, generate
    )


def build_nano_prompt(processor, image, prompt: str) -> ImagePrompt:
    from nanovllm import ImagePrompt

    text = processor.apply_chat_template(
        build_messages(image, prompt), tokenize=False, add_generation_prompt=True
    )
    if text.count(processor.image_token) != 1:
        raise RuntimeError("Qwen chat template must contain exactly one image placeholder")
    return ImagePrompt(text=text, image=image)


def run_nanovllm(
    model_path: str,
    processor,
    image,
    prompt: str,
    max_new_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_visual_tokens_per_request: int | None,
    use_decode_graph: bool,
    kv_options: dict,
) -> list[int]:
    import torch

    from nanovllm import LLM

    llm = LLM(
        model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_visual_tokens_per_request=max_visual_tokens_per_request,
        max_num_seqs=1,
        decode_graph_batch_sizes=(1,) if use_decode_graph else (),
        **kv_options,
    )
    try:
        request = build_nano_prompt(processor, image, prompt)
        tokens = run_nano_greedy(llm, request, max_new_tokens)
        torch.cuda.synchronize()
        if use_decode_graph and not any(
            bucket.replay_count > 0 for bucket in llm.model_runner.decode_graph_buckets.values()
        ):
            raise RuntimeError("decode graph was requested but no replay occurred")
        return tokens
    finally:
        llm.exit()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare one-image Qwen3.5 greedy generation.")
    add_model_argument(parser, required=True)
    add_kv_cache_arguments(parser)
    add_hf_experts_backend_argument(parser)
    parser.add_argument("image")
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens-per-request", type=int, default=None)
    parser.add_argument("--decode-graph", action="store_true")
    add_result_argument(parser)
    args = parser.parse_args(argv)
    validate_positive(parser, args, "max_new_tokens")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must leave room for the image prompt")
    return normalize_paths(args, "model", "image", "result_file")


def run(args):
    from PIL import Image
    from transformers import AutoProcessor

    is_moe = validate_model_family(args.model)
    hf_backend = args.hf_experts_backend if is_moe else None
    processor = AutoProcessor.from_pretrained(args.model)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    try:
        hf_tokens = run_hf(
            args.model, processor, image, args.prompt, args.max_new_tokens, hf_backend
        )
        nano_tokens = run_nanovllm(
            args.model,
            processor,
            image,
            args.prompt,
            args.max_new_tokens,
            args.max_model_len,
            args.max_num_batched_tokens,
            args.max_visual_tokens_per_request,
            args.decode_graph,
            kv_cache_kwargs(args),
        )
    finally:
        image.close()

    print(f"HF tokens:   {hf_tokens}\nNano tokens: {nano_tokens}")
    return require_greedy_parity(
        hf_tokens,
        nano_tokens,
        args.max_new_tokens,
        hf_backend,
        mismatch_prefix="FAIL: first multimodal mismatch at completion token",
        decode_graph_requested=args.decode_graph,
    )


def main(argv=None):
    return execute("image_parity", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
