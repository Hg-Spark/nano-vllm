import argparse
import gc
import os

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from nanovllm import ImagePrompt, LLM, SamplingParams


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    common = min(len(left), len(right))
    for index in range(common):
        if left[index] != right[index]:
            return index
    if len(left) != len(right):
        return common
    return None


def build_messages(image, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_hf(
    model_path: str,
    processor,
    image,
    prompt: str,
    max_new_tokens: int,
) -> list[int]:
    messages = build_messages(image, prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_len = inputs.input_ids.shape[1]
    inputs = inputs.to("cuda")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ).cuda().eval()
    model.generation_config.eos_token_id = None

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    completion = output_ids[0, input_len:].tolist()

    del output_ids, inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    return completion


def build_nano_prompt(processor, image, prompt: str) -> ImagePrompt:
    messages = build_messages(image, prompt)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_token = processor.image_token
    if text.count(image_token) != 1:
        raise RuntimeError(
            "Qwen chat template must contain exactly one image placeholder "
            f"before processor expansion; found {text.count(image_token)}"
        )
    return ImagePrompt(text=text, image=image)


def run_nano(
    model_path: str,
    processor,
    image,
    prompt: str,
    max_new_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_decode_graph: bool,
) -> list[int]:
    graph_sizes = (1,) if use_decode_graph else ()
    llm = LLM(
        model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=1,
        decode_graph_batch_sizes=graph_sizes,
    )
    try:
        request = build_nano_prompt(
            processor,
            image,
            prompt,
        )
        output = llm.generate(
            [request],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one-image Qwen3.5-MoE greedy generation between "
            "Transformers and nano-vLLM, optionally using decode CUDA Graph."
        )
    )
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument(
        "--prompt",
        default="Describe this image briefly.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=256,
        help=(
            "Lower values force the expanded image prompt through "
            "multiple prefill chunks."
        ),
    )
    parser.add_argument(
        "--decode-graph",
        action="store_true",
        help="enable the B=1 fixed-address decode CUDA Graph bucket",
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    image_path = os.path.expanduser(args.image)
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.max_num_batched_tokens <= 0:
        raise ValueError("max-num-batched-tokens must be positive")

    processor = AutoProcessor.from_pretrained(model_path)
    image = Image.open(image_path).convert("RGB")

    hf_tokens = run_hf(
        model_path,
        processor,
        image,
        args.prompt,
        args.max_new_tokens,
    )
    nano_tokens = run_nano(
        model_path,
        processor,
        image,
        args.prompt,
        args.max_new_tokens,
        args.max_model_len,
        args.max_num_batched_tokens,
        args.decode_graph,
    )

    print(f"HF tokens:   {hf_tokens}")
    print(f"Nano tokens: {nano_tokens}")
    mismatch = first_mismatch(hf_tokens, nano_tokens)
    if mismatch is None:
        mode = "CUDA Graph decode" if args.decode_graph else "eager decode"
        print(
            "PASS: image prompt greedy tokens match exactly "
            f"with {mode}"
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
        "FAIL: first multimodal mismatch at completion token "
        f"{mismatch}: hf={hf_token}, nano={nano_token}"
    )


if __name__ == "__main__":
    main()
