import argparse
import gc
import os

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from nanovllm import LLM, SamplingParams


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
    # Compare a fixed token count. Official generation configs may contain
    # multiple stop-token ids while nano-vLLM currently exposes one EOS id.
    model.generation_config.eos_token_id = None
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    completion = output_ids[0, input_ids.size(1) :].tolist()

    del output_ids, input_ids, model
    gc.collect()
    torch.cuda.empty_cache()
    return completion


def run_nanovllm(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    max_model_len: int,
    max_num_state_slots: int,
) -> list[int]:
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=max_num_state_slots,
        max_num_state_slots=max_num_state_slots,
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "--prompt",
        default="Explain why KV cache is useful in autoregressive inference.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-state-slots", type=int, default=4)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
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
        args.max_num_state_slots,
    )

    common = min(len(hf_tokens), len(nano_tokens))
    mismatch = next(
        (
            idx
            for idx in range(common)
            if hf_tokens[idx] != nano_tokens[idx]
        ),
        None,
    )

    print(f"HF tokens:   {hf_tokens}")
    print(f"Nano tokens: {nano_tokens}")
    if mismatch is None and hf_tokens == nano_tokens:
        print("PASS: greedy token sequences match exactly")
        return

    if mismatch is None:
        mismatch = common
    hf_token = hf_tokens[mismatch] if mismatch < len(hf_tokens) else None
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
