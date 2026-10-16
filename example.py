import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    model = os.environ.get("NANOVLLM_MODEL")
    if not model:
        raise SystemExit("Set NANOVLLM_MODEL to a Qwen3.5 Dense or MoE checkpoint directory")
    dtype = os.environ.get("NANOVLLM_DTYPE")
    if dtype not in ("float16", "bfloat16"):
        raise SystemExit("Set NANOVLLM_DTYPE to float16 or bfloat16")
    path = os.path.expanduser(model)
    tokenizer = AutoTokenizer.from_pretrained(path)

    llm = LLM(
        path,
        dtype=dtype,
        max_num_batched_tokens=256,
        max_num_seqs=2,
    )
    prompts = [
        "Explain why KV cache helps autoregressive inference.",
        "What makes a sparse MoE layer different from a dense FFN?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=64,
        ),
    )
    for output in outputs:
        print(output["text"])

    llm.exit()


if __name__ == "__main__":
    main()
