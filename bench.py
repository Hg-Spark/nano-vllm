import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def main():
    """Reference-path microbenchmark.

    This measures the current eager Qwen3.5-MoE implementation only. It is not
    intended as a vLLM/SGLang performance comparison until fused GDN/MoE
    kernels are implemented.
    """
    seed(0)
    path = os.path.expanduser(
        os.environ.get(
            "NANOVLLM_MODEL",
            "~/huggingface/Qwen3.5-35B-A3B/",
        )
    )
    num_seqs = 2
    max_input_len = 128
    max_output_len = 32

    llm = LLM(
        path,
        max_num_batched_tokens=256,
        max_num_seqs=num_seqs,
        max_num_state_slots=num_seqs,
    )
    prompt_token_ids = [
        [
            randint(0, 10000)
            for _ in range(
                randint(64, max_input_len)
            )
        ]
        for _ in range(num_seqs)
    ]
    params = [
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=max_output_len,
        )
        for _ in range(num_seqs)
    ]

    start = time.time()
    llm.generate(
        prompt_token_ids,
        params,
        use_tqdm=False,
    )
    elapsed = time.time() - start
    total_tokens = num_seqs * max_output_len
    print(
        f"Decode output: {total_tokens} tok, "
        f"time={elapsed:.2f}s, "
        f"throughput={total_tokens / elapsed:.2f} tok/s"
    )
    llm.exit()


if __name__ == "__main__":
    main()
