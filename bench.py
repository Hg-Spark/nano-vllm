import argparse
import json
import math
import os
import time
from random import Random

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p99": 0.0,
        }
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p99": percentile(values, 99),
    }


def build_sequences(
    llm: LLM,
    num_requests: int,
    min_input_len: int,
    max_input_len: int,
    output_len: int,
    seed: int,
) -> list[Sequence]:
    rng = Random(seed)
    vocab_size = llm.model_runner.hf_config.vocab_size
    params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=output_len,
    )
    sequences = []
    for _ in range(num_requests):
        length = rng.randint(min_input_len, max_input_len)
        token_ids = [
            rng.randrange(vocab_size)
            for _ in range(length)
        ]
        sequences.append(Sequence(token_ids, params))
    return sequences


def run_benchmark(
    llm: LLM,
    sequences: list[Sequence],
) -> dict:
    for seq in sequences:
        llm.scheduler.add(seq)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    first_token_at: dict[int, float] = {}
    finished_at: dict[int, float] = {}
    prefill_tokens = 0
    decode_tokens = 0
    prefill_seconds = 0.0
    decode_seconds = 0.0
    steps = 0

    def record_request_times(now: float) -> None:
        for seq in sequences:
            if (
                seq.seq_id not in first_token_at
                and seq.num_completion_tokens > 0
            ):
                first_token_at[seq.seq_id] = now
            if seq.is_finished and seq.seq_id not in finished_at:
                finished_at[seq.seq_id] = now

    while not llm.is_finished():
        scheduled = llm.scheduler.schedule()
        step_decode_tokens = scheduled.decode_tokens
        step_prefill_tokens = scheduled.prefill_tokens
        steps += 1

        if scheduled.decode_seqs:
            torch.cuda.synchronize()
            phase_start = time.perf_counter()
            try:
                llm._run_batch(
                    scheduled.decode_seqs,
                    False,
                )
            except Exception:
                if scheduled.prefill_seqs:
                    llm.scheduler.recover_failed_step(
                        scheduled.prefill_seqs
                    )
                raise
            torch.cuda.synchronize()
            phase_end = time.perf_counter()
            decode_tokens += step_decode_tokens
            decode_seconds += phase_end - phase_start
            record_request_times(phase_end)

        if scheduled.prefill_seqs:
            torch.cuda.synchronize()
            phase_start = time.perf_counter()
            llm._run_batch(
                scheduled.prefill_seqs,
                True,
            )
            torch.cuda.synchronize()
            phase_end = time.perf_counter()
            prefill_tokens += step_prefill_tokens
            prefill_seconds += phase_end - phase_start
            record_request_times(phase_end)

    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = end - start

    missing_first = [
        seq.seq_id
        for seq in sequences
        if seq.seq_id not in first_token_at
    ]
    missing_finish = [
        seq.seq_id
        for seq in sequences
        if seq.seq_id not in finished_at
    ]
    if missing_first or missing_finish:
        raise RuntimeError(
            "benchmark timing metadata is incomplete: "
            f"missing_first={missing_first}, missing_finish={missing_finish}"
        )

    ttft = [
        first_token_at[seq.seq_id] - start
        for seq in sequences
    ]
    e2e = [
        finished_at[seq.seq_id] - start
        for seq in sequences
    ]
    tpot = [
        (
            finished_at[seq.seq_id]
            - first_token_at[seq.seq_id]
        )
        / (seq.num_completion_tokens - 1)
        for seq in sequences
        if seq.num_completion_tokens > 1
    ]

    prompt_tokens = sum(
        seq.num_prompt_tokens
        for seq in sequences
    )
    output_tokens = sum(
        seq.num_completion_tokens
        for seq in sequences
    )

    return {
        "requests": len(sequences),
        "scheduler_steps": steps,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "request_throughput_rps": len(sequences) / elapsed,
        "output_throughput_tok_s": output_tokens / elapsed,
        "total_token_throughput_tok_s": (
            prompt_tokens + output_tokens
        ) / elapsed,
        "prefill_execution_tok_s": (
            prefill_tokens / prefill_seconds
            if prefill_seconds > 0
            else 0.0
        ),
        "decode_execution_tok_s": (
            decode_tokens / decode_seconds
            if decode_seconds > 0
            else 0.0
        ),
        "ttft_seconds": summarize(ttft),
        "tpot_seconds": summarize(tpot),
        "e2e_latency_seconds": summarize(e2e),
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        ),
    }


def print_summary(result: dict) -> None:
    print(
        "Requests={requests}, prompt={prompt_tokens} tok, "
        "output={output_tokens} tok, elapsed={elapsed_seconds:.3f}s".format(
            **result
        )
    )
    print(
        "Throughput: "
        f"{result['request_throughput_rps']:.2f} req/s, "
        f"{result['output_throughput_tok_s']:.2f} output tok/s, "
        f"{result['total_token_throughput_tok_s']:.2f} total tok/s"
    )
    print(
        "Execution: "
        f"prefill={result['prefill_execution_tok_s']:.2f} tok/s, "
        f"decode={result['decode_execution_tok_s']:.2f} tok/s"
    )
    for name, key in (
        ("TTFT", "ttft_seconds"),
        ("TPOT", "tpot_seconds"),
        ("E2E", "e2e_latency_seconds"),
    ):
        stats = result[key]
        print(
            f"{name}: mean={stats['mean'] * 1000:.2f} ms, "
            f"p50={stats['p50'] * 1000:.2f} ms, "
            f"p99={stats['p99'] * 1000:.2f} ms"
        )
    print(
        f"Peak CUDA memory: {result['peak_cuda_memory_gib']:.2f} GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-MoE eager serving benchmark"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=os.environ.get(
            "NANOVLLM_MODEL",
            "~/huggingface/Qwen3.5-35B-A3B/",
        ),
    )
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--min-input-len", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=512,
    )
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON record after the text summary",
    )
    args = parser.parse_args()

    if args.num_requests <= 0:
        raise ValueError("num-requests must be positive")
    if args.min_input_len <= 0:
        raise ValueError("min-input-len must be positive")
    if args.max_input_len < args.min_input_len:
        raise ValueError("max-input-len must be >= min-input-len")
    if args.output_len <= 0:
        raise ValueError("output-len must be positive")
    if args.max_input_len + args.output_len > args.max_model_len:
        raise ValueError(
            "max-input-len + output-len must not exceed max-model-len"
        )

    model_path = os.path.expanduser(args.model)
    llm = LLM(
        model_path,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_num_state_slots=args.max_num_seqs,
        enable_prefix_cache=not args.disable_prefix_cache,
    )
    try:
        sequences = build_sequences(
            llm,
            args.num_requests,
            args.min_input_len,
            args.max_input_len,
            args.output_len,
            args.seed,
        )
        result = run_benchmark(llm, sequences)
    finally:
        llm.exit()

    result["config"] = {
        "num_requests": args.num_requests,
        "min_input_len": args.min_input_len,
        "max_input_len": args.max_input_len,
        "output_len": args.output_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "prefix_cache": not args.disable_prefix_cache,
        "seed": args.seed,
    }
    print_summary(result)
    if args.json:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
