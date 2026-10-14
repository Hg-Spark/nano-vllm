from __future__ import annotations

import argparse
import json
import time
from random import Random
from statistics import quantiles
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm import LLM
    from nanovllm.engine.sequence import Sequence

from scripts._common import (
    add_dtype_argument,
    add_model_argument,
    add_result_argument,
    execute,
    normalize_paths,
    validate_positive,
)
from scripts._probes import enqueue_sequences, prepare_requests, vocabulary_size


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0}
    pct = quantiles(values, n=100, method="inclusive") if len(values) > 1 else [values[0]] * 99
    return {"mean": sum(values) / len(values), "p50": pct[49], "p99": pct[98]}


def build_sequences(
    llm: LLM,
    num_requests: int,
    min_input_len: int,
    max_input_len: int,
    output_len: int,
    seed: int,
) -> list[Sequence]:
    from nanovllm import SamplingParams

    rng = Random(seed)
    vocab_size = vocabulary_size(llm)
    params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
    prompts = [
        [rng.randrange(vocab_size) for _ in range(rng.randint(min_input_len, max_input_len))]
        for _ in range(num_requests)
    ]
    return prepare_requests(llm, prompts, params)


def run_benchmark(
    llm: LLM,
    sequences: list[Sequence],
    *,
    late_sequence: Sequence | None = None,
    late_after_decode_steps: int = 4,
) -> dict:
    import torch

    enqueue_sequences(llm, sequences)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    all_sequences = [*sequences, *([late_sequence] if late_sequence is not None else [])]
    arrival_at = {seq.seq_id: start for seq in sequences}
    short_ids = set(arrival_at)
    late_added = False
    decode_steps = 0
    short_gap_before: list[float] = []
    short_gap_after: list[float] = []
    previous_token_at: dict[int, float] = {}
    previous_token_count: dict[int, int] = {}
    first_token_at: dict[int, float] = {}
    finished_at: dict[int, float] = {}
    prefill_tokens = decode_tokens = max_prefill_step_tokens = max_decode_step_tokens = steps = 0
    prefill_seconds = decode_seconds = 0.0

    def record_request_times(now: float) -> None:
        for seq in all_sequences:
            count = seq.num_completion_tokens
            if count > previous_token_count.get(seq.seq_id, 0):
                previous = previous_token_at.get(seq.seq_id)
                if previous is not None and seq.seq_id in short_ids:
                    (short_gap_after if late_added else short_gap_before).append(now - previous)
                previous_token_count[seq.seq_id] = count
                previous_token_at[seq.seq_id] = now
                first_token_at.setdefault(seq.seq_id, now)
            if seq.is_finished and seq.seq_id not in finished_at:
                finished_at[seq.seq_id] = now

    while not llm.is_finished():
        _, stats = llm.step()
        # Benchmark user-visible completion only after both forwards have finished.
        torch.cuda.synchronize()
        now = time.perf_counter()
        record_request_times(now)
        steps += 1
        decode_tokens += stats.decode_tokens
        prefill_tokens += stats.prefill_tokens
        decode_seconds += stats.decode_seconds
        prefill_seconds += stats.prefill_seconds
        max_prefill_step_tokens = max(max_prefill_step_tokens, stats.prefill_tokens)
        max_decode_step_tokens = max(max_decode_step_tokens, stats.decode_tokens)
        if stats.decode_tokens:
            decode_steps += 1
        if late_sequence is not None and not late_added and decode_steps >= late_after_decode_steps:
            arrival_at[late_sequence.seq_id] = now
            llm.scheduler.add(late_sequence)
            late_added = True

    if late_sequence is not None and not late_added:
        raise RuntimeError("short requests ended before the long prompt could be injected")

    elapsed = time.perf_counter() - start
    ttft = [first_token_at[seq.seq_id] - arrival_at[seq.seq_id] for seq in all_sequences]
    e2e = [finished_at[seq.seq_id] - arrival_at[seq.seq_id] for seq in all_sequences]
    tpot = [
        (finished_at[seq.seq_id] - first_token_at[seq.seq_id]) / (seq.num_completion_tokens - 1)
        for seq in all_sequences
        if seq.num_completion_tokens > 1
    ]
    prompt_tokens = sum(seq.num_prompt_tokens for seq in all_sequences)
    output_tokens = sum(seq.num_completion_tokens for seq in all_sequences)

    num_requests = len(all_sequences)
    result = {
        "requests": num_requests,
        "scheduler_steps": steps,
        "max_prefill_step_tokens": max_prefill_step_tokens,
        "max_decode_step_tokens": max_decode_step_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "request_throughput_rps": num_requests / elapsed,
        "output_throughput_tok_s": output_tokens / elapsed,
        "total_token_throughput_tok_s": (prompt_tokens + output_tokens) / elapsed,
        "prefill_host_call_tok_s": prefill_tokens / prefill_seconds if prefill_seconds > 0 else 0.0,
        "decode_host_call_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else 0.0,
        "ttft_seconds": summarize(ttft),
        "tpot_seconds": summarize(tpot),
        "e2e_latency_seconds": summarize(e2e),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "cuda_free_gib_after": torch.cuda.mem_get_info()[0] / (1024**3),
    }
    if late_sequence is not None:
        result["long_prompt_interference"] = {
            "injected_after_decode_steps": late_after_decode_steps,
            "long_prompt_tokens": late_sequence.num_prompt_tokens,
            "long_ttft_seconds": first_token_at[late_sequence.seq_id]
            - arrival_at[late_sequence.seq_id],
            "short_token_gap_before_seconds": summarize(short_gap_before),
            "short_token_gap_after_seconds": summarize(short_gap_after),
        }
    return result


def print_summary(result: dict) -> None:
    print(
        f"Requests={result['requests']}, prompt={result['prompt_tokens']} tok, "
        f"output={result['output_tokens']} tok, elapsed={result['elapsed_seconds']:.3f}s"
    )
    print(
        f"Throughput: {result['request_throughput_rps']:.2f} req/s, "
        f"{result['output_throughput_tok_s']:.2f} output tok/s, "
        f"{result['total_token_throughput_tok_s']:.2f} total tok/s"
    )
    print(
        f"Forward host-call rate (async estimate): "
        f"prefill={result['prefill_host_call_tok_s']:.2f} tok/s, "
        f"decode={result['decode_host_call_tok_s']:.2f} tok/s"
    )
    for name, key in (
        ("TTFT", "ttft_seconds"),
        ("TPOT", "tpot_seconds"),
        ("E2E", "e2e_latency_seconds"),
    ):
        stats = result[key]
        print(
            f"{name}: mean={stats['mean'] * 1000:.2f} ms, "
            f"p50={stats['p50'] * 1000:.2f} ms, p99={stats['p99'] * 1000:.2f} ms"
        )
    print(
        f"GPU memory: allocated peak={result['peak_cuda_memory_gib']:.2f} GiB, "
        f"reserved peak={result['peak_cuda_reserved_gib']:.2f} GiB, "
        f"free after={result['cuda_free_gib_after']:.2f} GiB"
    )
    print(
        f"Largest scheduled step: prefill={result['max_prefill_step_tokens']} tok, "
        f"decode={result['max_decode_step_tokens']} tok"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Qwen3.5 hybrid eager serving benchmark")
    add_model_argument(parser)
    add_dtype_argument(parser)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--min-input-len", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--late-long-input-len", type=int, default=0)
    parser.add_argument("--late-after-decode-steps", type=int, default=4)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    add_result_argument(parser)
    args = parser.parse_args(argv)

    validate_positive(
        parser,
        args,
        "num_requests",
        "min_input_len",
        "output_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_model_len",
        "late_after_decode_steps",
    )
    if args.max_input_len < args.min_input_len:
        parser.error("--max-input-len must be >= --min-input-len")
    if args.max_input_len + args.output_len > args.max_model_len:
        parser.error("max-input-len + output-len must not exceed max-model-len")
    if args.late_long_input_len < 0:
        parser.error("--late-long-input-len must be non-negative")
    if args.late_long_input_len:
        if args.late_long_input_len <= args.max_input_len:
            parser.error("--late-long-input-len must exceed --max-input-len")
        if args.late_long_input_len + args.output_len > args.max_model_len:
            parser.error("late-long-input-len + output-len must not exceed max-model-len")
        if args.num_requests >= args.max_num_seqs:
            parser.error("leave one max-num-seqs slot for the late long request")
        if args.late_after_decode_steps >= args.output_len:
            parser.error("late-after-decode-steps must be less than output-len")
    if not args.model:
        parser.error("model path required: pass model or set NANOVLLM_MODEL")
    return normalize_paths(args, "model", "result_file")


def run(args):
    import torch

    from nanovllm import LLM

    llm = LLM(
        args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_prefix_cache_entries=0 if args.disable_prefix_cache else 16,
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
        late_sequence = (
            build_sequences(
                llm,
                1,
                args.late_long_input_len,
                args.late_long_input_len,
                args.output_len,
                args.seed + 1,
            )[0]
            if args.late_long_input_len
            else None
        )
        result = run_benchmark(
            llm,
            sequences,
            late_sequence=late_sequence,
            late_after_decode_steps=args.late_after_decode_steps,
        )
        cache = llm.model_runner.kv_cache
        result["kv_cache_blocks"] = llm.model_runner.num_kv_cache_blocks
        result["kv_cache_gib"] = cache.numel() * cache.element_size() / (1024**3)
    finally:
        llm.exit()

    result["environment"] = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    result["config"] = {
        "dtype": args.dtype,
        "num_requests": args.num_requests,
        "min_input_len": args.min_input_len,
        "max_input_len": args.max_input_len,
        "output_len": args.output_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": llm.max_model_len,
        "prefix_cache": not args.disable_prefix_cache,
        "seed": args.seed,
        "late_long_input_len": args.late_long_input_len,
        "late_after_decode_steps": (
            args.late_after_decode_steps if args.late_long_input_len else None
        ),
    }
    print_summary(result)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    return result


def main(argv=None):
    return execute("bench", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
