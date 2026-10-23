from __future__ import annotations

import time
from statistics import quantiles
from typing import TYPE_CHECKING

from scripts._common import cuda_memory_stats

P99_MIN_SAMPLES = 1000

if TYPE_CHECKING:
    from nanovllm import LLM
    from nanovllm.engine.sequence import Sequence


def summarize(values: list[float], *, p99_min_samples: int = 1) -> dict[str, float | None]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p99": None if p99_min_samples > 1 else 0.0}
    pct = quantiles(values, n=100, method="inclusive") if len(values) > 1 else [values[0]] * 99
    return {
        "mean": sum(values) / len(values),
        "p50": pct[49],
        "p99": pct[98] if len(values) >= p99_min_samples else None,
    }


def run_benchmark(
    llm: LLM,
    sequences: list[Sequence],
    *,
    late_sequence: Sequence | None = None,
    late_after_decode_steps: int = 4,
) -> dict:
    import torch

    for seq in sequences:
        llm.scheduler.add(seq)
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
    token_gaps: list[float] = []
    previous_token_at: dict[int, float] = {}
    previous_token_count: dict[int, int] = {}
    first_token_at: dict[int, float] = {}
    finished_at: dict[int, float] = {}
    prefill_tokens = decode_tokens = max_prefill_step_tokens = max_decode_step_tokens = steps = 0
    prefill_seconds = decode_seconds = snapshot_capture_seconds = 0.0

    def record_request_times(now: float) -> None:
        for seq in all_sequences:
            count = seq.num_completion_tokens
            if count > previous_token_count.get(seq.seq_id, 0):
                previous = previous_token_at.get(seq.seq_id)
                if previous is not None:
                    token_gaps.append(now - previous)
                    if seq.seq_id in short_ids:
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
        snapshot_capture_seconds += getattr(stats, "snapshot_capture_seconds", 0.0)
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
        "snapshot_capture_seconds": snapshot_capture_seconds,
        "ttft_seconds": summarize(ttft, p99_min_samples=P99_MIN_SAMPLES),
        "ttft_samples": len(ttft),
        "tpot_seconds": summarize(tpot, p99_min_samples=P99_MIN_SAMPLES),
        "tpot_samples": len(tpot),
        "itl_seconds": summarize(token_gaps, p99_min_samples=P99_MIN_SAMPLES),
        "itl_samples": len(token_gaps),
        "decode_forward_calls": decode_steps,
        "e2e_latency_seconds": summarize(e2e, p99_min_samples=P99_MIN_SAMPLES),
        **cuda_memory_stats(),
    }
    if late_sequence is not None:
        result["long_prompt_interference"] = {
            "injected_after_decode_steps": late_after_decode_steps,
            "long_prompt_tokens": late_sequence.num_prompt_tokens,
            "long_ttft_seconds": first_token_at[late_sequence.seq_id]
            - arrival_at[late_sequence.seq_id],
            "short_token_gap_before_seconds": summarize(
                short_gap_before, p99_min_samples=P99_MIN_SAMPLES
            ),
            "short_token_gap_before_samples": len(short_gap_before),
            "short_token_gap_after_seconds": summarize(
                short_gap_after, p99_min_samples=P99_MIN_SAMPLES
            ),
            "short_token_gap_after_samples": len(short_gap_after),
        }
    return result
