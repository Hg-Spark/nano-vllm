#!/usr/bin/env python3
"""Qwen3.5 Hybrid Runtime benchmark experiments for Hg-Spark/nano-vllm@hybrid-runtime.

Uses scripts.bench and nanovllm from the repository root.
Measures the engine directly (no HTTP, network time, or Poisson arrivals).
One process = one sample. Use the companion shell script for A/B repetitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
from time import perf_counter

from scripts._common import (
    add_kv_cache_arguments,
    add_model_argument,
    cuda_memory_stats,
    execute,
    execute_fixed_decode_step,
    kv_cache_kwargs,
    kv_cache_stats,
    normalize_paths,
    prepare_fixed_decode_batch,
    validate_gpu_utilization,
    validate_positive,
)


def parse_graph_sizes(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values = tuple(sorted({int(x.strip()) for x in raw.split(",")}))
    if not values or any(value < 1 for value in values):
        raise ValueError("graph sizes must be positive integers")
    return values


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_model_argument(p)
    add_kv_cache_arguments(p)
    p.add_argument(
        "--scenario",
        choices=("throughput", "prefix", "interference", "decode"),
        required=True,
    )
    p.add_argument(
        "--num-requests",
        type=int,
        default=16,
        help="total initial requests (interference: short only)",
    )
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--budget", type=int, default=512, help="max_num_batched_tokens")
    p.add_argument(
        "--mixed-prefill-limit",
        type=int,
        default=128,
        help="prefill tokens per mixed step; -1 removes the mixed-load cap",
    )
    p.add_argument(
        "--activation-reserve-mib",
        type=int,
        default=256,
        help="transient activation headroom reserved before sizing static KV (MiB)",
    )
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument(
        "--graph-sizes", default="", help="comma separated Graph buckets, e.g. 1,2,4; empty=Eager"
    )
    p.add_argument(
        "--min-graph-coverage",
        type=float,
        default=0.0,
        help="fail a Graph throughput run if fewer than this fraction of decode forwards replay",
    )
    p.add_argument("--cache-entries", type=int, default=0, help="0 disables prefix cache")
    p.add_argument(
        "--shared-prefix-len",
        type=int,
        default=1024,
        help="prefix scenario; 0 measures cache overhead on unrelated prompts",
    )
    p.add_argument("--suffix-len", type=int, default=256, help="prefix scenario")
    p.add_argument(
        "--target-requests", type=int, default=8, help="prefix: number of target requests"
    )
    p.add_argument(
        "--expect-evictions",
        action="store_true",
        help="prefix: fail if the target workload does not evict any joint snapshot",
    )
    p.add_argument("--warmup-steps", type=int, default=8, help="fixed decode warmup steps")
    p.add_argument("--measure-steps", type=int, default=128, help="fixed decode measured steps")
    p.add_argument(
        "--schedule-mode",
        choices=("mixed", "exclusive_decode_first"),
        default="mixed",
        help="exclusive mode only changes prefill scheduling",
    )
    p.add_argument("--late-input-len", type=int, default=2048, help="interference scenario")
    p.add_argument("--late-after-decode-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="JSON result path; must not exist")
    a = normalize_paths(p.parse_args(argv), "model", "out")
    if not a.model:
        p.error("model path or NANOVLLM_MODEL must be provided")
    a.graph_sizes = parse_graph_sizes(a.graph_sizes)
    if not 0.0 <= a.min_graph_coverage <= 1.0:
        p.error("--min-graph-coverage must be between 0 and 1")
    if a.min_graph_coverage and (a.scenario != "throughput" or not a.graph_sizes):
        p.error("--min-graph-coverage requires --scenario throughput and --graph-sizes")
    validate_positive(p, a, "input_len", "num_requests", "max_num_seqs", "budget", "max_model_len")
    validate_gpu_utilization(p, a)
    if a.mixed_prefill_limit < -1:
        p.error("--mixed-prefill-limit must be -1 or non-negative")
    if a.activation_reserve_mib < 0:
        p.error("--activation-reserve-mib must be non-negative")
    if a.output_len < 2:
        p.error("output-len must be at least 2")
    if a.scenario == "throughput" and a.input_len + a.output_len - 1 > a.max_model_len:
        p.error("input-len + output-len exceeds max-model-len")
    if a.expect_evictions and (a.scenario != "prefix" or a.cache_entries < 1):
        p.error("--expect-evictions requires --scenario prefix and --cache-entries > 0")
    if a.scenario == "prefix":
        if a.target_requests < 1:
            p.error("target-requests must be positive")
        if a.shared_prefix_len < 0 or a.shared_prefix_len % 16 or a.suffix_len < 1:
            p.error("shared-prefix-len must be zero or a multiple of 16; suffix-len >= 1")
        if a.shared_prefix_len + a.suffix_len + a.output_len - 1 > a.max_model_len:
            p.error("prefix-len + suffix-len + output-len exceeds max-model-len")
    if a.scenario == "decode":
        if a.warmup_steps < 0 or a.measure_steps < 2:
            p.error("decode requires warmup-steps >= 0 and measure-steps >= 2")
        if a.input_len + a.warmup_steps + a.measure_steps + 2 > a.max_model_len:
            p.error("decode requires input-len + warmup-steps + measure-steps + 2 <= max-model-len")
    if a.schedule_mode != "mixed" and a.scenario != "interference":
        p.error("exclusive_decode_first is supported only for interference")
    if a.scenario == "interference":
        if a.num_requests >= a.max_num_seqs:
            p.error("reserve one max-num-seqs slot for the injected long request")
        if a.late_input_len <= a.input_len or a.late_input_len + a.output_len - 1 > a.max_model_len:
            p.error("late-input-len must exceed input-len and fit in max-model-len")
        if not 0 < a.late_after_decode_steps < a.output_len:
            p.error("late-after-decode-steps must be between 1 and output-len-1")
    if Path(a.out).exists():
        p.error("output path already exists: refusing to overwrite experimental evidence")
    return a


def tokens_hash(seqs):
    # Hash all complete token sequences in order without bloating result files.
    data = json.dumps([list(s.completion_token_ids) for s in seqs], separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def random_prompts(rng, vocab, count, length):
    return [[rng.randrange(vocab) for _ in range(length)] for _ in range(count)]


def make_requests(engine, prompts, max_tokens):
    from nanovllm import SamplingParams

    params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens)
    return [engine._prepare_request(prompt, params) for prompt in prompts]


def graph_counters(runner):
    return (
        sum(bucket.replay_count for bucket in runner.decode_graph_buckets.values()),
        runner.decode_graph_fallbacks.copy(),
    )


def measured_graph_usage(runner, before, decode_calls):
    replay_count, fallback_counts = graph_counters(runner)
    replays = replay_count - before[0]
    fallbacks = {name: count - before[1][name] for name, count in fallback_counts.items()}
    if replays < 0 or replays > decode_calls or sum(fallbacks.values()) != decode_calls - replays:
        raise RuntimeError("Decode Graph replay/fallback accounting is inconsistent")
    return {
        "graph_replays": replays,
        "decode_forward_calls": decode_calls,
        "graph_coverage": replays / decode_calls if decode_calls else 0.0,
        "eager_decode_forward_calls": decode_calls - replays,
        "graph_fallback_reasons": fallbacks,
    }


def require_throughput_graph_coverage(graph_metrics: dict, minimum: float) -> None:
    """Enforce measured graph coverage, excluding setup and warmup replays."""
    if graph_metrics["graph_replays"] and graph_metrics["graph_coverage"] >= minimum:
        return
    raise RuntimeError(
        f"Graph coverage {graph_metrics['graph_coverage']:.1%} below required {minimum:.1%}"
    )


def run_fixed_decode(engine, a):
    """Measure real Scheduler + ModelRunner at a fixed active decode batch."""
    import torch

    from nanovllm import SamplingParams
    from scripts.bench import summarize

    vocab = engine.config.text_config.vocab_size
    prompt = [1 if vocab > 1 else 0] * a.input_len
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=a.warmup_steps + a.measure_steps + 2,
    )
    prepare_fixed_decode_batch(engine, prompt, sampling, a.max_num_seqs)

    for _ in range(a.warmup_steps):
        execute_fixed_decode_step(engine, a.max_num_seqs, "warmup")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    graph_before = graph_counters(engine.model_runner)
    step_durations = []
    gpu_step_durations = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    t_begin = perf_counter()
    for _ in range(a.measure_steps):
        t0 = perf_counter()
        start_ev.record()
        execute_fixed_decode_step(engine, a.max_num_seqs, "measurement")
        end_ev.record()
        end_ev.synchronize()
        t1 = perf_counter()
        step_durations.append(t1 - t0)
        gpu_step_durations.append(start_ev.elapsed_time(end_ev) / 1000.0)
    duration = perf_counter() - t_begin
    graph_metrics = measured_graph_usage(engine.model_runner, graph_before, a.measure_steps)
    if a.graph_sizes and graph_metrics["graph_replays"] != a.measure_steps:
        raise RuntimeError(
            f"expected {a.measure_steps} measured graph replays, got {graph_metrics['graph_replays']}"
        )
    if not a.graph_sizes and graph_metrics["graph_replays"]:
        raise RuntimeError("Eager unexpectedly used a graph")
    # CUDA event elapsed time includes host submission gaps, not pure kernel time.
    seqs = list(engine.scheduler.running)
    if len(seqs) != a.max_num_seqs:
        raise RuntimeError("fixed decode batch lost running requests")
    step_latency = summarize(step_durations)
    metrics = {
        "requests": a.max_num_seqs,
        "measured_decode_steps": a.measure_steps,
        "output_tokens": a.max_num_seqs * a.measure_steps,
        "elapsed_seconds": duration,
        "output_throughput_tok_s": a.max_num_seqs * a.measure_steps / duration,
        "step_latency_seconds": step_latency,
        "cuda_event_step_window_seconds": summarize(gpu_step_durations),
        "tpot_seconds": step_latency,
        **cuda_memory_stats(),
        **graph_metrics,
    }
    return metrics, {
        "token_hash": tokens_hash(seqs),
        "fixed_batch_size": a.max_num_seqs,
        "context_len": a.input_len,
    }


def enable_exclusive_decode_first(scheduler):
    """Suppress Prefill in a step that already schedules Decode."""
    original_prefill_pass = scheduler._schedule_prefill_pass

    def exclusive_prefill(prefill_chunks, decode_count):
        if not decode_count:
            original_prefill_pass(prefill_chunks, decode_count)

    scheduler._schedule_prefill_pass = exclusive_prefill


def run(a):
    import torch

    from nanovllm import LLM
    from scripts.bench import run_benchmark

    engine = LLM(
        a.model,
        max_num_batched_tokens=a.budget,
        max_num_seqs=a.max_num_seqs,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        activation_reserve_bytes=a.activation_reserve_mib * 1024**2,
        max_mixed_prefill_tokens=(None if a.mixed_prefill_limit == -1 else a.mixed_prefill_limit),
        **kv_cache_kwargs(a),
        max_prefix_cache_entries=a.cache_entries,
        decode_graph_batch_sizes=a.graph_sizes,
    )
    try:
        gpu = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        vocab = engine.config.text_config.vocab_size
        effective_max_len = engine.max_model_len
        kv_stats = kv_cache_stats(engine)
        if vocab < 2:
            raise RuntimeError("vocabulary is unexpectedly small")
        rng = Random(a.seed)
        # Warm both Prefill and one Decode Forward without reaching a KV-page
        # boundary or contaminating the prefix-cache measurement.
        warmup = make_requests(engine, [[1] * 14], 2)
        run_benchmark(engine, warmup)

        if a.schedule_mode == "exclusive_decode_first":
            enable_exclusive_decode_first(engine.scheduler)

        track_schedule = a.scenario == "prefix" or a.cache_entries > 0
        first_prefill_start = {}
        sched_prefill_tokens = 0
        if track_schedule:
            original_schedule = engine.scheduler.schedule

            def tracked_schedule():
                nonlocal sched_prefill_tokens
                outcome = original_schedule()
                sched_prefill_tokens += outcome.prefill_tokens
                if a.scenario == "prefix":
                    for chunk in outcome.prefill_chunks:
                        first_prefill_start.setdefault(chunk.seq.seq_id, chunk.start)
                return outcome

            engine.scheduler.schedule = tracked_schedule
        graph_before = graph_counters(engine.model_runner)

        if a.scenario == "decode":
            metrics, details = run_fixed_decode(engine, a)

        elif a.scenario == "throughput":
            prompts = random_prompts(rng, vocab, a.num_requests, a.input_len)
            seqs = make_requests(engine, prompts, a.output_len)
            metrics = run_benchmark(engine, seqs)
            completion_hash = tokens_hash(seqs)
            details = {"token_hash": completion_hash}

        elif a.scenario == "interference":
            prompts = random_prompts(rng, vocab, a.num_requests, a.input_len)
            short = make_requests(engine, prompts, a.output_len)
            late_prompt = random_prompts(rng, vocab, 1, a.late_input_len)[0]
            late = make_requests(engine, [late_prompt], a.output_len)[0]
            metrics = run_benchmark(
                engine,
                short,
                late_sequence=late,
                late_after_decode_steps=a.late_after_decode_steps,
            )
            details = {"token_hash": tokens_hash([*short, late])}

        else:
            shared = [rng.randrange(vocab) for _ in range(a.shared_prefix_len)]
            warm_tail = [rng.randrange(vocab) for _ in range(a.suffix_len)]
            target_tails = [
                [rng.randrange(vocab) for _ in range(a.suffix_len)]
                for _ in range(a.target_requests)
            ]
            for idx, tail in enumerate(target_tails):
                if tail == warm_tail:
                    tail[-1] = (tail[-1] + idx + 1) % vocab
            # Use a one-token sentinel so the immutable prompt checkpoint
            # lies exactly at the shared-prefix boundary.
            warm = make_requests(
                engine,
                [shared + warm_tail[:1] if shared else warm_tail[:1]],
                1,
            )
            run_benchmark(engine, warm)  # Build exactly the same history in both A/B runs.
            graph_before = graph_counters(engine.model_runner)
            cache = engine.scheduler.cache_manager.prefix_cache
            entries_before = len(cache)
            bytes_before = cache.snapshot_bytes
            published_before = cache.published_count
            evicted_before = cache.evicted_count
            target = make_requests(engine, [shared + tail for tail in target_tails], a.output_len)
            metrics = run_benchmark(engine, target)
            restored = [first_prefill_start.get(seq.seq_id, -1) for seq in target]
            if any(start < 0 for start in restored):
                raise RuntimeError("some target requests never executed Prefill")
            if a.cache_entries and a.shared_prefix_len and not any(start > 0 for start in restored):
                raise RuntimeError("enabled cache never restored a shared prefix")
            if (not a.cache_entries or not a.shared_prefix_len) and any(
                start != 0 for start in restored
            ):
                raise RuntimeError(
                    "unexpected prefix restore in cache-disabled or unshared workload"
                )
            if a.expect_evictions and cache.evicted_count == evicted_before:
                raise RuntimeError("eviction workload did not evict a joint snapshot")
            details = {
                "warm_prompt_tokens": warm[0].num_prompt_tokens,
                "target_requests": len(target),
                "target_prompt_tokens": target[0].num_prompt_tokens,
                "cache_entries_before_target": entries_before,
                "snapshot_bytes_before_target": bytes_before,
                "snapshot_bytes_after_target": cache.snapshot_bytes,
                "peak_snapshot_bytes_including_setup": cache.peak_snapshot_bytes,
                "peak_staging_snapshot_bytes_including_setup": cache.peak_staging_snapshot_bytes,
                "peak_total_snapshot_bytes_including_setup": cache.peak_total_snapshot_bytes,
                "staging_snapshot_bytes_after_target": cache.staging_snapshot_bytes,
                "published_snapshots_during_target": cache.published_count - published_before,
                "evicted_snapshots_during_target": cache.evicted_count - evicted_before,
                "retained_snapshots_after_target": len(cache),
                "restored_tokens_per_target": restored,
                "prefix_hit_rate": sum(start > 0 for start in restored) / len(restored),
                "total_restored_prefix_tokens": sum(max(start, 0) for start in restored),
                "mean_restored_prefix_tokens": sum(max(start, 0) for start in restored)
                / len(restored),
                "restored_prompt_token_ratio": sum(max(start, 0) for start in restored)
                / sum(seq.num_prompt_tokens for seq in target),
                "shared_prefix_fraction": a.shared_prefix_len
                / (a.shared_prefix_len + a.suffix_len),
                "token_hash": tokens_hash(target),
            }

        if a.scenario != "decode":
            graph_metrics = measured_graph_usage(
                engine.model_runner, graph_before, metrics["decode_forward_calls"]
            )
            metrics.update(graph_metrics)
            if a.graph_sizes and a.scenario == "throughput":
                require_throughput_graph_coverage(graph_metrics, a.min_graph_coverage)
        if track_schedule:
            details["scheduled_prefill_tokens_including_setup"] = sched_prefill_tokens
        result = {
            "source": "nano-vllm qwen3.5 engine, no HTTP",
            "gpu": gpu,
            "compute_capability": cc,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "effective_max_model_len": effective_max_len,
            **kv_stats,
            "metrics": metrics,
            "details": details,
        }
        return result
    finally:
        engine.exit()


def main():
    a = arguments()
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    a.result_file = a.out
    completed = None

    def run_and_capture(args):
        nonlocal completed
        completed = run(args)
        return completed

    code = execute(f"benchmark_{a.scenario}", a, run_and_capture)
    if code:
        return code

    result = completed
    metrics = result["metrics"]
    print(
        f"saved {target} | throughput={metrics['output_throughput_tok_s']:.1f} tok/s "
        f"TPOT={metrics['tpot_seconds']['mean'] * 1000:.2f} ms "
        f"graph_replays={metrics['graph_replays']}"
    )
    if "total_restored_prefix_tokens" in result["details"]:
        print(
            "restored prefix tokens:",
            result["details"]["total_restored_prefix_tokens"],
            "hit rate:",
            result["details"]["prefix_hit_rate"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
