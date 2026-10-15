#!/usr/bin/env python3
"""Qwen3.5 Hybrid Runtime benchmark experiments for Hg-Spark/nano-vllm@qwen3.5.

Place this file in the repository root; requires scripts.bench and nanovllm.
Measures the engine directly (no HTTP, network time, or Poisson arrivals).
One process = one sample. Use the companion shell script for A/B repetitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from time import perf_counter


def snapshot_bytes(cache):
    return sum(
        t.numel() * t.element_size()
        for entry in cache.entries()
        for layer in entry.state_snapshot.layers
        for t in layer
    )



def parse_graph_sizes(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values = tuple(sorted({int(x.strip()) for x in raw.split(",")}))
    if not values or any(value < 1 for value in values):
        raise ValueError("graph sizes must be positive integers")
    return values


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", nargs="?", default=os.environ.get("NANOVLLM_MODEL"))
    p.add_argument("--dtype", required=True, choices=("float16", "bfloat16"))
    p.add_argument("--scenario", choices=("throughput", "prefix", "interference", "decode", "preemption"), required=True)
    p.add_argument("--num-requests", type=int, default=16, help="total initial requests (interference: short only)")
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--budget", type=int, default=512, help="max_num_batched_tokens")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--graph-sizes", default="", help="comma separated Graph buckets, e.g. 1,2,4; empty=Eager")
    p.add_argument("--cache-entries", type=int, default=0, help="0 disables prefix cache")
    p.add_argument("--shared-prefix-len", type=int, default=1024, help="prefix scenario")
    p.add_argument("--suffix-len", type=int, default=256, help="prefix scenario")
    p.add_argument("--target-requests", type=int, default=8, help="prefix: number of target requests")
    p.add_argument("--warmup-steps", type=int, default=8, help="fixed decode warmup steps")
    p.add_argument("--measure-steps", type=int, default=128, help="fixed decode measured steps")
    p.add_argument("--schedule-mode", choices=("mixed", "exclusive_decode_first"),
                   default="mixed", help="exclusive mode only changes prefill scheduling")
    p.add_argument("--require-preemptions", action="store_true",
                   help="fail if preemption scenario does not preempt")
    p.add_argument("--late-input-len", type=int, default=2048, help="interference scenario")
    p.add_argument("--late-after-decode-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="JSON result path; must not exist")
    a = p.parse_args()
    if not a.model:
        p.error("model path or NANOVLLM_MODEL must be provided")
    a.graph_sizes = parse_graph_sizes(a.graph_sizes)
    if a.input_len < 1 or a.max_model_len < 1:
        p.error("input-len and max-model-len must be positive")
    if a.max_num_seqs < 1 or a.budget < 1 or a.num_requests < 1 or a.output_len < 2:
        p.error("max-num-seqs, budget, num-requests must be positive and output-len >= 2")
    if a.cache_entries < 0 or not 0 < a.gpu_memory_utilization <= 1:
        p.error("cache-entries >= 0 and 0 < gpu-memory-utilization <= 1 required")
    if a.graph_sizes and max(a.graph_sizes) > a.max_num_seqs:
        p.error("all graph buckets must be <= max-num-seqs")
    if a.scenario in ("throughput", "preemption") and a.input_len + a.output_len > a.max_model_len:
        p.error("input-len + output-len exceeds max-model-len")
    if a.scenario == "prefix":
        if a.target_requests < 1:
            p.error("target-requests must be positive")
        if a.shared_prefix_len < 16 or a.shared_prefix_len % 16 or a.suffix_len < 1:
            p.error("prefix must be a multiple of 16 and >=16; suffix-len >= 1")
        if a.shared_prefix_len + a.suffix_len + a.output_len > a.max_model_len:
            p.error("prefix-len + suffix-len + output-len exceeds max-model-len")
    if a.scenario == "decode":
        if a.warmup_steps < 0 or a.measure_steps < 2:
            p.error("decode requires warmup-steps >= 0 and measure-steps >= 2")
        if a.input_len + a.warmup_steps + a.measure_steps + 2 > a.max_model_len:
            p.error("decode requires input-len + warmup-steps + measure-steps + 2 <= max-model-len")
    if a.schedule_mode != "mixed" and a.scenario != "interference":
        p.error("exclusive_decode_first is supported only for interference")
    if a.require_preemptions and a.scenario != "preemption":
        p.error("require-preemptions requires scenario=preemption")
    if a.scenario == "interference":
        if a.num_requests >= a.max_num_seqs:
            p.error("reserve one max-num-seqs slot for the injected long request")
        if a.late_input_len <= a.input_len or a.late_input_len + a.output_len > a.max_model_len:
            p.error("late-input-len must exceed input-len and fit in max-model-len")
        if not 0 < a.late_after_decode_steps < a.output_len:
            p.error("late-after-decode-steps must be between 1 and output-len-1")
    if Path(a.out).exists():
        p.error("output path already exists: refusing to overwrite experimental evidence")
    return a


def git_snapshot():
    def cmd(*argv):
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": cmd("git", "rev-parse", "HEAD"), "dirty": bool(cmd("git", "status", "--porcelain"))}


def tokens_hash(seqs):
    # Hash all complete token sequences in order without bloating result files.
    data = json.dumps([list(s.completion_token_ids) for s in seqs], separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def validate_preemptions(count: int, *, required: bool) -> None:
    if required and count == 0:
        raise RuntimeError("no preemptions observed; increase pressure or lower GPU budget")


def make_requests(engine, prompts, max_tokens):
    from nanovllm import SamplingParams
    from scripts._probes import prepare_requests
    return prepare_requests(
        engine, prompts,
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens),
    )


def run_fixed_decode(engine, a, graph_replays):
    """Measure real Scheduler + ModelRunner at a fixed active decode batch."""
    import torch

    from nanovllm import SamplingParams
    from scripts._probes import prepare_fixed_decode_batch, vocabulary_size
    from scripts.bench import summarize

    vocab = vocabulary_size(engine)
    prompt = [1 if vocab > 1 else 0] * a.input_len
    sampling = SamplingParams(
        temperature=0.0, ignore_eos=True,
        max_tokens=a.warmup_steps + a.measure_steps + 2,
    )
    prepare_fixed_decode_batch(engine, prompt, sampling, a.max_num_seqs)

    def step(phase):
        _, stats = engine.step()
        if stats.prefill_tokens or stats.decode_tokens != a.max_num_seqs:
            raise RuntimeError(
                f"{phase} batch drift: prefill={stats.prefill_tokens}, "
                f"decode={stats.decode_tokens}, expected={a.max_num_seqs}"
            )

    for _ in range(a.warmup_steps):
        step("warmup")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    replay_before = graph_replays()
    step_durations = []
    gpu_step_durations = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    t_begin = perf_counter()
    for _ in range(a.measure_steps):
        t0 = perf_counter()
        start_ev.record()
        step("measurement")
        end_ev.record()
        end_ev.synchronize()
        t1 = perf_counter()
        step_durations.append(t1 - t0)
        gpu_step_durations.append(start_ev.elapsed_time(end_ev) / 1000.0)
    duration = perf_counter() - t_begin
    replays = graph_replays() - replay_before
    if a.graph_sizes and replays != a.measure_steps:
        raise RuntimeError(f"expected {a.measure_steps} measured graph replays, got {replays}")
    if not a.graph_sizes and replays:
        raise RuntimeError("Eager unexpectedly used a graph")
    # CUDA event elapsed time includes host submission gaps, not pure kernel time.
    seqs = list(engine.scheduler.running)
    if len(seqs) != a.max_num_seqs:
        raise RuntimeError("fixed decode batch lost running requests")
    metrics = {
        "requests": a.max_num_seqs,
        "measured_decode_steps": a.measure_steps,
        "output_tokens": a.max_num_seqs * a.measure_steps,
        "elapsed_seconds": duration,
        "output_throughput_tok_s": a.max_num_seqs * a.measure_steps / duration,
        "step_latency_seconds": summarize(step_durations),
        "cuda_event_step_window_seconds": summarize(gpu_step_durations),
        "tpot_seconds": summarize(step_durations),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "cuda_free_gib_after": torch.cuda.mem_get_info()[0] / 1024**3,
        "graph_replays": replays,
    }
    return metrics, {"token_hash": tokens_hash(seqs),
                     "fixed_batch_size": a.max_num_seqs, "context_len": a.input_len}


def run(a):
    import torch

    from nanovllm import LLM
    from scripts._probes import vocabulary_size
    from scripts.bench import run_benchmark

    engine = LLM(
        a.model, dtype=a.dtype, max_num_batched_tokens=a.budget,
        max_num_seqs=a.max_num_seqs, max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_prefix_cache_entries=a.cache_entries,
        decode_graph_batch_sizes=a.graph_sizes,
    )
    try:
        gpu = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        vocab = vocabulary_size(engine)
        effective_max_len = engine.max_model_len
        kv_blocks = engine.model_runner.num_kv_cache_blocks
        kv = engine.model_runner.kv_cache
        kv_cache_gib = kv.numel() * kv.element_size() / (1024 ** 3)
        if vocab < 2:
            raise RuntimeError("vocabulary is unexpectedly small")
        rng = Random(a.seed)
        # Pre-run initialization, outside measured window. No cache match with target prompts.
        warmup = make_requests(engine, [[1] * 16], 2)
        run_benchmark(engine, warmup)

        graph_replays = 0
        first_prefill_start = {}
        preempted = []
        sched_prefill_tokens = 0
        original_graph = engine.model_runner._run_decode_graph
        original_schedule = engine.scheduler.schedule
        original_preempt = engine.scheduler.preempt
        original_prefill_pass = engine.scheduler._schedule_prefill_pass

        def tracked_preempt(seq):
            preempted.append({"id": seq.seq_id, "committed_tokens": seq.committed_tokens})
            return original_preempt(seq)

        engine.scheduler.preempt = tracked_preempt

        if a.schedule_mode == "exclusive_decode_first":
            # Same scheduler/resources. Skip Prefill while at least one Decode is scheduled.
            # This is Decode-first step exclusivity, NOT a Prefill-first baseline.
            def exclusive_prefill(prefill_chunks, decode_count, num_batched_tokens,
                                  *, reserved_free_blocks=0):
                if decode_count:
                    return num_batched_tokens
                return original_prefill_pass(
                    prefill_chunks, decode_count, num_batched_tokens,
                    reserved_free_blocks=reserved_free_blocks,
                )
            engine.scheduler._schedule_prefill_pass = exclusive_prefill

        def tracked_graph(chunks, seqs):
            nonlocal graph_replays
            value = original_graph(chunks, seqs)
            if value is not None:
                graph_replays += 1
            return value

        def tracked_schedule():
            nonlocal sched_prefill_tokens
            outcome = original_schedule()
            sched_prefill_tokens += outcome.prefill_tokens
            for chunk in outcome.prefill_chunks:
                first_prefill_start.setdefault(chunk.seq.seq_id, chunk.start)
            return outcome

        engine.model_runner._run_decode_graph = tracked_graph
        engine.scheduler.schedule = tracked_schedule

        if a.scenario == "decode":
            metrics, details = run_fixed_decode(engine, a, lambda: graph_replays)

        elif a.scenario in ("throughput", "preemption"):
            prompts = [[rng.randrange(vocab) for _ in range(a.input_len)]
                       for _ in range(a.num_requests)]
            seqs = make_requests(engine, prompts, a.output_len)
            metrics = run_benchmark(engine, seqs)
            completion_hash = tokens_hash(seqs)
            details = {"token_hash": completion_hash}
            if a.scenario == "preemption":
                validate_preemptions(len(preempted), required=a.require_preemptions)

        elif a.scenario == "interference":
            prompts = [[rng.randrange(vocab) for _ in range(a.input_len)]
                       for _ in range(a.num_requests)]
            short = make_requests(engine, prompts, a.output_len)
            late_prompt = [rng.randrange(vocab) for _ in range(a.late_input_len)]
            late = make_requests(engine, [late_prompt], a.output_len)[0]
            metrics = run_benchmark(
                engine, short, late_sequence=late,
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
            warm = make_requests(engine, [shared + warm_tail], 1)
            run_benchmark(engine, warm)  # Build exactly the same history in both A/B runs.
            cache = engine.scheduler.block_manager.prefix_cache
            entries_before = len(cache)
            bytes_before = snapshot_bytes(cache)
            target = make_requests(engine, [shared + tail for tail in target_tails], a.output_len)
            metrics = run_benchmark(engine, target)
            restored = [first_prefill_start.get(seq.seq_id, -1) for seq in target]
            if any(start < 0 for start in restored):
                raise RuntimeError("some target requests never executed Prefill")
            if a.cache_entries and not any(start > 0 for start in restored):
                raise RuntimeError("enabled cache never restored a shared prefix")
            if not a.cache_entries and any(start != 0 for start in restored):
                raise RuntimeError("disabled cache unexpectedly restored a prefix")
            details = {
                "warm_prompt_tokens": warm[0].num_prompt_tokens,
                "target_requests": len(target),
                "target_prompt_tokens": target[0].num_prompt_tokens,
                "cache_entries_before_target": entries_before,
                "snapshot_bytes_before_target": bytes_before,
                "snapshot_bytes_after_target": snapshot_bytes(cache),
                "restored_tokens_per_target": restored,
                "prefix_hit_rate": sum(start > 0 for start in restored) / len(restored),
                "total_restored_prefix_tokens": sum(max(start, 0) for start in restored),
                "token_hash": tokens_hash(target),
            }

        if a.graph_sizes and a.scenario == "throughput" and graph_replays == 0:
            raise RuntimeError("Graph buckets were captured, but no replay was observed")
        if a.scenario != "decode":
            metrics["graph_replays"] = graph_replays
        details["preemptions"] = len(preempted)
        details["preempted_committed_tokens"] = sum(x["committed_tokens"] for x in preempted)
        details["scheduled_prefill_tokens_including_setup"] = sched_prefill_tokens
        result = {
            "status": "completed",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "source": "nano-vllm qwen3.5 engine, no HTTP",
            "git": git_snapshot(),
            "gpu": gpu, "compute_capability": cc,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "effective_max_model_len": effective_max_len,
            "kv_cache_blocks": kv_blocks,
            "kv_cache_gib": kv_cache_gib,
            "parameters": {k: v for k, v in vars(a).items() if k != "out"},
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
    # Write a failure record too; never mask an incomplete sample as a valid result.
    try:
        result = run(a)
    except BaseException as exc:
        record = {
            "status": "failed", "scenario": a.scenario,
            "error_type": type(exc).__name__, "error": str(exc),
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_snapshot(),
            "parameters": {k: v for k, v in vars(a).items() if k != "out"},
        }
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        raise
    else:
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        m = result["metrics"]
        print(f"saved {target} | throughput={m['output_throughput_tok_s']:.1f} tok/s "
              f"TPOT={m['tpot_seconds']['mean'] * 1000:.2f} ms "
              f"graph_replays={m['graph_replays']}")
        if "total_restored_prefix_tokens" in result["details"]:
            print("restored prefix tokens:", result["details"]["total_restored_prefix_tokens"],
                  "hit rate:", result["details"]["prefix_hit_rate"])


if __name__ == "__main__":
    main()
