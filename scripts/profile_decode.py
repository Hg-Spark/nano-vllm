from __future__ import annotations

import argparse
import json

from scripts._common import (
    add_dtype_argument,
    add_model_argument,
    add_result_argument,
    cuda_memory_stats,
    execute,
    execute_fixed_decode_step,
    kv_cache_stats,
    normalize_paths,
    prepare_fixed_decode_batch,
    validate_positive,
)


def _device_time_us(event) -> float:
    return float(event.device_time_total)


def _profiled_hotspots(event_times: dict[str, float]) -> dict[str, float]:
    return {
        "full_attention": event_times.get("nanovllm::full_attention_decode", 0.0),
        "gdn": event_times.get("nanovllm::gdn_layer", 0.0),
        "ffn": event_times.get("nanovllm::ffn_layer", 0.0),
        "moe": event_times.get("nanovllm::moe_layer", 0.0),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=("Profile Qwen3.5 hybrid decode and report GPU time breakdown.")
    )
    add_model_argument(parser, required=True)
    add_dtype_argument(parser)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--profile-steps", type=int, default=16)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8_e4m3"),
        default="auto",
    )
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--trace", default=None)
    add_result_argument(parser)
    args = parser.parse_args(argv)
    validate_positive(
        parser,
        args,
        "context_len",
        "batch_size",
        "profile_steps",
    )
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.token_id < 0:
        parser.error("--token-id must be non-negative")
    if args.max_num_batched_tokens < args.batch_size:
        parser.error("fixed decode batch requires max-num-batched-tokens >= batch-size")
    return normalize_paths(args, "model", "trace", "result_file")


def run(args):
    import torch
    from torch.profiler import ProfilerActivity, profile

    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams

    max_tokens = args.warmup_steps + args.profile_steps + 8
    engine = LLMEngine(
        args.model,
        dtype=args.dtype,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.context_len + max_tokens + 1,
        kv_cache_dtype=args.kv_cache_dtype,
        max_prefix_cache_entries=0,
    )
    try:
        if args.token_id >= engine.config.text_config.vocab_size:
            raise ValueError("token-id must be smaller than the model vocabulary size")
        sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        prompt = [args.token_id] * args.context_len
        prepare_fixed_decode_batch(
            engine,
            prompt,
            sampling,
            args.batch_size,
        )

        for _ in range(args.warmup_steps):
            execute_fixed_decode_step(engine, args.batch_size, "warmup")

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            start_event.record()
            for _ in range(args.profile_steps):
                execute_fixed_decode_step(engine, args.batch_size, "profile window")
            end_event.record()
            torch.cuda.synchronize()

        if args.trace:
            prof.export_chrome_trace(args.trace)

        decode_us = start_event.elapsed_time(end_event) * 1000.0
        event_times = {event.key: _device_time_us(event) for event in prof.key_averages()}
        hotspot_times = _profiled_hotspots(event_times)
        attention_us = hotspot_times["full_attention"]
        kv_store_us = event_times.get("nanovllm::kv_cache_store", 0.0)
        gdn_us = hotspot_times["gdn"]
        ffn_us = hotspot_times["ffn"]
        moe_us = hotspot_times["moe"]
        attention_ratio = attention_us / decode_us if decode_us > 0.0 else 0.0
        gdn_ratio = gdn_us / decode_us if decode_us > 0.0 else 0.0
        ffn_ratio = ffn_us / decode_us if decode_us > 0.0 else 0.0
        moe_ratio = moe_us / decode_us if decode_us > 0.0 else 0.0
        dominant_hotspot = (
            max(hotspot_times, key=hotspot_times.get)
            if any(value > 0 for value in hotspot_times.values())
            else None
        )

        result = {
            "batch_size": args.batch_size,
            "context_len": args.context_len,
            "profile_steps": args.profile_steps,
            "kv_cache_dtype": args.kv_cache_dtype,
            "dtype": args.dtype,
            "decode_gpu_ms": decode_us / 1000.0,
            "full_attention_decode_gpu_ms": (attention_us / 1000.0),
            "kv_store_gpu_ms": kv_store_us / 1000.0,
            "gdn_gpu_ms": gdn_us / 1000.0,
            "ffn_gpu_ms": ffn_us / 1000.0,
            "moe_gpu_ms": moe_us / 1000.0,
            "full_attention_fraction": attention_ratio,
            "gdn_fraction": gdn_ratio,
            "ffn_fraction": ffn_ratio,
            "moe_fraction": moe_ratio,
            "dominant_profiled_hotspot": dominant_hotspot,
            "hotspot_evidence": "available" if dominant_hotspot is not None else "unavailable",
            "effective_max_model_len": engine.max_model_len,
            **kv_cache_stats(engine),
            **cuda_memory_stats(),
        }
        print(json.dumps(result, indent=2))
        return result
    finally:
        engine.exit()


def main(argv=None):
    return execute("decode_profile", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
