import argparse
import json

import torch
from torch.profiler import ProfilerActivity, profile

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


def _device_time_us(event) -> float:
    for name in (
        "device_time_total",
        "cuda_time_total",
        "self_device_time_total",
        "self_cuda_time_total",
    ):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _event_time_us(prof, key: str) -> float:
    for event in prof.key_averages():
        if event.key == key:
            return _device_time_us(event)
    return 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile Qwen3.5 hybrid decode and decide whether "
            "Full-Attention GQA/paged decode deserves a custom kernel."
        )
    )
    parser.add_argument("model")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--profile-steps", type=int, default=16)
    parser.add_argument(
        "--max-batched-tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8_e4m3"),
        default="auto",
    )
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=0.15,
        help=(
            "Minimum fraction of profiled decode GPU time spent in "
            "Full Attention before a custom GQA/paged kernel is justified."
        ),
    )
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--trace", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.context_len <= 0:
        raise ValueError("context-len must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.profile_steps <= 0:
        raise ValueError("profile-steps must be positive")
    if not 0.0 <= args.attention_threshold <= 1.0:
        raise ValueError("attention-threshold must be in [0, 1]")

    max_tokens = (
        args.warmup_steps
        + args.profile_steps
        + 8
    )
    engine = LLMEngine(
        args.model,
        max_num_seqs=args.batch_size,
        max_num_state_slots=args.batch_size,
        max_num_batched_tokens=args.max_batched_tokens,
        max_model_len=args.context_len + max_tokens + 1,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    prompt = [args.token_id] * args.context_len

    try:
        for _ in range(args.batch_size):
            engine.add_request(prompt, sampling)

        # Fill each request's hybrid history before measuring decode. Chunked
        # prefill is allowed here; only the profile window must be decode-only.
        while engine.scheduler.waiting:
            engine.step()

        for _ in range(args.warmup_steps):
            _, stats = engine.step()
            if stats.prefill_tokens:
                raise RuntimeError(
                    "warmup unexpectedly contained prefill work"
                )

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

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
                _, stats = engine.step()
                if stats.prefill_tokens:
                    raise RuntimeError(
                        "profile window unexpectedly contained prefill work"
                    )
            end_event.record()
            torch.cuda.synchronize()

        if args.trace:
            prof.export_chrome_trace(args.trace)

        decode_us = start_event.elapsed_time(end_event) * 1000.0
        attention_us = _event_time_us(
            prof,
            "nanovllm::full_attention_decode",
        )
        kv_store_us = _event_time_us(
            prof,
            "nanovllm::kv_cache_store",
        )
        attention_ratio = (
            attention_us / decode_us
            if decode_us > 0.0
            else 0.0
        )

        result = {
            "batch_size": args.batch_size,
            "context_len": args.context_len,
            "profile_steps": args.profile_steps,
            "kv_cache_dtype": args.kv_cache_dtype,
            "decode_gpu_ms": decode_us / 1000.0,
            "full_attention_decode_gpu_ms": (
                attention_us / 1000.0
            ),
            "kv_store_gpu_ms": kv_store_us / 1000.0,
            "full_attention_fraction": attention_ratio,
            "attention_threshold": args.attention_threshold,
            "custom_gqa_paged_kernel_candidate": (
                attention_ratio >= args.attention_threshold
            ),
        }
        print(json.dumps(result, indent=2))

        if attention_ratio >= args.attention_threshold:
            print(
                "Profile gate passed: benchmark a fused GQA/paged "
                "decode kernel against the current backend."
            )
        else:
            print(
                "Profile gate not passed: keep the current attention "
                "backend and optimize the larger decode hotspots first."
            )
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
