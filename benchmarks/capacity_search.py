#!/usr/bin/env python3
"""Search for the largest tested full Decode batch on a fixed workload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8_e4m3"), default="auto")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_batch <= args.budget:
        parser.error("require 1 <= max-batch <= budget")
    args.model = str(Path(args.model).expanduser().resolve())
    return args


def probe(args, batch: int) -> bool:
    output = args.out_dir / f"batch_{batch}.json"
    log = args.out_dir / f"batch_{batch}.log"
    command = [
        sys.executable,
        "-m",
        "benchmarks.bench_experiments",
        args.model,
        "--scenario",
        "throughput",
        "--dtype",
        args.dtype,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--num-requests",
        str(batch),
        "--max-num-seqs",
        str(batch),
        "--budget",
        str(args.budget),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--out",
        str(output.resolve()),
    ]
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        ).returncode
    if code:
        raise RuntimeError(f"batch {batch} failed; inspect {log}")
    record = json.loads(output.read_text(encoding="utf-8"))
    if record["execution_status"] != "completed":
        raise RuntimeError(f"batch {batch} incomplete; inspect {output}")
    metrics = record["result"]["metrics"]
    return metrics["requests"] == batch and metrics["max_decode_step_tokens"] == batch


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    low, high = 0, args.max_batch + 1
    trials = []
    while high - low > 1:
        batch = (low + high) // 2
        passed = probe(args, batch)
        trials.append({"batch_size": batch, "passed": passed})
        print(f"batch={batch}: {'pass' if passed else 'partial'}", flush=True)
        if passed:
            low = batch
        else:
            high = batch
    report = {
        "config": {**vars(args), "out_dir": str(args.out_dir)},
        "max_verified_batch": low,
        "search_upper_bound_reached": low == args.max_batch,
        "trials": trials,
    }
    destination = args.out_dir / "capacity_summary.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Full decode batch: {low} (searched through {args.max_batch}); {destination}")
    return 0 if low else 1


if __name__ == "__main__":
    raise SystemExit(main())
