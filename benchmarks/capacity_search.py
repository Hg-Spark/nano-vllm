#!/usr/bin/env python3
"""Search for the largest tested full Decode batch on a fixed workload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts._common import add_kv_cache_arguments

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    add_kv_cache_arguments(parser)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_len < 4:
        parser.error("output-len must be at least 4 for fixed decode measurement")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input-len + output-len exceeds max-model-len")
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
        "decode",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--kv-cache-k-scale",
        str(args.kv_cache_k_scale),
        "--kv-cache-v-scale",
        str(args.kv_cache_v_scale),
        "--warmup-steps",
        "0",
        "--measure-steps",
        str(args.output_len - 2),
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
        failure = log.read_text(encoding="utf-8", errors="replace").lower()
        if any(
            reason in failure
            for reason in (
                "out of memory",
                "insufficient gpu memory",
                "physical kv cache capacity",
                "insufficient kv cache blocks",
                "fixed decode batch exceeds kv capacity",
            )
        ):
            return False
        raise RuntimeError(f"batch {batch} failed; inspect {log}")
    record = json.loads(output.read_text(encoding="utf-8"))
    if record["execution_status"] != "completed":
        raise RuntimeError(f"batch {batch} incomplete; inspect {output}")
    return True


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    low, high = 0, args.max_batch + 1
    trials = []
    while high - low > 1:
        batch = (low + high) // 2
        passed = probe(args, batch)
        trials.append({"batch_size": batch, "passed": passed})
        print(f"batch={batch}: {'pass' if passed else 'capacity-exceeded'}", flush=True)
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
