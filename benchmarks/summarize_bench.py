#!/usr/bin/env python3
"""Compare NanoVLLM repeated experiment JSONs with output-parity checks."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def workload_key(result):
    p = result["parameters"]
    # A/B toggles deliberately excluded: graph_sizes, cache_entries, schedule_mode.
    names = ("model", "dtype", "scenario", "seed", "budget", "max_num_seqs",
             "num_requests", "input_len", "output_len", "max_model_len",
             "gpu_memory_utilization", "shared_prefix_len", "suffix_len",
             "target_requests", "late_input_len", "late_after_decode_steps",
             "warmup_steps", "measure_steps")
    return (result["gpu"], tuple(result.get("compute_capability") or ()),
            result["git"]["commit"], result["git"]["dirty"],
            result.get("script_sha256"),
            *[p.get(n) for n in names])


def variant(result):
    p = result["parameters"]
    return (tuple(p["graph_sizes"]), p["cache_entries"], p["schedule_mode"])


def number(values, field):
    return median(x["metrics"][field] for x in values)


def milliseconds(values, field, percentile="mean"):
    items = [x["metrics"].get(field) for x in values]
    if any(x is None for x in items):
        return None
    return median(x[percentile] * 1000 for x in items)


def percent(base, changed, higher_better=True):
    if base is None or changed is None or base <= 0:
        return None
    return 100.0 * ((changed / base - 1.0) if higher_better else (1.0 - changed / base))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", default="results")
    ap.add_argument("--min-repeats", type=int, default=1)
    args = ap.parse_args(argv)

    samples = []
    failed = []
    for path in sorted(Path(args.folder).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf8"))
        if r.get("status") != "completed":
            failed.append(path.name)
        else:
            samples.append((path.name, r))
    if not samples:
        raise SystemExit("No completed benchmark JSON in folder")

    by_workload = defaultdict(list)
    grouped = defaultdict(list)
    for filename, r in samples:
        w, v = workload_key(r), variant(r)
        by_workload[w].append((filename, r["details"]["token_hash"]))
        grouped[(w, v)].append(r)

    mismatches = [(str(k), v) for k, v in by_workload.items()
                  if len({token_hash for _, token_hash in v}) > 1]
    if mismatches:
        for key, values in mismatches:
            print("PARITY FAILURE", key, values)
        raise SystemExit("Output token hashes differ for a supposedly identical workload")

    print("scenario            batch budget graph       cache mode                    runs tok/s  TPOT_ms TTFT_ms  hits replays preempts")
    for (w, v), runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        p = runs[0]["parameters"]
        hit = median(r["details"].get("prefix_hit_rate", 0) for r in runs)
        replay = median(r["metrics"].get("graph_replays", 0) for r in runs)
        preempts = median(r["details"].get("preemptions", 0) for r in runs)
        ttft = milliseconds(runs, "ttft_seconds")
        print(f"{p['scenario']:19} {p['max_num_seqs']:5} {p['budget']:6} "
              f"{','.join(map(str,v[0])) or '-':11} {v[1]:5} {v[2]:23} "
              f"{len(runs):4} {number(runs,'output_throughput_tok_s'):6.1f} "
              f"{milliseconds(runs,'tpot_seconds'):8.2f} "
              f"{ttft if ttft is not None else float('nan'):7.2f} "
              f"{hit:5.2f} {replay:7.0f} {preempts:8.0f}")

    issues = []
    compared = 0
    for w in by_workload:
        variants = {v: grouped[(w, v)] for ww, v in grouped if ww == w}
        sample = next(iter(variants.values()))[0]
        p = sample["parameters"]
        scenario = p["scenario"]
        if scenario in ("throughput", "decode"):
            baseline = variants.get(((), 0, "mixed"))
            matches = [(v, rs) for v, rs in variants.items() if v[0] and v[1] == 0]
            label = "Graph / Eager"
            latency_key = "tpot_seconds"
        elif scenario == "prefix":
            baseline = variants.get(((), 0, "mixed"))
            matches = [(v, rs) for v, rs in variants.items() if v[1] > 0 and not v[0]]
            label = "Prefix On / Off"
            latency_key = "ttft_seconds"
        elif scenario == "interference":
            baseline = variants.get(((), 0, "exclusive_decode_first"))
            matches = [(v, rs) for v, rs in variants.items() if v == ((), 0, "mixed")]
            label = "Mixed / Exclusive decode-first"
            latency_key = None
        else:
            continue
        if not baseline or not matches:
            issues.append(f"unpaired A/B: {scenario} batch={p['max_num_seqs']} budget={p['budget']}")
            continue
        for v, treatment in matches:
            if min(len(baseline), len(treatment)) < args.min_repeats:
                issues.append(f"too few repeats for {label}, {scenario} batch={p['max_num_seqs']}")
                continue
            throughput = percent(number(baseline,"output_throughput_tok_s"),
                                 number(treatment,"output_throughput_tok_s"))
            print(f"\n{label}: {scenario}, batch={p['max_num_seqs']}, budget={p['budget']}, variant={v}")
            print(f"  Output throughput improvement: {throughput:+.2f}%")
            if latency_key:
                improvement = percent(milliseconds(baseline,latency_key),
                                      milliseconds(treatment,latency_key), higher_better=False)
                print(f"  {latency_key} reduction: {improvement:+.2f}%")
            if scenario == "interference":
                def long_ttft(rs):
                    return median(r["metrics"]["long_prompt_interference"]["long_ttft_seconds"] * 1000
                                  for r in rs)
                b, t = long_ttft(baseline), long_ttft(treatment)
                print(f"  Late long request TTFT: {b:.2f} -> {t:.2f} ms "
                      f"(reduction {percent(b,t,higher_better=False):+.2f}%)")
                # Small synthetic samples: mean only; do not label tail latency as production P99.
                for label2, runs in (("exclusive", baseline),("mixed", treatment)):
                    gap = median(r["metrics"]["long_prompt_interference"]
                                 ["short_token_gap_after_seconds"]["mean"] * 1000 for r in runs)
                    p99 = median(r["metrics"]["long_prompt_interference"]
                                 ["short_token_gap_after_seconds"]["p99"] * 1000 for r in runs)
                    print(f"  Post-arrival observed token gap ({label2}): "
                          f"mean={gap:.2f} ms, p99={p99:.2f} ms (synthetic sample)")
            compared += 1
    print(f"\nSuccessful paired comparisons: {compared}")
    if failed:
        issues.append("failed/incomplete experiments: " + ", ".join(failed))
    if issues:
        for issue in issues:
            print("INCOMPLETE:", issue)
        raise SystemExit("Some runs could not be compared. No unsupported speedup should be reported.")


if __name__ == "__main__":
    main()
