#!/usr/bin/env python3
"""Compare NanoVLLM repeated experiment JSONs with output-parity checks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from scripts.bench import P99_MIN_SAMPLES


def workload_key(result, *, ignore_mixed_prefill_limit=False):
    p = result["config"]
    # A/B toggles deliberately excluded: graph_sizes, cache_entries, schedule_mode.
    names = (
        "model",
        "dtype",
        "kv_cache_dtype",
        "kv_cache_k_scale",
        "kv_cache_v_scale",
        "scenario",
        "seed",
        "budget",
        "mixed_prefill_limit",
        "activation_reserve_mib",
        "max_num_seqs",
        "num_requests",
        "input_len",
        "output_len",
        "max_model_len",
        "gpu_memory_utilization",
        "shared_prefix_len",
        "suffix_len",
        "target_requests",
        "conversation_turns",
        "late_input_len",
        "late_after_decode_steps",
        "warmup_steps",
        "measure_steps",
    )
    return (
        result["gpu"],
        tuple(result["compute_capability"]),
        result.get("torch_version"),
        result.get("torch_cuda_version"),
        *[
            None if ignore_mixed_prefill_limit and n == "mixed_prefill_limit" else p.get(n)
            for n in names
        ],
    )


def variant(result):
    p = result["config"]
    return (tuple(p["graph_sizes"]), p["cache_entries"], p["schedule_mode"])


def number(values, field):
    return median(x["metrics"][field] for x in values)


def milliseconds(values, field, percentile="mean"):
    items = [x["metrics"].get(field) for x in values]
    if any(x is None for x in items):
        return None
    return median(x[percentile] * 1000 for x in items)


def format_percent(value):
    return f"{value:+.2f}%" if value is not None else "n/a"


def format_ms(value, width=7):
    return f"{value:{width}.2f}" if value is not None else f"{'n/a':>{width}}"


def percent(base, changed, higher_better=True):
    if base is None or changed is None or base <= 0:
        return None
    return 100.0 * ((changed / base - 1.0) if higher_better else (1.0 - changed / base))


def describe_first_output_difference(left, right):
    """Locate the first different generated token using optional benchmark diagnostics."""
    if left is None or right is None:
        return "rerun with --record-output-tokens to locate the first mismatch"
    for request_index, (a, b) in enumerate(zip(left, right)):
        for token_index, (token_a, token_b) in enumerate(zip(a, b)):
            if token_a != token_b:
                return (
                    f"request={request_index}, token={token_index}: "
                    f"reference={token_a}, changed={token_b}"
                )
        if len(a) != len(b):
            return f"request={request_index}: output length {len(a)} vs {len(b)}"
    if len(left) != len(right):
        return f"request count {len(left)} vs {len(right)}"
    return "recorded completions match; inspect prompt history or hash inputs"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="directory of benchmark experiment JSON files")
    ap.add_argument("--min-repeats", type=int, default=5)
    ap.add_argument("--min-itl-samples", type=int, default=P99_MIN_SAMPLES)
    args = ap.parse_args(argv)
    if args.min_repeats < 1:
        ap.error("--min-repeats must be positive")
    if args.min_itl_samples < P99_MIN_SAMPLES:
        ap.error(f"--min-itl-samples must be at least {P99_MIN_SAMPLES} for P99 reporting")

    samples = []
    failed = []
    for path in sorted(Path(args.folder).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf8"))
        if not str(r.get("experiment", "")).startswith("benchmark_"):
            continue
        if r.get("execution_status") != "completed":
            failed.append(path.name)
            continue
        result = r["result"]
        if r["config"]["scenario"] == "prefix":
            required = (
                "peak_snapshot_bytes_including_setup",
                "peak_staging_snapshot_bytes_including_setup",
                "peak_total_snapshot_bytes_including_setup",
            )
            missing = [key for key in required if key not in result["details"]]
            if missing:
                raise SystemExit(
                    f"{path.name}: missing required prefix snapshot metrics {missing}; "
                    "rerun with the current benchmark schema"
                )
        samples.append((path.name, {**result, "config": r["config"]}))
    if not samples:
        raise SystemExit(
            "No comparable completed benchmark JSON in folder"
            + (": " + ", ".join(failed) if failed else "")
        )

    by_workload = defaultdict(list)
    grouped = defaultdict(list)
    for filename, r in samples:
        w, v = workload_key(r), variant(r)
        by_workload[w].append((filename, r))
        grouped[(w, v)].append(r)

    mismatches = [
        (str(k), v) for k, v in by_workload.items()
        if len({r["details"]["token_hash"] for _, r in v}) > 1
    ]
    if mismatches:
        for key, values in mismatches:
            reference_name, reference = values[0]
            changed_name, changed = next(
                (name, r) for name, r in values
                if r["details"]["token_hash"] != reference["details"]["token_hash"]
            )
            print("PARITY FAILURE", key, [(name, r["details"]["token_hash"]) for name, r in values])
            print(
                f"  First mismatch {reference_name} vs {changed_name}: "
                + describe_first_output_difference(
                    reference["details"].get("output_token_sequences"),
                    changed["details"].get("output_token_sequences"),
                )
            )
        raise SystemExit("Output token hashes differ for a supposedly identical workload")

    # An exclusive Decode-first run never uses mixed_prefill_limit: share the
    # budget-matched baseline without re-running it for every Mixed cap.
    exclusive_baselines = {}
    for (_, v), runs in grouped.items():
        if (
            v == ((), 0, "exclusive_decode_first")
            and runs[0]["config"]["scenario"] == "interference"
        ):
            key = workload_key(runs[0], ignore_mixed_prefill_limit=True)
            cap = runs[0]["config"].get("mixed_prefill_limit")
            if key not in exclusive_baselines or cap == -1:
                exclusive_baselines[key] = runs

    print(
        "scenario            batch budget cap  graph       cache mode                    runs tok/s  TPOT_ms TTFT_ms  hits replays"
    )
    for (w, v), runs in sorted(
        grouped.items(),
        key=lambda item: (
            item[1][0]["config"]["scenario"],
            item[1][0]["config"]["max_num_seqs"],
            item[1][0]["config"]["budget"],
            str(item[0]),
        ),
    ):
        p = runs[0]["config"]
        hit = median(r["details"].get("prefix_hit_rate", 0) for r in runs)
        replay = median(r["metrics"].get("graph_replays", 0) for r in runs)
        ttft = milliseconds(runs, "ttft_seconds")
        print(
            f"{p['scenario']:19} {p['max_num_seqs']:5} {p['budget']:6} "
            f"{str(p.get('mixed_prefill_limit', '-')):4} "
            f"{','.join(map(str, v[0])) or '-':11} {v[1]:5} {v[2]:23} "
            f"{len(runs):4} {number(runs, 'output_throughput_tok_s'):6.1f} "
            f"{format_ms(milliseconds(runs, 'tpot_seconds'), 8)} "
            f"{format_ms(ttft)} "
            f"{hit:5.2f} {replay:7.0f}"
        )

    issues = []
    compared = 0
    for w in by_workload:
        variants = {v: grouped[(w, v)] for ww, v in grouped if ww == w}
        sample = next(iter(variants.values()))[0]
        p = sample["config"]
        scenario = p["scenario"]
        if scenario in ("throughput", "decode"):
            baseline = variants.get(((), 0, "mixed"))
            matches = [(v, rs) for v, rs in variants.items() if v[0] and v[1] == 0]
            label = "Graph / Eager"
            latency_key = "tpot_seconds"
        elif scenario in ("prefix", "conversation"):
            baseline = variants.get(((), 0, "mixed"))
            matches = [(v, rs) for v, rs in variants.items() if v[1] > 0 and not v[0]]
            label = (
                "Conversation Cache On / Off" if scenario == "conversation" else "Prefix On / Off"
            )
            latency_key = "ttft_seconds"
        elif scenario == "interference":
            baseline = variants.get(((), 0, "exclusive_decode_first"))
            if baseline is None:
                baseline = exclusive_baselines.get(
                    workload_key(sample, ignore_mixed_prefill_limit=True)
                )
            matches = [(v, rs) for v, rs in variants.items() if v == ((), 0, "mixed")]
            label = "Mixed / Exclusive decode-first"
            latency_key = None
        else:
            continue
        if not baseline or not matches:
            issues.append(
                f"unpaired A/B: {scenario} batch={p['max_num_seqs']} budget={p['budget']}"
            )
            continue
        for v, treatment in matches:
            # The shared exclusive baseline can be from a different cap group.
            # Preserve strict token-output parity across that reuse boundary.
            if {r["details"]["token_hash"] for r in baseline} != {
                r["details"]["token_hash"] for r in treatment
            }:
                raise SystemExit(
                    "PARITY FAILURE: shared exclusive baseline differs from Mixed output; "
                    + describe_first_output_difference(
                        baseline[0]["details"].get("output_token_sequences"),
                        treatment[0]["details"].get("output_token_sequences"),
                    )
                )
            if min(len(baseline), len(treatment)) < args.min_repeats:
                issues.append(f"too few repeats for {label}, {scenario} batch={p['max_num_seqs']}")
                continue
            throughput = percent(
                number(baseline, "output_throughput_tok_s"),
                number(treatment, "output_throughput_tok_s"),
            )
            print(
                f"\n{label}: {scenario}, batch={p['max_num_seqs']}, budget={p['budget']}, "
                f"cap={p.get('mixed_prefill_limit', '-')}, variant={v}"
            )
            print(f"  Output throughput improvement: {format_percent(throughput)}")
            if scenario in ("throughput", "decode") and all(
                "graph_coverage" in r["metrics"] for r in treatment
            ):
                graph_coverage = median(r["metrics"]["graph_coverage"] for r in treatment)
                eager_calls = median(r["metrics"]["eager_decode_forward_calls"] for r in treatment)
                print(
                    f"  Graph coverage: {graph_coverage:.1%}; eager fallback calls: {eager_calls:.0f}"
                )
                reasons = ("missing_bucket", "layout_exceeded")
                for reason in reasons:
                    count = median(
                        r["metrics"]["graph_fallback_reasons"][reason] for r in treatment
                    )
                    print(f"    {reason}: {count:.0f}")
            if all("peak_cuda_memory_gib" in run["metrics"] for run in [*baseline, *treatment]):
                b_mem = number(baseline, "peak_cuda_memory_gib")
                t_mem = number(treatment, "peak_cuda_memory_gib")
                print(
                    f"  Peak CUDA allocated: {b_mem:.3f} -> {t_mem:.3f} GiB "
                    f"(reduction {format_percent(percent(b_mem, t_mem, higher_better=False))})"
                )
            if scenario == "conversation":
                followup = treatment[0]["details"]["followup_requests"]
                restored_fraction = median(
                    run["details"]["restored_prompt_token_ratio"] for run in treatment
                )
                completion_hits = median(
                    run["details"]["completion_checkpoint_hits"] for run in treatment
                )
                prompt_hits = median(run["details"]["prompt_checkpoint_hits"] for run in treatment)
                print(
                    f"  Follow-up TTFT ({followup} req/run): "
                    f"{format_ms(milliseconds(baseline, 'ttft_seconds'))} -> "
                    f"{format_ms(milliseconds(treatment, 'ttft_seconds'))} ms"
                )
                print(
                    f"  Follow-up restored/input: {restored_fraction:.1%}; "
                    f"Completion hits: {completion_hits:.0f}; Prompt hits: {prompt_hits:.0f}"
                )
                b_elapsed = number(baseline, "elapsed_seconds")
                t_elapsed = number(treatment, "elapsed_seconds")
                print(
                    f"  Accumulated engine execution (all turns): "
                    f"{b_elapsed:.3f} -> {t_elapsed:.3f} s "
                    f"(reduction {format_percent(percent(b_elapsed, t_elapsed, higher_better=False))})"
                )
            if scenario == "prefix":
                restored = median(
                    run["details"]["total_restored_prefix_tokens"] for run in treatment
                )
                hits = median(run["details"]["prefix_hit_rate"] for run in treatment)
                reused_fraction = median(
                    run["details"]["restored_prompt_token_ratio"] for run in treatment
                )
                mean_restored = median(
                    run["details"]["mean_restored_prefix_tokens"] for run in treatment
                )
                snapshot_peaks = (
                    ("Resident", "peak_snapshot_bytes_including_setup"),
                    ("Staging", "peak_staging_snapshot_bytes_including_setup"),
                    ("Total", "peak_total_snapshot_bytes_including_setup"),
                )
                print(
                    f"  Prefix restored tokens: {restored:.0f}, "
                    f"mean/request: {mean_restored:.0f}, hit rate: {hits:.2f}, "
                    f"restored/input: {reused_fraction:.1%}"
                )
                print("  Peak CPU GDN snapshot tensors:")
                for name, field in snapshot_peaks:
                    peak_mib = median(run["details"][field] for run in treatment) / (1024 * 1024)
                    print(f"    {name}: {peak_mib:.2f} MiB")
                if p["shared_prefix_len"] == 0:
                    if restored:
                        raise SystemExit("Unshared-prefix workload unexpectedly restored tokens")
                    print("  Zero-overlap workload: TTFT reduction below zero is cache overhead")
                if all("published_snapshots_during_target" in r["details"] for r in treatment):
                    fields = (
                        "published_snapshots_during_target",
                        "evicted_snapshots_during_target",
                        "retained_snapshots_after_target",
                    )
                    for field in fields:
                        value = median(r["details"][field] for r in treatment)
                        print(f"  {field}: {value:.0f}")
            if latency_key:
                improvement = percent(
                    milliseconds(baseline, latency_key),
                    milliseconds(treatment, latency_key),
                    higher_better=False,
                )
                print(f"  {latency_key} reduction: {format_percent(improvement)}")
            if scenario == "interference":

                def long_ttft(rs):
                    return median(
                        r["metrics"]["long_prompt_interference"]["long_ttft_seconds"] * 1000
                        for r in rs
                    )

                b, t = long_ttft(baseline), long_ttft(treatment)
                print(
                    f"  Late long request TTFT: {b:.2f} -> {t:.2f} ms "
                    f"(reduction {format_percent(percent(b, t, higher_better=False))})"
                )
                # Count actual Decode+Prefill steps after injection, not just the configured cap.
                if all(
                    "post_arrival_mixed_steps" in r["metrics"]["long_prompt_interference"]
                    for r in [*baseline, *treatment]
                ):
                    print(
                        "  Mixed steps after injection: "
                        f"{median(r['metrics']['long_prompt_interference']['post_arrival_mixed_steps'] for r in baseline):.0f}"
                        " -> "
                        f"{median(r['metrics']['long_prompt_interference']['post_arrival_mixed_steps'] for r in treatment):.0f}; "
                        "Prefill tokens alongside Decode: "
                        f"{median(r['metrics']['long_prompt_interference']['post_arrival_mixed_prefill_tokens'] for r in baseline):.0f}"
                        " -> "
                        f"{median(r['metrics']['long_prompt_interference']['post_arrival_mixed_prefill_tokens'] for r in treatment):.0f}"
                    )
                # Small synthetic samples: mean only; do not label tail latency as production P99.
                tail_stats = []
                for label2, runs in (("exclusive", baseline), ("mixed", treatment)):
                    gap = median(
                        r["metrics"]["long_prompt_interference"]["short_token_gap_after_seconds"][
                            "mean"
                        ]
                        * 1000
                        for r in runs
                    )
                    samples = min(
                        r["metrics"]["long_prompt_interference"].get(
                            "short_token_gap_after_samples", 0
                        )
                        for r in runs
                    )
                    p99_values = [
                        r["metrics"]["long_prompt_interference"][
                            "short_token_gap_after_seconds"
                        ].get("p99")
                        for r in runs
                    ]
                    p99 = (
                        median(value * 1000 for value in p99_values)
                        if samples >= args.min_itl_samples
                        and all(value is not None for value in p99_values)
                        else None
                    )
                    tail_stats.append((p99, samples))
                    print(
                        f"  Post-arrival observed token gap ({label2}): "
                        f"mean={gap:.2f} ms, p99={format_ms(p99).strip()} ms, "
                        f"min samples/run={samples} (synthetic workload)"
                    )
                if all(p99 is not None for p99, _ in tail_stats):
                    reduction = percent(tail_stats[0][0], tail_stats[1][0], higher_better=False)
                    print(f"  Synthetic P99 ITL reduction: {format_percent(reduction)}")
                else:
                    print(
                        "  Synthetic P99 ITL comparison withheld: fewer than "
                        f"{args.min_itl_samples} gaps in at least one run"
                    )
            compared += 1
    print(f"\nSuccessful paired comparisons: {compared}")
    if failed:
        issues.append("failed/incomplete experiments: " + ", ".join(failed))
    if issues:
        for issue in issues:
            print("INCOMPLETE:", issue)
        raise SystemExit(
            "Some runs could not be compared. No unsupported speedup should be reported."
        )


if __name__ == "__main__":
    main()
