# NanoVLLM Qwen3.5 Hybrid benchmark package (v2)

Target: https://github.com/Hg-Spark/nano-vllm/tree/qwen3.5 (API reviewed 2026-10-15).

Place `bench_experiments.py`, `summarize_bench.py`, `run_sweep.sh` and optional
`run_preemption_sweep.sh` in the repository **root**, alongside `scripts/`.
Install the repository dependencies and provide a local, unquantized Qwen3.5 checkpoint.

## Complete experiments

```bash
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5
export NANOVLLM_DTYPE=float16   # bfloat16 if supported

REPEATS=1 bash run_sweep.sh             # first: smoke test, 2 decode batch sizes
REPEATS=5 FULL_SWEEP=1 bash run_sweep.sh # full repeats: batch 1/2/4/8, three prefix lengths and budgets
```

All runs use different processes (model initialization is excluded from the performance
window). The default sweep first verifies HF greedy parity and joint prefix resume;
set `RUN_HF_VALIDATION=0` only if these validations have already passed on the **same**
checkpoint/commit/device. Raw JSON and `summary.txt` are written under a timestamped
`results/ab_*` directory. Failed runs do not become passing measurements.

Each experiment can also be run separately:

### A. Fixed active Decode, Eager vs CUDA Graph

```bash
python bench_experiments.py "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario decode --input-len 1024 --max-num-seqs 4 \
  --warmup-steps 8 --measure-steps 128 --max-model-len 4096 \
  --out results/decode_eager.json

python bench_experiments.py "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario decode --input-len 1024 --max-num-seqs 4 --graph-sizes 4 \
  --warmup-steps 8 --measure-steps 128 --max-model-len 4096 \
  --out results/decode_graph.json
```

Uses the project's `prepare_fixed_decode_batch`, Engine scheduler, real model runner and
actual Graph replay. Each measured step must process exactly **4** Decode tokens and
zero Prefill tokens. Reports wall-clock Decode tok/s, observed per-step TPOT, GPU-stream
elapsed time and actual Graph replays. The GPU Event window includes host scheduling
and launch gaps; it is **not** an exclusive kernel execution-time sum.
The output hash includes the prepared/warmup context; matched hashes permit A/B parity.
The fixed batch is synthetic and does not imply production online TPOT P99.

### B. Shared-prefix multi-request A/B

```bash
python bench_experiments.py "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario prefix --shared-prefix-len 1024 --suffix-len 256 \
  --target-requests 8 --cache-entries 0 --out results/prefix_off.json

python bench_experiments.py "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario prefix --shared-prefix-len 1024 --suffix-len 256 \
  --target-requests 8 --cache-entries 32 --out results/prefix_on.json
```

Runs a matching source prompt **outside** the measured target window and then eight
target prompts with identical shared prefix and different suffixes. It records the
first target Prefill chunk's `start` offset, prefix hit ratio, total restored tokens,
cache entries and GDN Snapshot **stored byte count**. Snapshot transfer latency and
CPU peak memory are **not** currently measured: profile `nanovllm::gdn_snapshot_d2h`
separately, and do not equate stored bytes with GPU peak savings. Shared prefix length
must be a 16-token multiple. Valid tested prefix lengths also depend on model/context.

### C. Mixed-step vs step-exclusive Decode-first

```bash
python bench_experiments.py "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario interference --num-requests 3 --max-num-seqs 4 \
  --input-len 128 --output-len 128 --late-input-len 2048 \
  --late-after-decode-steps 8 --budget 512 \
  --schedule-mode exclusive_decode_first --out results/exclusive.json

python bench_experiments.py "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario interference --num-requests 3 --max-num-seqs 4 \
  --input-len 128 --output-len 128 --late-input-len 2048 \
  --late-after-decode-steps 8 --budget 512 \
  --schedule-mode mixed --out results/mixed.json
```

The control uses the **same scheduler/resources**, but skips Prefill when any Decode
chunks exist. Both still support Chunked Prefill, so it tests *mixed-step execution*
vs *Decode-first step exclusivity*, **not** the unmodified original nano-vLLM
scheduler or true unchunked Prefill. `run_benchmark` returns long-request TTFT,
short-request observed token gaps and output throughput. Run the same A/B with token
budgets 256/512/1024 to test sensitivity; changing budget alone does not isolate the
scheduling algorithm. Small synthetic P99 estimates are not claims about online tail
latency. Both modes can starve new Prefill in unusual saturation cases.

### D. Hybrid Preemption (optional)

```bash
REPEATS=3 bash run_preemption_sweep.sh
```

Scans GPU allocation budgets 0.70/0.75/0.80, logging actual KV block capacity,
preemption count and sum of committed tokens held by victims. This counter is **not**
a measured restoration time, and the sum is **not** automatically equal to recomputed
Tokens. Some GPUs/configurations fail before a high-pressure run starts; report these
as FAILED/BLOCKED. If all preemption counts remain zero, adjust request lengths,
concurrency or budgets. To refuse successful runs without preemption, append
`--require-preemptions` to the individual benchmark command.

## Reading the summary

```bash
python summarize_bench.py results/ab_TIMESTAMP --min-repeats 5
```

- Checks token output hash for all comparable runs **before** displaying percentage changes.
  Hashes must match for A/B as well as repeat runs.
- Displays medians of independent process measurements; improvement formulas use
  `(treatment/base-1)*100%` for throughput and `(1-treatment/base)*100%` for latency.
- Refuses unmatched A/B, runs with too few repeats and failed samples. Different
  GPUs, commits, runtime args and scripts should never be pooled.
- Model checkpoint/revision and driver, PyTorch, CUDA, FlashInfer versions must be
  recorded separately in a final report. This script records paths, GPU, torch/CUDA,
  script hash and Git state but **does not freeze checkpoint file contents**.

## Limitations

These are **direct Engine offline** benchmarks (no network, HTTP request arrivals,
open-loop QPS, Goodput, or production P99). The original `scripts.bench.run_benchmark`
records completion after step-level GPU synchronization; host-visible token gaps
include decode and prefill work in the step. They do not isolate GPU kernel time.
Results must be preceded by passing correctness tests for the actual GPU, model and
runtime configuration. The package has been Python/Shell syntax-checked and its
summarizer smoke-tested with synthetic JSON; **CUDA execution is not yet verified**.
