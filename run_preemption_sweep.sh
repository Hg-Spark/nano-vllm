#!/usr/bin/env bash
# Optional stress test. It may fail at low budgets or observe zero preemptions.
set -euo pipefail
: "${NANOVLLM_MODEL:?export NANOVLLM_MODEL=/path/to/model}"
: "${NANOVLLM_DTYPE:?export NANOVLLM_DTYPE=float16 or bfloat16}"
RESULT_DIR="${RESULT_DIR:-results/preempt_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
mkdir -p "$RESULT_DIR"
for ((r=1; r<=REPEATS; r++)); do
  for utilization in 0.70 0.75 0.80; do
    python bench_experiments.py "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
      --scenario preemption --max-model-len 4096 \
      --input-len 1024 --output-len 256 --num-requests 32 \
      --max-num-seqs 8 --budget 512 --cache-entries 0 \
      --gpu-memory-utilization "$utilization" \
      --out "$RESULT_DIR/preempt_${utilization}_r${r}.json"
  done
done
python summarize_bench.py "$RESULT_DIR" --min-repeats "$REPEATS" | tee "$RESULT_DIR/summary.txt"
echo "Check preemptions in every JSON; zero preemptions means this workload did not exercise recovery."
