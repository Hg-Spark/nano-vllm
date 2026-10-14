#!/usr/bin/env bash
# Engine-level benchmark experiments; run from any working directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${NANOVLLM_MODEL:?export NANOVLLM_MODEL=/path/to/model}"
: "${NANOVLLM_DTYPE:?export NANOVLLM_DTYPE=float16 or bfloat16}"
MODE="${1:-smoke}"
if (( $# > 1 )); then
  echo "Usage: bash benchmarks/run.sh [smoke|full|preemption]" >&2
  exit 2
fi
case "$MODE" in
  smoke|full|preemption) ;;
  *) echo "Unknown benchmark mode: $MODE" >&2; exit 2 ;;
esac
REPEATS="${REPEATS:-3}"
RESULT_DIR="${RESULT_DIR:-results/${MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"

if [[ "$MODE" == "preemption" ]]; then
for ((r=1; r<=REPEATS; r++)); do
  for utilization in 0.70 0.75 0.80; do
    python -m benchmarks.bench_experiments "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
      --scenario preemption --max-model-len 4096 \
      --input-len 1024 --output-len 256 --num-requests 32 \
      --max-num-seqs 8 --budget 512 --cache-entries 0 \
      --gpu-memory-utilization "$utilization" --require-preemptions \
      --out "$RESULT_DIR/preempt_${utilization}_r${r}.json"
  done
done
else
  RUN_HF_VALIDATION="${RUN_HF_VALIDATION:-1}"
  if [[ "$RUN_HF_VALIDATION" == 1 ]]; then
    python -m scripts.verify_qwen3_5 "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" --max-new-tokens 64
    python -m scripts.verify_prefix_resume "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" --max-new-tokens 32
  fi
  COMMON=("$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" --max-model-len 4096 --seed 0)
  if [[ "$MODE" == "full" ]]; then
    BATCHES=(1 2 4 8)
    PREFIXES=(512 1024 2048)
    BUDGETS=(256 512 1024)
  else
    BATCHES=(1 4)
    PREFIXES=(1024)
    BUDGETS=(512)
  fi
for ((r=1; r<=REPEATS; r++)); do
  echo "Repeat $r of $REPEATS"
  # Fixed active Decode: measure pure Decode without Prefill/queueing effects.
  for b in "${BATCHES[@]}"; do
    for mode in eager graph; do
      EXTRA=()
      if [[ "$mode" == graph ]]; then EXTRA=(--graph-sizes "$b"); fi
      python -m benchmarks.bench_experiments "${COMMON[@]}" --scenario decode \
        --max-num-seqs "$b" --budget 512 --input-len 1024 \
        --warmup-steps 8 --measure-steps 128 "${EXTRA[@]}" \
        --out "$RESULT_DIR/decode_b${b}_${mode}_r${r}.json"
    done
  done
  # Multi-request shared-prefix cache; each A/B uses identical prompt tokens.
  for prefix in "${PREFIXES[@]}"; do
    for cache in 0 32; do
      python -m benchmarks.bench_experiments "${COMMON[@]}" --scenario prefix \
        --max-num-seqs 4 --budget 512 --output-len 128 \
        --shared-prefix-len "$prefix" --suffix-len 256 \
        --target-requests 8 --cache-entries "$cache" \
        --out "$RESULT_DIR/prefix_p${prefix}_cache${cache}_r${r}.json"
    done
  done
  # Compare mixed-step vs step-exclusive Decode-first with SAME real scheduler.
  for budget in "${BUDGETS[@]}"; do
    for mode in mixed exclusive_decode_first; do
      python -m benchmarks.bench_experiments "${COMMON[@]}" --scenario interference \
        --num-requests 3 --input-len 128 --output-len 128 \
        --max-num-seqs 4 --budget "$budget" --cache-entries 0 \
        --late-input-len 2048 --late-after-decode-steps 8 \
        --schedule-mode "$mode" \
        --out "$RESULT_DIR/interference_budget${budget}_${mode}_r${r}.json"
    done
  done
  # General throughput A/B (batch may vary dynamically, different from fixed Decode).
  for mode in eager graph; do
    EXTRA=()
    if [[ "$mode" == graph ]]; then EXTRA=(--graph-sizes 1,2,3,4); fi
    python -m benchmarks.bench_experiments "${COMMON[@]}" --scenario throughput \
      --num-requests 16 --input-len 1024 --output-len 256 \
      --max-num-seqs 4 --budget 512 --cache-entries 0 "${EXTRA[@]}" \
      --out "$RESULT_DIR/throughput_${mode}_r${r}.json"
  done
done
fi
python -m benchmarks.summarize_bench "$RESULT_DIR" --min-repeats "$REPEATS" | tee "$RESULT_DIR/summary.txt"
echo "Completed: $RESULT_DIR"
