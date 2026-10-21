#!/usr/bin/env bash
# Engine-level benchmark experiments; run from any working directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
: "${NANOVLLM_MODEL:?export NANOVLLM_MODEL=/path/to/model}"
: "${NANOVLLM_DTYPE:?export NANOVLLM_DTYPE=float16 or bfloat16}"
MODE="${1:-smoke}"
if (( $# > 1 )); then
  echo "Usage: bash benchmarks/run.sh [smoke|full|tail]" >&2
  exit 2
fi
case "$MODE" in
  smoke|full|tail) ;;
  *) echo "Unknown benchmark mode: $MODE" >&2; exit 2 ;;
esac
REPEATS="${REPEATS:-5}"
RESULT_DIR="${RESULT_DIR:-results/${MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"

COMMON=("$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" --max-model-len 4096 --seed 0)
SHORT_REQUESTS=3
SHORT_OUTPUT=128
SHORT_MAX_SEQS=4
if [[ "$MODE" == "full" ]]; then
  BATCHES=(1 2 4 8)
  PREFIXES=(512 1024 2048)
  BUDGETS=(256 512 1024)
elif [[ "$MODE" == "tail" ]]; then
  BATCHES=()
  PREFIXES=()
  BUDGETS=(512)
  SHORT_REQUESTS=7
  SHORT_OUTPUT=512
  SHORT_MAX_SEQS=8
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
      "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario decode \
        --max-num-seqs "$b" --budget 512 --input-len 1024 \
        --warmup-steps 8 --measure-steps 128 "${EXTRA[@]}" \
        --out "$RESULT_DIR/decode_b${b}_${mode}_r${r}.json"
    done
  done
  # Multi-request shared-prefix cache; each A/B uses identical prompt tokens.
  for prefix in "${PREFIXES[@]}"; do
    for cache in 0 32; do
      "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario prefix \
        --max-num-seqs 4 --budget 512 --output-len 128 \
        --shared-prefix-len "$prefix" --suffix-len 256 \
        --target-requests 8 --cache-entries "$cache" \
        --out "$RESULT_DIR/prefix_p${prefix}_cache${cache}_r${r}.json"
    done
  done
  # Compare mixed-step vs step-exclusive Decode-first with SAME real scheduler.
  for budget in "${BUDGETS[@]}"; do
    for mode in mixed exclusive_decode_first; do
      "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario interference \
        --num-requests "$SHORT_REQUESTS" --input-len 128 --output-len "$SHORT_OUTPUT" \
        --max-num-seqs "$SHORT_MAX_SEQS" --budget "$budget" --cache-entries 0 \
        --late-input-len 2048 --late-after-decode-steps 8 \
        --schedule-mode "$mode" \
        --out "$RESULT_DIR/interference_budget${budget}_${mode}_r${r}.json"
    done
  done
  # Tail mode measures interference only; it does not repeat throughput/Graph experiments.
  if [[ "$MODE" == "tail" ]]; then
    continue
  fi
  # General throughput A/B (batch may vary dynamically, different from fixed Decode).
  for mode in eager graph; do
    EXTRA=()
    if [[ "$mode" == graph ]]; then EXTRA=(--graph-sizes 1,2,3,4); fi
    "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario throughput \
      --num-requests 16 --input-len 1024 --output-len 256 \
      --max-num-seqs 4 --budget 512 --cache-entries 0 "${EXTRA[@]}" \
      --out "$RESULT_DIR/throughput_${mode}_r${r}.json"
  done
done
"$PYTHON" -m benchmarks.summarize_bench "$RESULT_DIR" --min-repeats "$REPEATS" | tee "$RESULT_DIR/summary.txt"
echo "Completed: $RESULT_DIR"
