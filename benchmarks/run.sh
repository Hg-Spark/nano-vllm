#!/usr/bin/env bash
# Engine-level benchmark experiments; run from any working directory.
# 'long', 'prefix', and 'conversation' are opt-in benchmark modes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
: "${NANOVLLM_MODEL:?export NANOVLLM_MODEL=/path/to/model}"
MODE="${1:-smoke}"
if (( $# > 1 )); then
  echo "Usage: bash benchmarks/run.sh [smoke|full|tail|long|prefix|conversation]" >&2
  exit 2
fi
case "$MODE" in
  smoke|full|tail|long|prefix|conversation) ;;
  *) echo "Unknown benchmark mode: $MODE" >&2; exit 2 ;;
esac
REPEATS="${REPEATS:-5}"
RESULT_DIR="${RESULT_DIR:-results/${MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"

# Shared model KV storage settings; BF16 model compute precision is fixed.
KV_CACHE_DTYPE="${NANOVLLM_KV_CACHE_DTYPE:-auto}"
KV_CACHE_K_SCALE="${NANOVLLM_KV_CACHE_K_SCALE:-1.0}"
KV_CACHE_V_SCALE="${NANOVLLM_KV_CACHE_V_SCALE:-1.0}"
COMMON=(
  "$NANOVLLM_MODEL" --seed 0
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --kv-cache-k-scale "$KV_CACHE_K_SCALE"
  --kv-cache-v-scale "$KV_CACHE_V_SCALE"
)
SHORT_REQUESTS=3
SHORT_OUTPUT=128
SHORT_MAX_SEQS=4
if [[ "$MODE" == "full" ]]; then
  BATCHES=(1 2 4 8)
  PREFIXES=(512 1024 2048)
  BUDGETS=(256 512 1024)
elif [[ "$MODE" == "prefix" || "$MODE" == "conversation" ]]; then
  BATCHES=()
  PREFIXES=()
  BUDGETS=()
elif [[ "$MODE" == "tail" || "$MODE" == "long" ]]; then
  BATCHES=()
  PREFIXES=()
  BUDGETS=(512)
  SHORT_REQUESTS=7
  SHORT_OUTPUT=512
  SHORT_MAX_SEQS=8
  if [[ "$MODE" == "long" ]]; then BUDGETS=(); fi
else
  BATCHES=(1 4)
  PREFIXES=(1024)
  BUDGETS=(512)
fi

# Reuse the same workload across both scheduling policies and parameter sweeps.
run_interference() {
  local budget="$1" limit="$2" mode="$3" length="$4" model_len="$5" label="$6"
  "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario interference \
    --num-requests "$SHORT_REQUESTS" --input-len 128 --output-len "$SHORT_OUTPUT" \
    --max-num-seqs "$SHORT_MAX_SEQS" --budget "$budget" --cache-entries 0 \
    --mixed-prefill-limit "$limit" --late-input-len "$length" \
    --max-model-len "$model_len" --late-after-decode-steps 8 \
    --schedule-mode "$mode" \
    --out "$RESULT_DIR/interference_${label}_${mode}_r${r}.json"
}

# Specialized cache evaluation: separate from full benchmark to bound run time.
# Two independent A/B workloads:
# (1) hot prefix, different suffixes; (2) fixed total input with 0-75% shared tokens.
run_prefix_probe() {
  local tag="$1" shared="$2" suffix="$3" batch="$4" model_len="$5" cache="$6"
  local targets="${7:-$batch}" require_evictions="${8:-0}"
  local EXTRA=()
  if [[ "$require_evictions" == "1" ]]; then EXTRA=(--expect-evictions); fi
  "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario prefix \
    --shared-prefix-len "$shared" --suffix-len "$suffix" \
    --target-requests "$targets" --max-num-seqs "$batch" \
    --output-len 16 --budget 512 --max-model-len "$model_len" \
    --cache-entries "$cache" "${EXTRA[@]}" \
    --out "$RESULT_DIR/prefix_${tag}_p${shared}_b${batch}_cache${cache}_r${r}.json"
}

for ((r=1; r<=REPEATS; r++)); do
  echo "Repeat $r of $REPEATS"
  if [[ "$MODE" == "conversation" ]]; then
    # Alternate AB/BA order across repetitions to reduce order effects.
    if (( r % 2 )); then CACHE_ORDER=(0 32); else CACHE_ORDER=(32 0); fi
    for cache in "${CACHE_ORDER[@]}"; do
      "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario conversation \
        --num-requests 8 --conversation-turns 6 --input-len 2048 \
        --suffix-len 256 --output-len 96 --max-model-len 4096 \
        --max-num-seqs 1 --budget 512 --cache-entries "$cache" \
        --out "$RESULT_DIR/conversation_cache${cache}_r${r}.json"
    done
    continue
  fi
  if [[ "$MODE" == "prefix" ]]; then
    for prefix in 512 2048 8192; do
      for batch in 1 4; do
        max_len=$((prefix + 256 + 16 + 128))
        for cache in 0 32; do
          run_prefix_probe hot "$prefix" 256 "$batch" "$max_len" "$cache"
        done
      done
    done
    # Isolate cache overhead and benefit as the shared fraction changes.
    # Total prompt length remains 4096 tokens across all variants.
    for prefix in 0 1024 2048 3072; do
      for cache in 0 32; do
        run_prefix_probe fraction "$prefix" "$((4096-prefix))" 1 4608 "$cache"
      done
    done
    # Small cache: new branch checkpoints should evict older checkpoints.
    # Same target tokens in OFF/ON runs; asserts actual eviction in ON.
    run_prefix_probe eviction 2048 256 1 2560 0 6
    run_prefix_probe eviction 2048 256 1 2560 2 6 1
    continue
  fi
  # Fixed active Decode: measure pure Decode without Prefill/queueing effects.
  for b in "${BATCHES[@]}"; do
    for mode in eager graph; do
      EXTRA=()
      if [[ "$mode" == graph ]]; then EXTRA=(--graph-sizes "$b"); fi
      "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario decode \
        --max-num-seqs "$b" --budget 512 --input-len 1024 --max-model-len 4096 \
        --warmup-steps 8 --measure-steps 128 "${EXTRA[@]}" \
        --out "$RESULT_DIR/decode_b${b}_${mode}_r${r}.json"
    done
  done
  # Multi-request shared-prefix cache; each A/B uses identical prompt tokens.
  for prefix in "${PREFIXES[@]}"; do
    for cache in 0 32; do
      "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario prefix \
        --max-num-seqs 4 --budget 512 --output-len 128 --max-model-len 4096 \
        --shared-prefix-len "$prefix" --suffix-len 256 \
        --target-requests 8 --cache-entries "$cache" \
        --out "$RESULT_DIR/prefix_p${prefix}_cache${cache}_r${r}.json"
    done
  done
  # Budget sweep: disable the mixed cap so the configured budget is effective.
  limit=128
  if [[ "$MODE" == "full" ]]; then limit=-1; fi
  for budget in "${BUDGETS[@]}"; do
    for mode in mixed exclusive_decode_first; do
      run_interference "$budget" "$limit" "$mode" 2048 4096 "budget${budget}"
    done
  done
  # Only Mixed changes with the mixed-prefill cap. Exclusive at budget=512
  # is equivalent for all caps and has already run in the Budget sweep.
  if [[ "$MODE" == "full" ]]; then
    for limit in 64 128 256; do
      run_interference 512 "$limit" mixed 2048 4096 "budget512_cap${limit}"
    done
  fi
  # Long-context cases are opt-in because model and KV capacity are required.
  if [[ "$MODE" == "long" ]]; then
    for length in 8192 16384; do
      max_context=$((length + SHORT_OUTPUT + 256))
      for mode in mixed exclusive_decode_first; do
        run_interference 512 128 "$mode" "$length" "$max_context" "long${length}"
      done
    done
  fi
  # Tail and long modes focus exclusively on interference.
  if [[ "$MODE" == "tail" || "$MODE" == "long" ]]; then
    continue
  fi
  # General throughput A/B (batch may vary dynamically, different from fixed Decode).
  for mode in eager graph; do
    EXTRA=()
    if [[ "$mode" == graph ]]; then EXTRA=(--graph-sizes 1,2,3,4 --min-graph-coverage 0.95); fi
    "$PYTHON" -m benchmarks.bench_experiments "${COMMON[@]}" --scenario throughput \
      --num-requests 16 --input-len 1024 --output-len 256 --max-model-len 4096 \
      --max-num-seqs 4 --budget 512 --cache-entries 0 "${EXTRA[@]}" \
      --out "$RESULT_DIR/throughput_${mode}_r${r}.json"
  done
done
"$PYTHON" -m benchmarks.summarize_bench "$RESULT_DIR" --min-repeats "$REPEATS" | tee "$RESULT_DIR/summary.txt"
echo "Completed: $RESULT_DIR"
