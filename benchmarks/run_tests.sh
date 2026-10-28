#!/usr/bin/env bash
# Run tests in an already provisioned Python environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  echo 'Usage: bash benchmarks/run_tests.sh [CONFIG.toml] [all|check|bench|tail|long|capacity]'
  echo 'Default config: configs/tests.toml; default stage: the stage in that file.'
  exit 0
fi
(( $# <= 2 )) || { echo 'Expected a config path and an optional stage.' >&2; exit 2; }
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
ARGS=(--config "${1:-configs/tests.toml}")
if (( $# == 2 )); then ARGS+=(--stage "$2"); fi
exec "$PYTHON" -m benchmarks.run_tests "${ARGS[@]}"
