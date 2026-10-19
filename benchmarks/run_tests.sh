#!/usr/bin/env bash
# uv installs dependencies into system Python; tests reuse that interpreter.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  echo 'Usage: bash benchmarks/run_tests.sh [CONFIG.toml] [all|check|bench|tail|capacity]'
  echo 'Default config: configs/tests.toml; default stage: the stage in that file.'
  exit 0
fi
(( $# <= 2 )) || { echo 'Expected a config path and an optional stage.' >&2; exit 2; }
command -v uv >/dev/null || { echo 'uv is required; install it before running this script.' >&2; exit 2; }
cd "$ROOT"
PYTHON="$(uv python find --system 3.12)"
uv pip install --system --python "$PYTHON" --break-system-packages \
  -r pyproject.toml --extra dev
ARGS=(--config "${1:-configs/tests.toml}")
if (( $# == 2 )); then ARGS+=(--stage "$2"); fi
exec "$PYTHON" -m benchmarks.run_tests "${ARGS[@]}"
