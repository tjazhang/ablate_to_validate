#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/external/mull/upstream/src/eval_bench_ablation_novllm.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "missing $TARGET" >&2
  echo "run: ./tools/bootstrap_overlay.py mull" >&2
  exit 1
fi

cd "$ROOT/external/mull/upstream"
bash "$TARGET" "$@"

