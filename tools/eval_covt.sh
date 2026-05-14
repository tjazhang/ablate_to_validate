#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/external/covt/upstream/eval.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "missing $TARGET" >&2
  echo "run: ./tools/bootstrap_overlay.py covt" >&2
  exit 1
fi

cd "$ROOT/external/covt/upstream"
bash "$TARGET" "$@"

