#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/external/mirage/upstream/test.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "missing $TARGET" >&2
  echo "run: ./tools/bootstrap_overlay.py mirage" >&2
  exit 1
fi

cd "$ROOT/external/mirage/upstream"
bash "$TARGET" "$@"

