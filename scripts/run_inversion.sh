#!/usr/bin/env bash
# Run SpinFlow inversion from repository root (POSIX).
# Usage: ./scripts/run_inversion.sh [DATASET] [-- extra spinflow args...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

DATASET="${1:-YTDJ}"
shift || true
exec python -m spinflow --dataset "$DATASET" "$@"
