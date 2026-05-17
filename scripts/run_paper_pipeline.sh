#!/usr/bin/env bash
# Paper tables + main/comparison figures (requires repo requirements.txt, incl. torch).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

python experiments/run_paper_benchmarks.py
python experiments/render_main_text_figures.py
python experiments/render_benchmark_comparison_figures.py
# Optional sensitivity figure (uses cache when present):  python experiments/run_regularization_sweep.py
