"""
Centralized filesystem paths for paper-reproduction scripts.

All paths are resolved relative to the repository root (parent of ``experiments/``).
``SPIN_SRC`` must appear on ``sys.path`` (via ``pip install -e .`` or ``sys.path.insert``)
before importing the ``spinflow`` package from ``src/spinflow``.
"""
from __future__ import annotations

import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
SPIN_SRC = os.path.join(REPO_ROOT, "src")
FIG_DIR = os.path.join(REPO_ROOT, "results", "paper_figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
