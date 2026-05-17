"""Resolve paths relative to the SpinFlow repository root."""
from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent  # .../src/spinflow


def repo_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return _PKG_DIR.parent.parent


def pipeline_config_path() -> Path:
    """Default YAML merged into CLI args: ``configs/pipeline.yaml`` (repo root)."""
    return repo_root() / "configs" / "pipeline.yaml"


def data_dir() -> Path:
    return repo_root() / "data"


def results_dir() -> Path:
    return repo_root() / "results"


def paper_dir() -> Path:
    return repo_root() / "paper"
