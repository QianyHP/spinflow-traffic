"""
Observation utilities (Edie gridding, SLE aggregation).

Single source of truth for computing macroscopic fields from trajectory data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_LANE_COUNT_THRESHOLD = 10  # samples per cell per lane


@dataclass(frozen=True)
class EdieSLEOutput:
    """SLE macroscopic fields on (t, x) grid."""

    rho_sle: np.ndarray  # [K, M] veh/m
    q_sle: np.ndarray    # [K, M] veh/s
    lanes: np.ndarray    # [M] effective lane count per cell
    dt_frame: float      # seconds


def compute_edie_sle(
    df: pd.DataFrame,
    *,
    dx: float,
    dt: float,
    t_start: float,
    t_end: float,
    road_length: float,
    fps: float,
    lane_count_threshold: int = DEFAULT_LANE_COUNT_THRESHOLD,
) -> EdieSLEOutput:
    """
    Compute Edie macroscopic fields (SLE) on a space-time grid.

    Requirements in df:
      - 't' time (s)
      - 's' position along direction (m)
      - 'vx' longitudinal speed (m/s) OR caller has precomputed it
      - 'lane' (optional, int). If missing, assumed single lane.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    dt_frame = 1.0 / fps

    if 'lane' not in df.columns:
        df = df.copy()
        df['lane'] = 1

    # window + bounds
    d = df[(df['t'] >= t_start) & (df['t'] < t_end)].copy()
    d = d[(d['s'] >= 0.0) & (d['s'] <= road_length)].copy()

    if 'vx' not in d.columns:
        raise ValueError("compute_edie_sle requires 'vx' (m/s) in df.")

    K = int((t_end - t_start) / dt)
    M = int(np.ceil(road_length / dx))
    Theta = np.zeros((K, M))  # vehicle time
    Xi = np.zeros((K, M))     # vehicle distance

    k_idx = np.clip(((d['t'] - t_start) / dt).values.astype(int), 0, K - 1)
    i_idx = np.clip((d['s'] / dx).values.astype(int), 0, M - 1)
    v_vals = np.abs(d['vx'].values)

    np.add.at(Theta, (k_idx, i_idx), dt_frame)
    np.add.at(Xi, (k_idx, i_idx), v_vals * dt_frame)

    # Effective lane count per cell: count lanes that have enough observations
    lanes = np.ones(M, dtype=int)
    if 'lane' in d.columns:
        lane_ids = sorted(d["lane"].dropna().unique().tolist())
        lane_mask_sum = np.zeros(M, dtype=int)
        for ln in lane_ids:
            if ln <= 0:
                continue
            di = d[d["lane"] == ln]
            if len(di) == 0:
                continue
            i_idx_ln = np.clip((di["s"] / dx).values.astype(int), 0, M - 1)
            counts = np.bincount(i_idx_ln, minlength=M)
            lane_mask_sum += (counts >= lane_count_threshold).astype(int)
        lanes = np.maximum(lane_mask_sum, 1)

    area = dx * dt
    rho_all = Theta / area  # veh/m (all lanes)
    q_all = Xi / area       # veh/s (all lanes)

    Lx = lanes[np.newaxis, :]
    rho_sle = rho_all / Lx
    q_sle = q_all / Lx

    return EdieSLEOutput(rho_sle=rho_sle, q_sle=q_sle, lanes=lanes, dt_frame=dt_frame)

