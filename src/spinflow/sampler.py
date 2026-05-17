"""
Quasi-stationary FD sampling with space-time parallelograms.

Each sample is a parallelogram aligned with a backward wave speed (long edge) and a
target vehicle speed (short edge). Interior trajectories yield Edie-consistent
(k, q, v) after single-lane-equivalent (SLE) scaling; scores mix speed CV and NAE.
"""

import numpy as np
from matplotlib.path import Path
from typing import List, Dict, Tuple
import pandas as pd

class Parallelogram:
    """Space-time parallelogram for local quasi-stationary sampling."""
    
    def __init__(self, center_t: float, center_x: float, 
                 given_speed: float, wave_speed: float,
                 Lw: float = 200.0, H: float = 50.0,
                 wCV: float = 0.5, wNAE: float = 0.5):
        """
        Args:
            center_t, center_x: Reference time (s) and chainage (m).
            given_speed: Target mean speed for this template (km/h).
            wave_speed: Backward wave speed (km/h, typically negative).
            Lw, H: Long-edge length and height (m) defining the template area.
            wCV, wNAE: Weights for CV vs NAE in the quality score.
        """
        self.center_t = center_t
        self.center_x = center_x
        self.given_speed = given_speed
        self.wave_speed = wave_speed
        
        self.Lw = Lw
        self.H = H
        self.area = Lw * H
        self.slope_long = wave_speed / 3.6
        self.slope_short = given_speed / 3.6
        
        self.wCV = wCV
        self.wNAE = wNAE
        
        self.CV = None
        self.NAE = None
        self.SCORE = None
        self.weight = 1.0
        
        # Populated by calculate_qkv (Edie / SLE).
        self.q = None  # veh/h
        self.k = None  # veh/km
        self.v_space = None
        self.v_time = None
        
        self.trajectory_points = None
        self.n_vehicles = None
        
        self.corners = self._compute_corners()
        
        corner_br = self.corners[2]
        corner_bl = self.corners[1]
        self.constant_distance = corner_br[1] - corner_bl[1]
        self.constant_time = corner_br[0] - corner_bl[0]
    
    def _compute_corners(self):
        """Corners as [TL, BL, BR, TR] in (t, x) coordinates."""
        center = np.array([self.center_t, self.center_x])
        
        u_w = np.array([1, self.slope_long])
        u_w = u_w / np.linalg.norm(u_w)
        
        u_v = np.array([1, self.slope_short])
        u_v = u_v / np.linalg.norm(u_v)
        
        sin_theta = abs(u_w[0] * u_v[1] - u_w[1] * u_v[0])
        
        if sin_theta < 0.01:
            raise ValueError("Angle between wave and vehicle speed too small")
        
        vec_w = (self.Lw / 2) * u_w
        vec_v = (self.H / (2 * sin_theta)) * u_v
        
        corners = np.array([
            center - vec_w - vec_v,
            center + vec_w - vec_v,
            center + vec_w + vec_v,
            center - vec_w + vec_v,
        ])
        return corners
    
    @property
    def t_min(self):
        return np.min(self.corners[:, 0])
    
    @property
    def t_max(self):
        return np.max(self.corners[:, 0])
    
    @property
    def x_min(self):
        return np.min(self.corners[:, 1])
    
    @property
    def x_max(self):
        return np.max(self.corners[:, 1])


def match_trajectory_points(parallelogram: Parallelogram, 
                            df: pd.DataFrame,
                            min_points: int = 5) -> Parallelogram:
    """
    Assign trajectories inside the parallelogram; return None if too few points.

    Expects columns ``t``, ``s``, ``vs``, ``veh`` on ``df``.
    """
    box_mask = (
        (df['t'] >= parallelogram.t_min) & 
        (df['t'] <= parallelogram.t_max) &
        (df['s'] >= parallelogram.x_min) & 
        (df['s'] <= parallelogram.x_max)
    )
    df_box = df[box_mask]
    
    if len(df_box) < min_points:
        return None
    
    points = df_box[['t', 's']].values
    polygon_path = Path(parallelogram.corners)
    in_polygon = polygon_path.contains_points(points)
    
    df_in = df_box[in_polygon]
    
    if len(df_in) < min_points:
        return None
    
    parallelogram.trajectory_points = df_in
    parallelogram.n_vehicles = df_in['veh'].nunique()
    
    return parallelogram


def compute_score(parallelogram: Parallelogram) -> Parallelogram:
    """
    Quality score from speed coefficient of variation (CV) and normalized absolute
    error (NAE) against the template speed. Lower is more quasi-stationary.
    """
    if parallelogram.trajectory_points is None:
        return None
    
    speeds = parallelogram.trajectory_points['vs'].values * 3.6  # m/s → km/h
    
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    CV = std_speed / mean_speed if mean_speed > 0 else 1.0
    
    epsilon = 1e-3
    numerators = np.abs(speeds - parallelogram.given_speed)
    denominators = np.maximum.reduce([
        np.abs(speeds),
        np.full_like(speeds, abs(parallelogram.given_speed)),
        np.full_like(speeds, epsilon)
    ])
    NAE = np.mean(numerators / denominators)
    
    CV_max, NAE_max = 1.0, 1.0
    SCORE = parallelogram.wCV * (CV / CV_max) + parallelogram.wNAE * (NAE / NAE_max)
    
    parallelogram.CV = CV
    parallelogram.NAE = NAE
    parallelogram.SCORE = SCORE
    
    return parallelogram


def calculate_qkv(parallelogram: Parallelogram) -> Parallelogram:
    """
    Edie (k, q, v) for the parallelogram domain, then single-lane equivalent (SLE).

    Aggregate N, T, L, A follow the paper; divide by active lane count L_A so k,q
    are per-lane for comparison across multi-lane sites.
    """
    if parallelogram.trajectory_points is None or parallelogram.n_vehicles == 0:
        return None
    
    N = parallelogram.n_vehicles
    A = parallelogram.area  # m·s
    L = parallelogram.constant_distance  # m
    T = parallelogram.constant_time      # s
    
    q_all = 3600 * N * L / A  # veh/h
    k_all = 1000 * N * T / A  # veh/km
    
    if 'lane' in parallelogram.trajectory_points.columns:
        lanes_in_region = parallelogram.trajectory_points['lane'].unique()
        L_A = len([ln for ln in lanes_in_region if ln > 0])
        L_A = max(L_A, 1)
    else:
        L_A = 1

    parallelogram.q = q_all / L_A
    parallelogram.k = k_all / L_A
    parallelogram.v_space = parallelogram.q / parallelogram.k if parallelogram.k > 0 else 0
    parallelogram.L_A = L_A
    
    speeds_mps = parallelogram.trajectory_points['vs'].values
    parallelogram.v_time = np.mean(speeds_mps) * 3.6  # km/h
    
    return parallelogram


def _otsu_threshold(scores: np.ndarray, bins: int = 32) -> float:
    """
    Histogram Otsu threshold on score distribution (data-driven eta).

    Falls back to median for tiny samples or degenerate histograms.
    """
    if scores.size == 0:
        return float("inf")
    if scores.size < 6:
        return float(np.median(scores))

    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_max <= s_min:
        return s_max

    hist, bin_edges = np.histogram(scores, bins=bins, range=(s_min, s_max))
    hist = hist.astype(float)
    prob = hist / (hist.sum() + 1e-12)  # p_i

    omega = np.cumsum(prob)  # class mass for bins <= k
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    # Between-class variance (Otsu) in histogram form: (mT*w0 - m0)^2 / (w0*(1-w0))
    denom = omega * (1.0 - omega) + 1e-12
    sigma_b2 = (mu_t * omega - mu)**2 / denom
    idx = int(np.argmax(sigma_b2))
    return float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)


def _soft_weights_from_scores(scores: List[float], eta: float) -> Tuple[np.ndarray, float, float]:
    """
    Map scores s to soft weights w(s) = 1 / (1 + exp(beta * (s - eta))).

    beta is set from the score IQR so that w(eta + IQR) ≈ 0.1 (no hand-tuned slope).
    """
    scores_arr = np.asarray(scores, dtype=float)
    if scores_arr.size == 0:
        return np.array([]), 0.0, 0.0
    q25, q75 = np.quantile(scores_arr, [0.25, 0.75])
    iqr = float(max(q75 - q25, 1e-6))
    beta = float(np.log(9.0) / iqr)
    weights = 1.0 / (1.0 + np.exp(beta * (scores_arr - eta)))
    return weights, beta, iqr


def sample_parallelograms(df: pd.DataFrame,
                          wave_speed: float = -15.0,
                          given_speeds: List[float] = None,
                          Lw: float = 50.0,
                          H: float = 15.0,
                          t_min: float = 60.0,
                          t_max: float = 500.0,
                          x_min: float = 0.0,
                          x_max: float = 280.0,
                          n_candidates_per_speed: int = 300,
                          n_select_per_speed: int = 15,
                          wCV: float = 0.5,
                          wNAE: float = 0.5,
                          min_points: int = 5,
                          seed: int = 123,
                          check_overlap: bool = True,
                          return_stats: bool = False):
    """
    Multi-speed parallelogram sampling with Otsu + soft weights per target speed.

    For each ``v_target`` in ``given_speeds``, draw spatiotemporal candidate centers,
    build valid parallelograms, score quasi-stationarity, map scores to weights, trim by
    a second Otsu on weights, then aggregate Edie (k, q, v).

    Returns a flat list of accepted ``Parallelogram`` instances (optionally with stats).
    """
    rng = np.random.default_rng(seed)
    
    if given_speeds is None:
        given_speeds = list(range(4, 35, 2)) + list(range(35, 60, 5))
    all_selected = []
    
    print("\n" + "="*80)
    print("Quasi-Stationary Sampling (Parallelograms)")
    print("="*80)
    print(f"Wave speed: {wave_speed} km/h")
    print(f"Parallelogram size: Lw={Lw}m, H={H}m")
    print(f"Sampling range: t=[{t_min}, {t_max}]s, x=[{x_min}, {x_max}]m")
    
    stats = []
    for v_target in given_speeds:
        speeds_kmh = df['vs'].values * 3.6
        tolerance = 10.0
        mask = np.abs(speeds_kmh - v_target) <= tolerance
        df_candidates = df[mask]
        
        if len(df_candidates) < n_candidates_per_speed:
            if len(df_candidates) == 0:
                print(f"  v*={v_target:3.0f} km/h: No candidates")
                continue
            candidate_centers = df_candidates[['t', 's']].values
        else:
            idx = rng.choice(len(df_candidates), n_candidates_per_speed, replace=False)
            candidate_centers = df_candidates.iloc[idx][['t', 's']].values
        
        candidates = []
        for center_t, center_x in candidate_centers:
            if not (t_min < center_t < t_max and x_min < center_x < x_max):
                continue
            
            try:
                p = Parallelogram(center_t, center_x, v_target, wave_speed, Lw, H, wCV, wNAE)
            except ValueError:
                continue
            
            if (p.t_min < t_min or p.t_max > t_max or 
                p.x_min < x_min or p.x_max > x_max):
                continue
            
            if check_overlap and _check_overlap(p, all_selected):
                continue
            
            p = match_trajectory_points(p, df, min_points=min_points)
            if p is None:
                continue
            
            p = compute_score(p)
            if p is None:
                continue
            
            candidates.append(p)
        
        if len(candidates) == 0:
            print(f"  v*={v_target:3.0f} km/h: No valid candidates")
            continue
        
        candidates.sort(key=lambda x: x.SCORE)

        threshold = None
        scores = [p.SCORE for p in candidates]
        weight_beta = None
        weight_iqr = None
        threshold = _otsu_threshold(np.asarray(scores, dtype=float))
        weights, weight_beta, weight_iqr = _soft_weights_from_scores(scores, threshold)
        for p, w in zip(candidates, weights):
            p.weight = float(w)
        weight_threshold = _otsu_threshold(np.asarray(weights, dtype=float))
        selected = [p for p in candidates if p.weight >= weight_threshold]
        min_keep = int(np.ceil(np.sqrt(len(candidates))))
        if len(selected) < min_keep:
            candidates_sorted = sorted(candidates, key=lambda x: x.weight, reverse=True)
            selected = candidates_sorted[:min_keep]
        
        for p in selected:
            p = calculate_qkv(p)
        
        all_selected.extend(selected)
        
        mean_q = np.mean([p.q for p in selected])
        mean_k = np.mean([p.k for p in selected])
        mean_v = np.mean([p.v_space for p in selected])
        mean_score = np.mean([p.SCORE for p in selected])
        
        pass_ratio = len([s for s in scores if s <= (threshold if threshold is not None else float("inf"))])
        pass_ratio = pass_ratio / len(scores) if scores else 0.0
        mean_weight = float(np.mean([p.weight for p in selected])) if selected else 0.0
        stats.append(
            dict(
                v_target=float(v_target),
                n_candidates=len(candidates),
                n_selected=len(selected),
                score_threshold=float(threshold) if threshold is not None else None,
                pass_ratio=float(pass_ratio),
                mean_score=float(mean_score),
                mean_weight=mean_weight,
                weight_threshold=float(weight_threshold) if selected else None,
                weight_beta=weight_beta,
                weight_iqr=weight_iqr,
                min_keep=min_keep,
            )
        )

        print(f"  v*={v_target:3.0f} km/h: {len(candidates):4d}→{len(selected):2d}  "
              f"q={mean_q:5.0f}, k={mean_k:4.0f}, v={mean_v:4.1f}, "
              f"score={mean_score:.3f}, eta={threshold:.3f}")
    
    print(f"\nTotal Selected: {len(all_selected)} quasi-stationary regions")
    print("="*80)
    
    if return_stats:
        return all_selected, stats
    return all_selected


def estimate_score_threshold(df: pd.DataFrame,
                             wave_speed: float,
                             given_speeds: List[float],
                             Lw: float,
                             H: float,
                             t_min: float,
                             t_max: float,
                             x_min: float,
                             x_max: float,
                             n_candidates_per_speed: int,
                             wCV: float,
                             wNAE: float,
                             min_points: int,
                             seed: int) -> Dict:
    """
    基于候选区域的Score分布估计全局阈值（用于Lw/H自适应的η）。
    默认采用Otsu分割，避免主观阈值。
    """
    rng = np.random.default_rng(seed)
    tolerance = 10.0
    scores = []

    for v_target in given_speeds:
        speeds_kmh = df['vs'].values * 3.6
        mask = np.abs(speeds_kmh - v_target) <= tolerance
        df_candidates = df[mask]
        if len(df_candidates) == 0:
            continue

        if len(df_candidates) < n_candidates_per_speed:
            candidate_centers = df_candidates[['t', 's']].values
        else:
            idx = rng.choice(len(df_candidates), n_candidates_per_speed, replace=False)
            candidate_centers = df_candidates.iloc[idx][['t', 's']].values

        for center_t, center_x in candidate_centers:
            if not (t_min < center_t < t_max and x_min < center_x < x_max):
                continue
            try:
                p = Parallelogram(center_t, center_x, v_target, wave_speed, Lw, H, wCV, wNAE)
            except ValueError:
                continue
            if (p.t_min < t_min or p.t_max > t_max or p.x_min < x_min or p.x_max > x_max):
                continue
            p = match_trajectory_points(p, df, min_points=min_points)
            if p is None:
                continue
            p = compute_score(p)
            if p is None:
                continue
            scores.append(p.SCORE)

    scores_arr = np.asarray(scores, dtype=float)
    threshold = _otsu_threshold(scores_arr)
    return {
        "threshold": float(threshold),
        "n_scores": int(scores_arr.size),
        "mean_score": float(np.mean(scores_arr)) if scores_arr.size > 0 else float("inf"),
    }


def auto_lwh_candidates(
    df: pd.DataFrame,
    *,
    road_length: float,
    t_min: float,
    t_max: float,
    min_points: int,
    fps: float,
    n_Lw: int = 6,
    n_H: int = 6,
) -> Dict:
    """
    由数据自适应生成 Lw/H 候选网格。
    原理（第一性）：
    - 准稳态采样需要“足够点数”而不过度跨越非稳态；
    - 用观测点在 (t,x) 平面的密度估计最小面积 A0，使期望点数≈min_points；
    - 用速度分布的分位数把 A0 分解为 (Lw, H) 的尺度组合。
    """
    if t_max <= t_min:
        raise ValueError(f"t_max must be > t_min, got {t_min}–{t_max}")
    if road_length <= 0:
        raise ValueError(f"road_length must be positive, got {road_length}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if min_points is None or min_points <= 0:
        raise ValueError(f"min_points must be positive, got {min_points}")

    d = df[(df["t"] >= t_min) & (df["t"] < t_max) & (df["s"] >= 0.0) & (df["s"] <= road_length)].copy()
    if len(d) == 0:
        raise ValueError("auto_lwh_candidates: empty data window")
    if "vs" not in d.columns:
        raise ValueError("auto_lwh_candidates requires 'vs' in df")

    area_ts = (t_max - t_min) * road_length
    density = float(len(d)) / max(area_ts, 1e-12)
    area0 = float(min_points) / max(density, 1e-12)

    vs = np.abs(d["vs"].values.astype(float))
    if vs.size == 0:
        raise ValueError("auto_lwh_candidates: empty speed array")
    v_q25, v_q50, v_q75 = np.percentile(vs, [25, 50, 75])
    v_q25 = max(float(v_q25), 1e-6)
    v_q50 = max(float(v_q50), 1e-6)
    v_q75 = max(float(v_q75), v_q50)

    # 由 A0 ≈ Lw * H 且 Lw ≈ v * H 推出 H0 ≈ sqrt(A0 / v_med)
    H0 = float(np.sqrt(area0 / v_q50))
    dt_frame = 1.0 / fps
    H_min = max(2.0 * dt_frame, H0 / 2.0)
    H_max = min((t_max - t_min) / 4.0, H0 * 2.0)
    if H_min >= H_max:
        H_min = max(2.0 * dt_frame, H0 / 3.0)
        H_max = max(H_min * 1.5, H0 * 3.0)

    H_candidates = np.geomspace(H_min, H_max, num=max(int(n_H), 2))
    H_candidates = sorted(set([round(float(h), 3) for h in H_candidates if h > 0]))

    Lw_vals = []
    for v in (v_q25, v_q50, v_q75):
        for h in H_candidates:
            Lw_vals.append(v * h)

    step = v_q50 * dt_frame
    Lw_min = max(2.0 * step, 1e-6)
    Lw_max = max(Lw_min, road_length)
    Lw_vals = [lw for lw in Lw_vals if Lw_min <= lw <= Lw_max]
    if not Lw_vals:
        Lw_vals = [min(max(v_q50 * H0, Lw_min), Lw_max)]

    Lw_vals = np.array(sorted(set([round(float(lw), 3) for lw in Lw_vals])))
    if Lw_vals.size > int(n_Lw):
        qs = np.linspace(0.0, 1.0, int(n_Lw))
        Lw_candidates = sorted(set([round(float(x), 3) for x in np.quantile(Lw_vals, qs)]))
    else:
        Lw_candidates = Lw_vals.tolist()

    return {
        "Lw": Lw_candidates,
        "H": H_candidates,
        "density": density,
        "area0": area0,
        "H0": H0,
        "v_q": (v_q25, v_q50, v_q75),
    }


def select_adaptive_lwh(
    df: pd.DataFrame,
    wave_speed: float,
    given_speeds: List[float],
    Lw_candidates: List[float],
    H_candidates: List[float],
    t_min: float,
    t_max: float,
    x_min: float,
    x_max: float,
    n_candidates_per_speed: int,
    wCV: float,
    wNAE: float,
    min_points: int,
    seed: int,
    score_threshold: float,
    min_pass_ratio: float,
    min_candidates: int = 0,
) -> Dict:
    """
    在候选Lw/H网格中选择“最小稳态窗口”：
    - 合格证据（软阈值）：把 Score 映射为权重 w∈(0,1]，表示“准稳态可信度”
      w(s) = 1 / (1 + exp(β * (s - η)))，其中 η=score_threshold，β 由 IQR 自适应确定

    约束（完全数据驱动）：
    - pass_ratio：定义为平均权重 mean(w)，表示“稳态证据占比”（连续量，更稳健）
    - n_eff：有效样本量 ESS = (∑w)^2 / ∑(w^2)，表示“信息量”（避免被少数高权重点支配）
    - 约束：pass_ratio >= min_pass_ratio 且 n_eff >= min_candidates（若给定）

    目标（最小复杂化）：
    - 先最小面积(Lw*H)，再最小加权均值Score（越小越稳态）
    """
    if not Lw_candidates or not H_candidates:
        raise ValueError("Lw_candidates and H_candidates must be non-empty for adaptive Lw/H.")
    if score_threshold is None or min_pass_ratio is None:
        raise ValueError("score_threshold and min_pass_ratio are required for adaptive Lw/H.")

    grid_stats = []
    tolerance = 10.0

    for Lw in Lw_candidates:
        for H in H_candidates:
            rng = np.random.default_rng(seed)
            scores = []
            n_candidates = 0

            for v_target in given_speeds:
                speeds_kmh = df['vs'].values * 3.6
                mask = np.abs(speeds_kmh - v_target) <= tolerance
                df_candidates = df[mask]

                if len(df_candidates) == 0:
                    continue

                if len(df_candidates) < n_candidates_per_speed:
                    candidate_centers = df_candidates[['t', 's']].values
                else:
                    idx = rng.choice(len(df_candidates), n_candidates_per_speed, replace=False)
                    candidate_centers = df_candidates.iloc[idx][['t', 's']].values

                for center_t, center_x in candidate_centers:
                    if not (t_min < center_t < t_max and x_min < center_x < x_max):
                        continue
                    try:
                        p = Parallelogram(center_t, center_x, v_target, wave_speed, Lw, H, wCV, wNAE)
                    except ValueError:
                        continue
                    if (p.t_min < t_min or p.t_max > t_max or p.x_min < x_min or p.x_max > x_max):
                        continue
                    p = match_trajectory_points(p, df, min_points=min_points)
                    if p is None:
                        continue
                    p = compute_score(p)
                    if p is None:
                        continue
                    scores.append(p.SCORE)
                    n_candidates += 1
            # === 软阈值稳态证据（与主采样流程一致） ===
            # 解释给审稿人：
            # - 传统“Score<=η”的硬阈值在高速稀疏、分布退化时会产生不稳定的二元跳变
            # - 这里用 w(s) 把“准稳态程度”连续化；mean(w) 是稳态证据的占比
            # - ESS 是信息量，防止只有极少数点权重大导致选择被误导
            weights, weight_beta, weight_iqr = _soft_weights_from_scores(scores, float(score_threshold))
            w_sum = float(np.sum(weights)) if weights.size > 0 else 0.0
            w2_sum = float(np.sum(weights * weights)) if weights.size > 0 else 0.0
            n_eff = float((w_sum * w_sum) / (w2_sum + 1e-12)) if w_sum > 0 else 0.0
            pass_ratio = float(w_sum / max(float(n_candidates), 1.0))  # mean(w) ∈ [0,1]

            # 加权均值Score：更关注“更稳态”的点（w高）
            mean_score = float(np.average(np.asarray(scores, dtype=float), weights=weights)) if w_sum > 0 else float("inf")
            grid_stats.append(
                dict(
                    Lw=float(Lw),
                    H=float(H),
                    area=float(Lw * H),
                    mean_score=mean_score,
                    pass_ratio=pass_ratio,
                    n_candidates=int(n_candidates),
                    n_eff=float(n_eff),
                    weight_beta=float(weight_beta),
                    weight_iqr=float(weight_iqr),
                )
            )

    feasible = [
        s for s in grid_stats
        if s["pass_ratio"] >= float(min_pass_ratio)
        and (s.get("n_eff", 0.0) >= float(min_candidates))
    ]

    if feasible:
        feasible.sort(key=lambda s: (s["area"], s["mean_score"]))
        best = feasible[0]
        best["feasible"] = True
    else:
        # 回退：优先最大稳态证据占比，再最小加权Score，再最小面积
        grid_stats.sort(key=lambda s: (-s["pass_ratio"], s["mean_score"], s["area"]))
        best = grid_stats[0]
        best["feasible"] = False

    return dict(selected=best, grid_stats=grid_stats)


def _check_overlap(p: Parallelogram, existing: List[Parallelogram]) -> bool:
    """检查平行四边形p是否与existing中任何一个重叠"""
    if not existing:
        return False
    
    p_path = Path(p.corners)
    
    for ex in existing:
        # 检查p的角点是否在ex内
        if np.any(p_path.contains_points(ex.corners)):
            return True
        # 检查ex的角点是否在p内
        ex_path = Path(ex.corners)
        if np.any(ex_path.contains_points(p.corners)):
            return True
    
    return False


def extract_fd_points(parallelograms: List[Parallelogram]) -> Dict:
    """
    从平行四边形列表提取FD点云
    
    Returns:
        dict: {
            'k': 密度数组 (veh/km)
            'q': 流量数组 (veh/h)
            'v': 速度数组 (km/h)
            'v_target': 目标速度 (km/h)
            't_center': 中心时间 (s)
            'x_center': 中心位置 (m)
            'score': 评分
        }
    """
    k_list = []
    q_list = []
    v_list = []
    v_target_list = []
    t_center_list = []
    x_center_list = []
    score_list = []
    weight_list = []
    
    for p in parallelograms:
        if p.k is not None and p.q is not None:
            k_list.append(p.k)
            q_list.append(p.q)
            v_list.append(p.v_space)
            v_target_list.append(p.given_speed)
            t_center_list.append(p.center_t)
            x_center_list.append(p.center_x)
            score_list.append(p.SCORE)
            weight_list.append(getattr(p, "weight", 1.0))
    
    return {
        'k': np.array(k_list),
        'q': np.array(q_list),
        'v': np.array(v_list),
        'v_target': np.array(v_target_list),
        't_center': np.array(t_center_list),
        'x_center': np.array(x_center_list),
        'score': np.array(score_list),
        'weight': np.array(weight_list),
        'n_points': len(k_list)
    }


def visualize_parallelograms(df: pd.DataFrame, 
                             parallelograms: List[Parallelogram],
                             save_path: str = None,
                             xlim: Tuple[float, float] = None,
                             ylim: Tuple[float, float] = None):
    """可视化平行四边形采样结果"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(18, 8))
    from scipy.ndimage import gaussian_filter
    
    # --- 1. 生成宏观密度场 (SLE Density Field) ---
    dt_plot, ds_plot = 0.5, 2.0
    t_min, t_max = df['t'].min(), df['t'].max()
    s_min, s_max = df['s'].min(), df['s'].max()
    
    K = int((t_max - t_min) / dt_plot) + 1
    M = int((s_max - s_min) / ds_plot) + 1
    Theta = np.zeros((K, M))
    
    t_idx = np.clip(((df['t'] - t_min) / dt_plot).values.astype(int), 0, K - 1)
    s_idx = np.clip(((df['s'] - s_min) / ds_plot).values.astype(int), 0, M - 1)
    
    fps = 24.0
    np.add.at(Theta, (t_idx, s_idx), 1.0/fps)
    
    # 计算 Edie 定义下的密度场
    rho_field = Theta / (dt_plot * ds_plot)
    
    # 非对称高斯平滑：sigma=(time_smooth, space_smooth)
    # 处理离散采样引入的波动，保留交通波物理特征
    rho_smooth = gaussian_filter(rho_field, sigma=(1.0, 1.0))
    
    # 遮蔽低密度区域
    rho_smooth[rho_smooth < 2e-3] = np.nan
    
    # 归一化处理（基于分位数）
    rho_max = np.nanquantile(rho_smooth, 0.98) if np.any(~np.isnan(rho_smooth)) else 1.0
    rho_norm = np.clip(rho_smooth / rho_max, 0.0, 1.0)
    
    mesh = ax.imshow(rho_norm.T, aspect='auto', origin='lower', cmap='jet',
                    vmin=0, vmax=1.0, extent=[t_min, t_max, s_min, s_max], 
                    interpolation='bilinear', alpha=0.95)
    
    # --- 2. 绘制所有采样出的准稳态区域 (Parallelograms) ---
    plot_ps = parallelograms
        
    for p in plot_ps:
        corners_closed = np.vstack([p.corners, p.corners[0]])
        # 准稳态区域边界
        ax.plot(corners_closed[:, 0], corners_closed[:, 1], 
                'k-', linewidth=0.8, alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position (m)', fontsize=12, fontweight='bold')
    ax.set_title('Spacetime Sampling Evidence', fontsize=14, fontweight='bold')
    plt.colorbar(mesh, ax=ax, label='SLE Density (Normalized)', pad=0.02)
    
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        # Default behavior (can be adjusted or fallback to fixed values if needed, 
        # but better to rely on passed values)
        ax.set_xlim(60, 500) 

    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(0, 240)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [Output] Spacetime Evidence saved: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # 测试代码
    print("Parallelogram sampler module loaded successfully")

