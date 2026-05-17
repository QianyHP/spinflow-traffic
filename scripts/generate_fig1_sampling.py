"""
Fig. 1 — Parallelogram vs. Rectangular spatiotemporal sampling.

Key design choices:
  • Asymmetric shock: formation W_FORM = -3 m/s (fast, abrupt onset),
    dissipation W_DISS = +1.5 m/s (slow, gradual clearing after T_CLEAR).
    This creates a triangular congested zone with a steep left edge and a
    shallow right edge — matching real bottleneck observations.
  • N = 50 vehicles launched from t = -18 s to t = 55 s, filling the entire
    lower-right region of the space–time diagram.
  • Each vehicle carries a per-vehicle sinusoidal oscillation (random phase/freq)
    to produce the realistic "wavy" appearance of real trajectory data.
  • Single parallelogram placed inside the congested quasi-stationary zone.
    Long edge ‖ W_FORM (wave speed); short edge ‖ V_C (vehicle speed),
    matching exactly the sampler.py Parallelogram geometry.
  • Sampling windows rendered at high zorder with thick, opaque borders so they
    are clearly visible above the trajectory layer.
  • Square subplot boxes via ax.set_box_aspect(1).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import time as _time
import os

# Paper-style figures land under results/paper_figures (same as experiments.paths.FIG_DIR).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(_REPO_ROOT, "results", "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
    "xtick.labelsize": 9,  "ytick.labelsize": 9,
    "figure.dpi": 400, "savefig.dpi": 400,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.65, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

rng = np.random.default_rng(42)

# ── Physics ──────────────────────────────────────────────────────────────────
V_F     = 8.0     # m/s   free-flow speed
V_C     = 2.0     # m/s   congested speed
W_FORM  = -3.0    # m/s   FAST formation shock  (upstream, |W_FORM| > W_DISS)
W_DISS  = +1.5    # m/s   SLOW dissipation shock (downstream, queue shrinking)
T_CLEAR = 28.0    # s     jam begins to clear
X_BOT   = 240.0   # m     bottleneck position (jam front, fixed)

DT = 0.08;  T_MAX = 68.0
t_view = (0.0, 40.0);  x_view = (50.0, 350.0)

def x_shock(t: float) -> float:
    """Dynamic shock rear position.
    Phase 1 (t < T_CLEAR): grows quickly upstream at W_FORM = -3 m/s.
    Phase 2 (t ≥ T_CLEAR): dissipates slowly downstream at W_DISS = +1.5 m/s.
    Net: the congested zone forms steeply then clears gradually — visually asymmetric.
    """
    if t < T_CLEAR:
        return X_BOT + W_FORM * t                          # = 240 − 3t
    return (X_BOT + W_FORM * T_CLEAR) + W_DISS * (t - T_CLEAR)  # = 156 + 1.5·(t−28)

# ── Trajectory simulation ────────────────────────────────────────────────────
N_VEH      = 50
t_launches = np.linspace(-18, 55, N_VEH)
osc_phases = rng.uniform(0, 2 * np.pi, N_VEH)
osc_freqs  = rng.uniform(0.22, 0.58, N_VEH)
osc_amps   = rng.uniform(0.20, 0.55, N_VEH)

def sim_vehicle(t_launch, phase, freq, amp, rng):
    t = max(0.0, t_launch)
    x = V_F * max(0.0, -t_launch) + rng.normal(0, 1.2)
    x = max(0.0, x)
    v = V_F + rng.normal(0, 0.5)
    ts, xs, vs = [t], [x], [v]

    while t < T_MAX and x < x_view[1] + 30:
        xsr = x_shock(t)
        osc = amp * np.sin(freq * t + phase)

        if x >= X_BOT:
            v_tgt    = V_F + 0.45 * osc
            n_std    = 0.48
        else:
            dist     = x - xsr
            alpha    = 1.0 / (1.0 + np.exp(-dist / 5.0))
            v_tgt    = V_F * (1 - alpha) + V_C * alpha + osc * (1 - alpha)
            n_std    = 0.48 * (1 - alpha) + 0.12 * alpha

        # First-order lag (τ ≈ 1.0 s)
        v += (DT / 1.0) * (v_tgt - v) + rng.normal(0, n_std) * DT
        v  = max(0.25, v)
        x += v * DT;  t += DT
        ts.append(t); xs.append(x); vs.append(v)

    return np.array(ts), np.array(xs), np.array(vs)

trajs = [sim_vehicle(t_launches[i], osc_phases[i],
                     osc_freqs[i], osc_amps[i], rng)
         for i in range(N_VEH)]

# ── Parallelogram geometry (mirrors sampler.py Parallelogram._compute_corners) ─
def make_para(ct, cx, Lw, H, v_star, wave_speed):
    """
    Build a parallelogram in (t, x) space.
    Long  edge ‖ wave_speed  (Lw = physical length along long axis, metres)
    Short edge ‖ v_star      (H  = perpendicular height, metres)
    Vertex ordering: [p0, p1, p2, p3] counter-clockwise.
    """
    center = np.array([ct, cx])
    u_w = np.array([1.0, wave_speed]); u_w /= np.linalg.norm(u_w)
    u_v = np.array([1.0, v_star]);     u_v /= np.linalg.norm(u_v)
    sin_theta = abs(u_w[0] * u_v[1] - u_w[1] * u_v[0])
    vec_w = (Lw / 2.0) * u_w
    vec_v = (H  / (2.0 * sin_theta)) * u_v
    return np.array([center - vec_w - vec_v,
                     center + vec_w - vec_v,
                     center + vec_w + vec_v,
                     center - vec_w + vec_v])

# Single congested parallelogram: all 4 vertices ≥11 m inside the jam → sharp histogram
# p0=(9.9,221.4) p1=(22.5,183.4) p2=(30.1,198.6) p3=(17.5,237.6) — all verified
P_CONG = make_para(ct=20, cx=210, Lw=40, H=12, v_star=V_C, wave_speed=W_FORM)

# Rectangle: straddles shock diagonally → ~50% free-flow + ~50% congested → bimodal histogram
# Shock at t=10: x=210m (within rect); at t=22: x=174m (within rect) ✓
RECT_T0, RECT_T1 = 10.0, 22.0
RECT_X0, RECT_X1 = 148.0, 228.0

# ── Extract speeds for histograms ────────────────────────────────────────────
def speeds_in_rect(t0, t1, x0, x1):
    sp = []
    for ts, xs, vs in trajs:
        m = (ts >= t0) & (ts <= t1) & (xs >= x0) & (xs <= x1)
        sp.extend(vs[m] * 3.6)
    return np.array(sp)

def speeds_in_poly(verts):
    path = MplPath(verts)
    sp = []
    for ts, xs, vs in trajs:
        m = path.contains_points(np.column_stack([ts, xs]))
        sp.extend(vs[m] * 3.6)
    return np.array(sp)

v_rect = speeds_in_rect(RECT_T0, RECT_T1, RECT_X0, RECT_X1)
v_para = speeds_in_poly(P_CONG)
print(f"  n_rect={len(v_rect):,}  n_para={len(v_para):,}")
print(f"  sigma_rect={np.std(v_rect):.2f}  sigma_para={np.std(v_para):.2f}  km/h")

# ── Colours & colourmap ──────────────────────────────────────────────────────
COL_R  = "#D97A28";  COL_RD = "#8B4010"
COL_P  = "#5B4DBF";  COL_PD = "#34278A"
CMAP   = plt.cm.RdYlBu_r
V_LO, V_HI = 4.0, 32.0   # km/h colour range

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.16, 3.1))
gs  = GridSpec(2, 3, width_ratios=[1, 1, 0.80],
               hspace=0.32, wspace=0.36,
               left=0.07, right=0.97, top=0.80, bottom=0.22)

ax_a  = fig.add_subplot(gs[:, 0])
ax_b  = fig.add_subplot(gs[:, 1])
ax_c1 = fig.add_subplot(gs[0, 2])
ax_c2 = fig.add_subplot(gs[1, 2], sharex=ax_c1)

# ── Draw speed-coloured trajectories (zorder=3) ──────────────────────────────
def draw_trajs(ax, lw=1.25, alpha=0.72):
    for ts, xs, vs in trajs:
        mask = ((ts >= t_view[0]) & (ts <= t_view[1]) &
                (xs >= x_view[0]) & (xs <= x_view[1] + 5))
        ts_, xs_, vs_ = ts[mask], xs[mask], vs[mask]
        if len(ts_) < 2:
            continue
        pts  = np.column_stack([ts_, xs_]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=CMAP,
                              norm=plt.Normalize(V_LO, V_HI),
                              linewidths=lw, alpha=alpha, zorder=3)
        lc.set_array(vs_ * 3.6)
        ax.add_collection(lc)

for ax in (ax_a, ax_b):
    draw_trajs(ax)
    ax.set_xlim(*t_view)
    ax.set_ylim(*x_view)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Space (m)")
    ax.set_yticks([100, 150, 200, 250, 300, 350])
    # Formation shock: steep solid line (fast, |W_FORM|=3)
    t1_arr = np.linspace(t_view[0], T_CLEAR, 60)
    ax.plot(t1_arr, [x_shock(t) for t in t1_arr],
            color="k", lw=0.8, alpha=0.28, ls="-", zorder=4)
    # Dissipation: shallower dashed line (slow, W_DISS=1.5)
    t2_arr = np.linspace(T_CLEAR, t_view[1], 60)
    ax.plot(t2_arr, [x_shock(t) for t in t2_arr],
            color="k", lw=1.0, alpha=0.32, ls="--", zorder=4)
    # Bottleneck marker
    ax.axhline(X_BOT, color="gray", lw=1.0, ls=":", alpha=0.42, zorder=4)

# ── Panel (a): Rectangle at high zorder ──────────────────────────────────────
rect_patch = FancyBboxPatch(
    (RECT_T0, RECT_X0), RECT_T1 - RECT_T0, RECT_X1 - RECT_X0,
    boxstyle="square,pad=0",
    fc=COL_R, alpha=0.30, ec=COL_RD, lw=2.2, zorder=15)
ax_a.add_patch(rect_patch)

ax_a.annotate(
    "High Variance\n(Non-stationary)",
    xy=((RECT_T0 + RECT_T1) / 2, RECT_X0 - 2),
    xytext=((RECT_T0 + RECT_T1) / 2 - 5, RECT_X0 - 62),
    fontsize=8.5, color=COL_RD, ha="center", fontstyle="italic",
    arrowprops=dict(arrowstyle="-|>", color=COL_RD, lw=0.9),
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.90),
    zorder=20)

ax_a.set_title("")
ax_a.text(0.5, -0.26, "(a) Rectangular Sampling",
          transform=ax_a.transAxes, ha="center", va="top",
          fontsize=10, fontweight="bold")

# ── Panel (b): Single parallelogram at high zorder ───────────────────────────
para_patch = MplPolygon(P_CONG, closed=True,
                        fc=COL_P, alpha=0.30, ec=COL_PD, lw=2.2, zorder=15)
ax_b.add_patch(para_patch)

# --- w arrow (long edge p0 → p1: wave direction) ---
mid_long       = 0.5 * (P_CONG[0] + P_CONG[1])
tang_long      = P_CONG[1] - P_CONG[0]
tang_long_unit = tang_long / np.linalg.norm(tang_long)   # ≈ (0.316, -0.949)
L_arr = 12.0

ax_b.annotate("",
    xy=mid_long + tang_long_unit * L_arr,
    xytext=mid_long - tang_long_unit * L_arr,
    arrowprops=dict(arrowstyle="-|>", color=COL_PD, lw=1.3),
    zorder=22)

# label "w" perpendicularly offset above the long edge
perp_long = np.array([tang_long_unit[1], -tang_long_unit[0]])  # 90° CW → points toward free-flow side
ax_b.text(*(mid_long + perp_long * 5 - tang_long_unit * 2.5),
          r"$w$", fontsize=12, color=COL_PD, fontweight="bold",
          ha="center", va="center", zorder=22)

# --- v* arrow (short edge p1 → p2: vehicle direction) ---
mid_short       = 0.5 * (P_CONG[1] + P_CONG[2])
tang_short      = P_CONG[2] - P_CONG[1]
tang_short_unit = tang_short / np.linalg.norm(tang_short)   # ≈ (0.448, 0.894)
S_arr = 7.0

ax_b.annotate("",
    xy=mid_short + tang_short_unit * S_arr,
    xytext=mid_short - tang_short_unit * S_arr,
    arrowprops=dict(arrowstyle="-|>", color=COL_PD, lw=1.3),
    zorder=22)

# label "v*" offset to the right of the short edge
perp_short = np.array([-tang_short_unit[1], tang_short_unit[0]])  # 90° CCW
ax_b.text(*(mid_short - perp_short * 6 + tang_short_unit * 1),
          r"$v^*$", fontsize=12, color=COL_PD, fontweight="bold",
          ha="center", va="center", zorder=22)

# "Aligned, Low Variance" annotation
p_low = P_CONG[1]   # lowest-x vertex
ax_b.annotate(
    "Aligned, Low Variance\n(Quasi-stationary)",
    xy=(p_low[0], p_low[1] + 3),
    xytext=(p_low[0] - 8, p_low[1] - 72),
    fontsize=8.5, color=COL_PD, ha="center", fontstyle="italic",
    arrowprops=dict(arrowstyle="-|>", color=COL_PD, lw=0.9),
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.90),
    zorder=20)

ax_b.set_title("")
ax_b.text(0.5, -0.26, "(b) Parallelogram Sampling",
          transform=ax_b.transAxes, ha="center", va="top",
          fontsize=10, fontweight="bold")

# ── Panel (c): Histograms ─────────────────────────────────────────────────────
bins = np.linspace(0, 34, 34)

ax_c1.hist(v_rect, bins=bins, color=COL_R, edgecolor="white",
           lw=0.25, alpha=0.85, density=True)
ax_c1.set_ylabel("Frequency", fontsize=8.5)
ax_c1.set_title("")
plt.setp(ax_c1.get_xticklabels(), visible=False)
ax_c1.text(0.5, -0.14, "(c) Rectangular Distribution",
           transform=ax_c1.transAxes, ha="center", va="top",
           fontsize=10, fontweight="bold")

ax_c2.hist(v_para, bins=bins, color=COL_P, edgecolor="white",
           lw=0.25, alpha=0.85, density=True)
ax_c2.set_xlabel(r"Velocity ($v$)", fontsize=9.5)
ax_c2.set_ylabel("Frequency", fontsize=9.5)
ax_c2.set_title("")
ax_c2.text(0.5, -0.60, "(d) Parallelogram Distribution",
           transform=ax_c2.transAxes, ha="center", va="top",
           fontsize=10, fontweight="bold")

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    target = os.path.join(FIG_DIR, f"fig1_sampling_mechanism.{ext}")
    try:
        fig.savefig(target, bbox_inches="tight", pad_inches=0.04)
    except PermissionError:
        ts_ = int(_time.time())
        target = os.path.join(FIG_DIR, f"fig1_sampling_mechanism_{ts_}.{ext}")
        fig.savefig(target, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved: {target}")
plt.close(fig)
