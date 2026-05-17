"""
Generate publication-quality figures for ITSS paper from SpinFlow inversion results.

Datasets: YTDJ, RML, HighD, NGSIM-I80 — all real, 4 columns.

Fig 2: FD Points + Prototypes         (1×4, figure*)
Fig 3: Forward Consistency Scatters   (2×4, figure*) + inset error histograms
Fig 4: Phase Distribution π_g(x)     (1×4, figure*)
Fig 5: Convergence Curves             (1×4, figure*)

All subfigure labels placed BELOW the axes.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from paths import FIG_DIR, RESULTS_DIR, SPIN_SRC

sys.path.insert(0, SPIN_SRC)
from spinflow.fd_model import TriangularFD, PHASES, PHASE_DISPLAY, PHASE_COLORS

os.makedirs(FIG_DIR, exist_ok=True)

# Matplotlib defaults (publication DPI and typography).
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.4,
})

PROTO_COLORS = ['#2166AC', '#4DAF4A', '#D6604D']
PHASE_COLS   = {'free': '#2166AC', 'critical': '#4DAF4A', 'congested': '#D6604D'}
# Fig.~6 top row (π_g): intuitive F=blue, S=green, J=red — clear on screen & print
FIG6_TOP_LINE = {
    'free': '#1b6aae',       # free flow
    'critical': '#2a8f52',  # synchronized / coexistence
    'congested': '#c4333c',  # wide-moving jam
}
FIG6_TOP_BG = {
    'free': '#e5f0fb',
    'critical': '#e8f5eb',
    'congested': '#fdecea',
}
# Fig.~6 PED row: accent orange aligned with Fig.~3 flow-error inset (#D97A28 family)
_FIG3_ACCENT_ORANGE = np.array([0.851, 0.478, 0.157])  # #D97A28

# 4 columns: all real datasets
REAL_DATASETS = {
    'YTDJ':     {'npz': os.path.join(RESULTS_DIR, 'YTDJ', 'YTDJ_inverse_init.npz'),         'label': 'YTDJ'},
    'RML':      {'npz': os.path.join(RESULTS_DIR, 'RML', 'RML_inverse_init.npz'),             'label': 'RML'},
    'HighD':    {'npz': os.path.join(RESULTS_DIR, 'HighD', 'HighD_inverse_init.npz'),         'label': 'HighD'},
    'NGSIM':    {'npz': os.path.join(RESULTS_DIR, 'NGSIM-I80', 'NGSIM-I80_inverse_init.npz'), 'label': 'NGSIM I-80'},
}
PLACEHOLDER_DATASETS = {}
COL_ORDER  = ['YTDJ', 'RML', 'HighD', 'NGSIM']
COL_LABELS = {'YTDJ': 'YTDJ (Urban Tunnel)', 'RML': 'RML (On-ramp)',
              'HighD': 'HighD (Freeway)',    'NGSIM': 'NGSIM I-80'}


def load_dataset(key):
    cfg = REAL_DATASETS[key]
    npz_path = cfg['npz']
    data = np.load(npz_path, allow_pickle=True)

    pi_final   = data['pi_final']
    proto_data = data['prototypes']
    fd_points  = data['fd_points'].item()
    history    = data['history'].item()
    meta       = data['meta'].item()

    prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) for p in proto_data]
    dx    = meta['dx'];  cells = meta['cells']
    x_pos = np.arange(cells) * dx

    k_pts = fd_points['k'] / 1000.0
    q_pts = fd_points['q'] / 3600.0
    v_pts = fd_points['v'] / 3.6
    x_pts = fd_points['x_center']
    n_pts = len(k_pts)

    q_pred, v_pred = [], []
    for i in range(n_pts):
        cell = int(np.clip(x_pts[i] / dx, 0, cells - 1))
        qp = sum(pi_final[cell, g] * prototypes[g].flow(np.array([k_pts[i]]))[0]
                 for g in range(len(prototypes)))
        q_pred.append(qp)
        v_pred.append(qp / k_pts[i] if k_pts[i] > 0 else 0.0)

    q_pred = np.array(q_pred);  v_pred = np.array(v_pred)
    q_err  = (q_pred - q_pts) * 3600
    v_err  = (v_pred - v_pts) * 3.6
    rmse_q = np.sqrt(np.mean(q_err**2));  rmse_v = np.sqrt(np.mean(v_err**2))
    mae_q  = np.mean(np.abs(q_err));       mae_v  = np.mean(np.abs(v_err))
    r2_q = 1 - np.sum(q_err**2) / (np.sum((q_pts*3600 - (q_pts*3600).mean())**2) + 1e-12)
    r2_v = 1 - np.sum(v_err**2) / (np.sum((v_pts*3.6 - (v_pts*3.6).mean())**2) + 1e-12)
    eps = 1e-10
    H_x = -np.sum(np.clip(pi_final, eps, 1.0) * np.log(np.clip(pi_final, eps, 1.0)), axis=1)

    return {
        'key': key, 'label': COL_LABELS[key], 'real': True,
        'prototypes': prototypes, 'pi_final': pi_final,
        'x_pos': x_pos, 'cells': cells, 'dx': dx,
        'fd_points': fd_points, 'n_pts': n_pts,
        'k_pts': k_pts, 'q_pts': q_pts, 'v_pts': v_pts, 'x_pts': x_pts,
        'q_pred': q_pred, 'v_pred': v_pred,
        'q_err': q_err, 'v_err': v_err,
        'sx0': data['sx0'], 'sy0': data['sy0'], 'sz0': data['sz0'],
        'rmse_q': rmse_q, 'rmse_v': rmse_v,
        'mae_q': mae_q, 'mae_v': mae_v,
        'r2_q': r2_q, 'r2_v': r2_v,
        'history': history, 'H_x': H_x,
    }


print("Loading datasets...")
DS = {}
for k in COL_ORDER:
    DS[k] = load_dataset(k)

print("\n% Table II / forward metrics data:")
for k in COL_ORDER:
    d = DS[k]
    print(f"  {d['label']:25s}  N={d['n_pts']}  RMSE_q={d['rmse_q']:.1f}  R2_q={d['r2_q']:.4f}"
          f"  RMSE_v={d['rmse_v']:.2f}  R2_v={d['r2_v']:.4f}")


def draw_placeholder(ax, msg="Pending\n(experiment not run)"):
    ax.set_facecolor('#F5F5F5')
    for spine in ax.spines.values():
        spine.set_edgecolor('#BBBBBB')
        spine.set_linestyle('--')
        spine.set_visible(True)
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha='center', va='center', fontsize=7, color='#999999',
            fontstyle='italic')
    ax.set_xticks([]); ax.set_yticks([])


def label_below(ax, text, fontsize=8.5, y=-0.28):
    ax.set_title('')
    ax.text(0.5, y, text, transform=ax.transAxes,
            ha='center', va='top', fontsize=fontsize, fontweight='bold')


def metrics_upper_left(ax, text, fontsize=6.0, zorder=6):
    """RMSE / R^2 box in subplot upper-left (data coordinates clutter-safe)."""
    ax.text(0.03, 0.97, text, transform=ax.transAxes, zorder=zorder,
            ha='left', va='top', fontsize=fontsize, linespacing=1.15,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white', alpha=0.92,
                      edgecolor='0.75', linewidth=0.35))


# Fig.~3 flow-error inset layout (fractions of parent axes width/height).
INSET_WIDTH_FRAC  = 0.4
INSET_HEIGHT_FRAC = 0.2
INSET_X0_FRAC     = 0.60
INSET_Y0_FRAC     = 0.175

def add_error_inset(parent_ax, errors, color, xlabel, zorder=10):
    """Histogram inset for flow or speed errors; geometry from INSET_* above."""
    fig = parent_ax.figure
    ax_pos = parent_ax.get_position()
    w_frac = ax_pos.width  * INSET_WIDTH_FRAC
    h_frac = ax_pos.height * INSET_HEIGHT_FRAC
    x0 = ax_pos.x0 + ax_pos.width  * INSET_X0_FRAC
    y0 = ax_pos.y0 + ax_pos.height * INSET_Y0_FRAC
    ins = fig.add_axes([x0, y0, w_frac, h_frac], zorder=zorder)
    ins.hist(errors, bins=18, color=color, alpha=0.82, edgecolor='white', linewidth=0.2,
             density=True)
    ins.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6)
    ins.axvline(errors.mean(), color='#D6604D', lw=0.8, ls='--', alpha=0.85)
    ins.set_xlabel(xlabel, fontsize=5.5, labelpad=1)
    ins.tick_params(axis='both', labelsize=5, pad=1, length=2)
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_linewidth(0.4)
    ins.patch.set_facecolor('white')
    ins.patch.set_alpha(0.92)
    return ins


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2: FD Points + Prototypes  (1×4)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 4, figsize=(9, 1.8))
fig2.subplots_adjust(left=0.06, right=0.98, bottom=0.22, top=0.93, wspace=0.33)

for idx, key in enumerate(COL_ORDER):
    ax = axes2[idx]
    d  = DS[key]
    fp = d['fd_points']
    ax.scatter(fp['k'], fp['q'], c=fp['score'], s=14, alpha=0.7,
               cmap='viridis_r', edgecolors='k', linewidths=0.25, zorder=3)
    rho_range = np.linspace(0, 0.35, 300)
    for g, proto in enumerate(d['prototypes']):
        ax.plot(rho_range*1000, proto.flow(rho_range)*3600,
                color=PROTO_COLORS[g], lw=1.6, ls='--',
                label=f'P{g}: $v_f$={proto.vf*3.6:.0f}', alpha=0.85, zorder=4)
    ax.set_xlabel('Density (veh/km)')
    ax.set_ylabel('Flow (veh/h)' if idx == 0 else '')
    ax.set_xlim(0, min(fp['k'].max()*1.15, 360))
    ax.set_ylim(0, min(fp['q'].max()*1.15, 2600))
    ax.legend(fontsize=5.5, loc='upper right', framealpha=0.85, edgecolor='none')
    ax.grid(True, alpha=0.15, lw=0.4)
    label_below(ax, f'({chr(97+idx)}) {d["label"]}', y=-0.32)

for ext in ('pdf', 'png'):
    fig2.savefig(os.path.join(FIG_DIR, f'fig2_fd_prototypes.{ext}'),
                 bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: fig2_fd_prototypes.{ext}")
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Forward Consistency (2×4) — all-q top row, all-v bottom row
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 4, figsize=(9, 3.7))
fig3.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.95, wspace=0.30, hspace=0.50)

for idx, key in enumerate(COL_ORDER):
    d = DS[key]
    ax_q = axes3[0, idx]
    ax_v = axes3[1, idx]

    # Flow (q) scatter
    ax_q.scatter(d['q_pts']*3600, d['q_pred']*3600, c=d['x_pts'], s=12, alpha=0.60,
                 cmap='viridis', edgecolors='k', linewidths=0.2, zorder=3)
    lim_q = max((d['q_pts']*3600).max(), (d['q_pred']*3600).max()) * 1.08
    ax_q.plot([0, lim_q], [0, lim_q], 'r--', lw=0.9, alpha=0.65)
    ax_q.set_xlim(0, lim_q); ax_q.set_ylim(0, lim_q)
    ax_q.set_xlabel('Observed $q$ (veh/h)')
    ax_q.set_ylabel('Predicted $q$ (veh/h)' if idx == 0 else '')
    ax_q.grid(True, alpha=0.12, lw=0.35)
    add_error_inset(ax_q, d['q_err'], '#D97A28', 'Error (veh/h)')
    metrics_upper_left(
        ax_q,
        f'RMSE = {d["rmse_q"]:.0f} veh/h\n$R^2$ = {d["r2_q"]:.3f}')
    label_below(ax_q, f'({chr(97+idx)}) {d["label"]}', fontsize=8.0, y=-0.35)

    # Velocity scatter
    ax_v.scatter(d['v_pts']*3.6, d['v_pred']*3.6, c=d['x_pts'], s=12, alpha=0.60,
                 cmap='viridis', edgecolors='k', linewidths=0.2, zorder=3)
    lim_v = max((d['v_pts']*3.6).max(), (d['v_pred']*3.6).max()) * 1.08
    ax_v.plot([0, lim_v], [0, lim_v], 'r--', lw=0.9, alpha=0.65)
    ax_v.set_xlim(0, lim_v); ax_v.set_ylim(0, lim_v)
    ax_v.set_xlabel('Observed $v$ (km/h)')
    ax_v.set_ylabel('Predicted $v$ (km/h)' if idx == 0 else '')
    ax_v.grid(True, alpha=0.12, lw=0.35)
    add_error_inset(ax_v, d['v_err'], '#5B4DBF', 'Error (km/h)')
    metrics_upper_left(
        ax_v,
        f'RMSE = {d["rmse_v"]:.2f} km/h\n$R^2$ = {d["r2_v"]:.3f}')
    label_below(ax_v, f'({chr(101+idx)}) {d["label"]}', fontsize=8.0, y=-0.35)

for ext in ('pdf', 'png'):
    fig3.savefig(os.path.join(FIG_DIR, f'fig3_forward_consistency.{ext}'),
                 bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: fig3_forward_consistency.{ext}")
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers shared by Fig 4 and Table II
# ══════════════════════════════════════════════════════════════════════════════

def draw_dominant_bg(ax, x_pos, pi_final, dx):
    """Shade background by dominant phase — light blue / green / red tints (traffic semantics)."""
    dominant = np.argmax(pi_final, axis=1)
    n = len(dominant)
    i = 0
    while i < n:
        g = dominant[i]
        j = i + 1
        while j < n and dominant[j] == g:
            j += 1
        ax.axvspan(x_pos[i], x_pos[j-1] + dx, color=FIG6_TOP_BG[PHASES[g]], alpha=1.0, zorder=0, lw=0)
        i = j


def compute_ped(d, bandwidth_cells=4):
    """
    Compute PED(x) = exp(-ΔF(x)) where ΔF is the local normalised
    squared FD residual at each spatial grid cell.

    Physics: PED → 1 in equilibrium zones (model fits data well);
             PED → 0 at phase-transition nucleation sites.

    For each cell i we collect FD points within ±bandwidth, compute
        q_pred(j) = Σ_g  π_g(x_i) · q_g(ρ_j)
    and
        ΔF(x_i) = mean[(q_pred - q_obs)²] / σ_q²
    so PED(x_i) = exp(−ΔF(x_i)).
    """
    pi       = d['pi_final']
    x_pos    = d['x_pos']
    dx       = d['dx']
    bw       = bandwidth_cells * dx

    k_pts    = d['k_pts']                  # veh/m
    q_obs_vh = d['q_pts'] * 3600           # veh/h (observed)
    x_pts    = d['x_pts']                  # m (position of each FD point)
    protos   = d['prototypes']

    # Global residual std (floor at 50 veh/h to avoid divide-by-zero)
    sigma_q  = max(float(np.std(d['q_err'])), 50.0)

    n_cells  = len(x_pos)
    PED      = np.ones(n_cells)

    # Pre-compute q_g(ρ_p) for all FD points and all prototypes [3 × n_pts]
    q_g_all  = np.array([proto.flow(k_pts) * 3600 for proto in protos])  # veh/h

    for i in range(n_cells):
        mask = np.abs(x_pts - x_pos[i]) <= bw
        if mask.sum() < 2:
            continue
        # Predicted flow: π_g(x_i) · q_g(ρ_p)
        q_pred = np.dot(pi[i], q_g_all[:, mask])      # [n_pts_local] veh/h
        res    = q_pred - q_obs_vh[mask]
        delta_F = float(np.mean(res**2)) / sigma_q**2
        PED[i]  = np.exp(-delta_F)

    return PED


def compute_all_diagnostics(d):
    """Compute all interpretability signals and return a summary dict."""
    pi  = d['pi_final']
    sz  = d['sz0']
    sy  = d['sy0']
    x   = d['x_pos']
    eps = 1e-10

    H_x  = -np.sum(pi * np.log(np.clip(pi, eps, 1.0)), axis=1)
    D_x  = pi.max(axis=1)
    diff = pi[1:] - pi[:-1]
    G_pi = np.concatenate([[0.0], np.sum(diff**2, axis=1)])
    # smooth G_pi with 3-cell running average to reduce noise
    G_pi_s = np.convolve(G_pi, np.ones(3)/3, mode='same')

    tau_H = np.log(2)   # 2-phase coexistence threshold

    # PED from FD reconstruction
    PED = compute_ped(d)

    # Combine detection score: spatial instability × non-equilibrium
    # High G_pi_s and low PED → transition nucleation zone
    det_score = G_pi_s * (1.0 - PED)

    # Ω_Q: cells where H ≥ τ_H (high entropy coexistence zone)
    omega_Q = H_x >= tau_H

    # x* = argmin PED within Ω_Q (primary bottleneck site: strongest non-equilibrium)
    # Exclude boundary 12% to avoid edge artefacts
    n_cells = len(x)
    margin  = max(3, int(0.12 * n_cells))
    interior = np.zeros(n_cells, dtype=bool)
    interior[margin:-margin] = True

    cand = np.where(omega_Q & interior)[0]
    if len(cand) == 0:
        cand = np.where(interior)[0]
    x_star_idx = int(cand[np.argmin(PED[cand])])
    x_star = x[x_star_idx]

    # Stable zone: PED percentile > 60%  (well-equilibrated cells)
    ped_thresh = np.percentile(PED, 40)
    stable     = PED >= ped_thresh
    ped_stable = float(np.mean(PED[stable]))
    ped_min    = float(PED.min())
    delta_ped  = (ped_stable - ped_min) / ped_stable * 100.0   # % drop

    return {
        'H_x': H_x, 'D_x': D_x, 'G_pi': G_pi_s, 'PED': PED,
        'det_score': det_score, 'tau_H': tau_H, 'omega_Q': omega_Q,
        'x_star': x_star, 'x_star_idx': x_star_idx,
        'ped_min': ped_min, 'ped_stable': ped_stable, 'delta_ped': delta_ped,
        'H_at_xstar': float(H_x[x_star_idx]),
        'Gpi_max': float(G_pi_s.max()),
        'H_mean': float(H_x.mean()), 'H_std': float(H_x.std()),
    }


# Diagnostics (PED, phase dominance) for all datasets
print("\nComputing PED and transition diagnostics...")
DIAG = {}
for key in COL_ORDER:
    DIAG[key] = compute_all_diagnostics(DS[key])
    d = DIAG[key]
    print(f"  [{key}]  x*={d['x_star']:.1f}m  PED_min={d['ped_min']:.3f}"
          f"  PED_stable={d['ped_stable']:.3f}  ΔPED={d['delta_ped']:.1f}%"
          f"  H(x*)={d['H_at_xstar']:.3f}  G_π_max={d['Gpi_max']:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4: Phase Distribution (row 0) + PED-based Transition Detection (row 1)
# ══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 4, figsize=(9, 3.2))
fig4.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.96,
                     wspace=0.28, hspace=0.10)

for idx, key in enumerate(COL_ORDER):
    ax0 = axes4[0, idx]
    ax1 = axes4[1, idx]
    d   = DS[key]

    x  = d['x_pos']
    dg = DIAG[key]

    # Row 0: pi_g(x) with dominant-phase background
    draw_dominant_bg(ax0, x, d['pi_final'], d['dx'])
    for g, phase in enumerate(PHASES):
        ax0.plot(x, d['pi_final'][:, g],
                 label=rf"$\pi_{{\mathrm{{{PHASE_DISPLAY[phase]}}}}}$",
                 lw=1.65, alpha=0.92, color=FIG6_TOP_LINE[phase], zorder=3)
    ax0.set_ylabel(r'$\pi_g(x)$' if idx == 0 else '')
    ax0.set_xlim(0, x.max()); ax0.set_ylim(0, 1.04)
    ax0.set_xticks([])
    ax0.legend(fontsize=5.7, loc='upper right', framealpha=0.92, edgecolor='none',
               handlelength=1.2)
    ax0.grid(True, alpha=0.12, lw=0.35, zorder=1)

    # Row 1: PED(x); shade low-PED; mark primary bottleneck x*
    PED    = dg['PED']
    x_star = dg['x_star']

    ped_thr = dg['ped_stable'] * 0.70
    low_ped = PED < ped_thr
    i = 0
    while i < len(low_ped):
        if low_ped[i]:
            j = i + 1
            while j < len(low_ped) and low_ped[j]:
                j += 1
            ax1.axvspan(x[i], x[min(j, len(x)-1)],
                        color=(*_FIG3_ACCENT_ORANGE, 0.18), zorder=0, lw=0)
            i = j
        else:
            i += 1

    ax1.plot(x, PED, color=plt.cm.viridis(0.36), lw=1.65, label='PED$(x)$', zorder=3)
    ax1.axhline(dg['ped_stable'], color='#5f6b7a', lw=0.95, ls='--', alpha=0.66,
                label='Stable baseline')
    ax1.axvline(x_star, color=(*_FIG3_ACCENT_ORANGE, 1.0), lw=1.45, ls='-', alpha=0.88, zorder=5,
                label=r'$x^*$ (bottleneck)')
    offset = x.max() * 0.02
    ax1.text(x_star + offset, 0.92,
             rf"$x^*\!=\!{x_star:.0f}$m", fontsize=5.8,
             color=tuple(np.clip(_FIG3_ACCENT_ORANGE * 0.72, 0.0, 1.0)), va='top')

    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('PED$(x)$' if idx == 0 else '')
    ax1.set_xlim(0, x.max()); ax1.set_ylim(0, 1.06)
    ax1.legend(fontsize=5.5, framealpha=0.92, edgecolor='none',
               handlelength=1.2)
    ax1.grid(True, alpha=0.12, lw=0.35, zorder=1)

    label_below(ax1, f'({chr(97+idx)}) {d["label"]}', y=-0.32)

for ext in ('pdf', 'png'):
    fig4.savefig(os.path.join(FIG_DIR, f'fig4_phase_distribution.{ext}'),
                 bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: fig4_phase_distribution.{ext}")
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5: Convergence  (1×4)
# ══════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(1, 4, figsize=(9, 1.8))
fig5.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.93, wspace=0.32)

for idx, key in enumerate(COL_ORDER):
    ax = axes5[idx]
    d  = DS[key]
    h = d['history']
    iters = np.arange(1, len(h['loss']) + 1)
    ax.plot(iters, h['loss'], 'o-', ms=2.5, lw=1.4, color='#7B2D8E', label='Total')
    ax.plot(iters, h['loss_fd_q'], 's--', ms=1.8, lw=1.1, color='#2166AC',
            alpha=0.85, label='Flow')
    ax.set_xlabel('EM Iteration')
    ax.set_ylabel('Loss' if idx == 0 else '')
    ax.set_yscale('log')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.85, edgecolor='none')
    ax.grid(True, alpha=0.15, lw=0.4)
    label_below(ax, f'({chr(97+idx)}) {d["label"]}')

for ext in ('pdf', 'png'):
    fig5.savefig(os.path.join(FIG_DIR, f'fig5_convergence.{ext}'),
                 bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: fig5_convergence.{ext}")
plt.close(fig5)

print("\n[Done] All paper figures generated.")
