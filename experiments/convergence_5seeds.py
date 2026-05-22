"""
SpinFlow EM convergence across 4 datasets x 5 seeds (Appendix C figure).

Re-runs em_inverse_fd for each (dataset, seed) pair, collects the full per-iter
loss history (total + flow + physics + cross-entropy + smoothness), pads
trajectories that early-stop to a common length, and plots seed-mean
total/data loss trajectories.

Layout uses a 2x2 panel to improve readability in single-column appendix mode.

Outputs overwrite:
  Figures/fig5_convergence.{pdf,png}
Cache (so style edits don't trigger re-runs):
  convergence_5seeds_cache.npz

Usage:
  python convergence_5seeds.py            # run if cache missing, else plot
  python convergence_5seeds.py --rerun    # force re-run all (dataset, seed) pairs
  python convergence_5seeds.py --plot-only  # plot from cache only
"""
import os
import sys
import argparse
import io
import contextlib
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'spin_inversion'))
FIG_DIR    = os.path.join(SCRIPT_DIR, '..', 'Figures')
CACHE_FILE = os.path.join(SCRIPT_DIR, 'convergence_5seeds_cache.npz')
EXP_CSV    = os.path.join(SCRIPT_DIR, '..', 'Figures', 'experiment_results.csv')
os.makedirs(FIG_DIR, exist_ok=True)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--rerun',     action='store_true')
parser.add_argument('--plot-only', action='store_true')
args, _ = parser.parse_known_args()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 12, "axes.labelsize": 12.5, "axes.titlesize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10.5, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.5,
})

# Order matches Figs.~2--5 in root.tex so multi-figure visual flow is consistent.
DS_KEYS  = ['YTDJ', 'RML', 'HighD', 'NGSIM-I80']
DS_LABEL = {'YTDJ':      '(a) YTDJ (Urban Tunnel)',
            'RML':       '(b) RML (On-ramp)',
            'HighD':     '(c) HighD (Freeway)',
            'NGSIM-I80': '(d) NGSIM I-80'}
DATASETS = {
    'YTDJ':      os.path.join(SCRIPT_DIR, '..', '..', 'results', 'YTDJ',      'YTDJ_inverse_init.npz'),
    'RML':       os.path.join(SCRIPT_DIR, '..', '..', 'results', 'RML',       'RML_inverse_init.npz'),
    'HighD':     os.path.join(SCRIPT_DIR, '..', '..', 'results', 'HighD',     'HighD_inverse_init.npz'),
    'NGSIM-I80': os.path.join(SCRIPT_DIR, '..', '..', 'results', 'NGSIM-I80', 'NGSIM-I80_inverse_init.npz'),
}

# Same seeds as run_experiments.py to keep cross-table reporting comparable.
SEEDS    = [42, 137, 256, 314, 628]
EM_ITERS = 80


def _load_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    from fd_model import TriangularFD
    proto_data = data['prototypes']
    prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0'])
                  for p in proto_data]
    meta      = data['meta'].item()
    fd_points = data['fd_points'].item()
    return dict(
        fd_points = fd_points,
        prototypes = prototypes,
        dx       = meta['dx'],
        cells    = meta['cells'],
        rho_obs  = data['rho0'],
        rho_start = data['rho_start'] if 'rho_start' in data else None,
        rho_end   = data['rho_end']   if 'rho_end'   in data else None,
        dt_window = meta.get('T_sample', None),
    )


def _run_one(ds, seed):
    """Run SpinFlow EM for one (dataset, seed); return the full history dict."""
    from solver import em_inverse_fd
    np.random.seed(seed)
    rho_obs = ds['rho_obs']
    cells   = ds['cells']
    v_obs   = np.zeros(cells)
    mask    = rho_obs > 1e-6
    for g in range(len(ds['prototypes'])):
        q_g = ds['prototypes'][g].flow(rho_obs)
        v_obs[mask] += q_g[mask] / rho_obs[mask] / len(ds['prototypes'])
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, _, _, history = em_inverse_fd(
            ds['fd_points'], ds['dx'], M=cells,
            rho_obs=rho_obs, v_obs=v_obs,
            rho_start=ds['rho_start'], rho_end=ds['rho_end'],
            dt_window=ds['dt_window'],
            n_prototypes=3, em_iters=EM_ITERS, inner_iters=20,
            lam_fd_q=1.0, lam_phys=0.1, lam_smo=0.02,
            convergence_tol=5e-4, learning_rate=0.05, seed=seed,
        )
    return history


def _pad_with_tail(curve, target_len):
    """Right-pad a 1D curve to ``target_len`` by repeating its last value.

    Physical meaning: when EM early-stops, the loss has reached a stationary
    plateau by construction, so extending the trajectory at its plateau value
    is consistent with the augmented free-energy landscape. This produces a
    well-defined mean +/- std envelope across heterogeneous run lengths.
    """
    arr = np.asarray(curve, dtype=float)
    if len(arr) >= target_len:
        return arr[:target_len]
    tail = np.full(target_len - len(arr), arr[-1])
    return np.concatenate([arr, tail])


def _load_spinflow_rt_from_csv(csv_path):
    """Load per-dataset SpinFlow runtime samples from experiment_results.csv.

    Returns a dict:
      {dataset_key: [runtime_s_seed1, runtime_s_seed2, ...]}
    Missing datasets are returned as NaN placeholders.
    """
    rt = {k: [] for k in DS_KEYS}
    if not os.path.exists(csv_path):
        return {k: [np.nan] * len(SEEDS) for k in DS_KEYS}

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get('dataset', '')
            mk = row.get('model_key', '')
            model = row.get('model', '')
            if ds in rt and (mk == 'spinflow' or model == 'SpinFlow'):
                try:
                    rt_val = float(row['runtime_s'])
                except Exception:
                    continue
                rt[ds].append(rt_val)

    for k in DS_KEYS:
        if len(rt[k]) == 0:
            rt[k] = [np.nan] * len(SEEDS)
    return rt


# ── Sweep (skipped when cache present unless --rerun) ─────────────────────────
need_run = (not args.plot_only) and (args.rerun or not os.path.exists(CACHE_FILE))

if need_run:
    print("Loading datasets ...")
    DS = {k: _load_dataset(v) for k, v in DATASETS.items()}

    histories = {k: [] for k in DS_KEYS}
    conv_iters = {k: [] for k in DS_KEYS}

    for k in DS_KEYS:
        ds = DS[k]
        print(f"\n[{k}] running 5 seeds ...")
        for s in SEEDS:
            h = _run_one(ds, s)
            histories[k].append(h)
            n_iter = len(h['loss'])
            conv_iters[k].append(n_iter)
            print(f"  seed={s}  conv_iter={n_iter}  "
                  f"loss[0]={h['loss'][0]:.4e}  loss[-1]={h['loss'][-1]:.4e}")

    # Pad and stack: shape (4 datasets, 5 seeds, EM_ITERS).
    keys_loss = ['loss', 'loss_fd_q', 'loss_phys', 'loss_pi', 'loss_smooth']
    stacks = {kl: {} for kl in keys_loss}
    for k in DS_KEYS:
        for kl in keys_loss:
            rows = [_pad_with_tail(h[kl], EM_ITERS) for h in histories[k]]
            stacks[kl][k] = np.stack(rows, axis=0)   # (5, EM_ITERS)

    np.savez(
        CACHE_FILE,
        seeds=SEEDS,
        ds_keys=DS_KEYS,
        em_iters=EM_ITERS,
        **{f'conv_iter_{k}': np.array(conv_iters[k]) for k in DS_KEYS},
        **{f'{kl}_{k}': stacks[kl][k]
           for kl in keys_loss for k in DS_KEYS},
    )
    print(f"\nCache saved -> {CACHE_FILE}")

else:
    print(f"Loading cached histories from {CACHE_FILE}")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    EM_ITERS = int(cache['em_iters'])
    stacks = {}
    for kl in ['loss', 'loss_fd_q', 'loss_phys', 'loss_pi', 'loss_smooth']:
        stacks[kl] = {k: cache[f'{kl}_{k}'] for k in DS_KEYS}
    conv_iters = {k: cache[f'conv_iter_{k}'].tolist() for k in DS_KEYS}

# Runtime annotation in figure must use the same source as Table III.
rt_seconds = _load_spinflow_rt_from_csv(EXP_CSV)


# ── Plot (2 x 2) for single-column appendix readability ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.78))
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.96, wspace=0.23, hspace=0.54)
axes = axes.flatten()

iters_x = np.arange(1, EM_ITERS + 1)
eps = 1e-9

for idx, k in enumerate(DS_KEYS):
    ax = axes[idx]

    # Mean trajectories across seeds for representative loss components.
    L_total = stacks['loss'][k]                # (5, EM_ITERS)
    L_flow  = stacks['loss_fd_q'][k]
    m_tot = L_total.mean(axis=0)
    m_q   = L_flow.mean(axis=0)
    # Sparse markers every 5 iterations to avoid over-plotting.
    ax.plot(iters_x, np.clip(m_tot, eps, None), '-', lw=2.05, color='#7B2D8E',
            marker='o', ms=2.2, markevery=5,
            label='Total loss', zorder=4)
    ax.plot(iters_x, np.clip(m_q, eps, None), '--', lw=1.4, color='#2166AC',
            marker='s', ms=2.0, markevery=5,
            alpha=0.95, label='Data loss', zorder=3)

    # Inset: zoom into the first 20 EM iterations.
    # Parent axis is wider than tall; use narrower width to render a visual square inset.
    axins = ax.inset_axes([0.18, 0.34, 0.30, 0.34])  # more square, slightly left
    x_zoom = iters_x[:20]
    axins.plot(x_zoom, np.clip(m_tot[:20], eps, None), '-', lw=1.2, color='#7B2D8E')
    axins.plot(x_zoom, np.clip(m_q[:20], eps, None), '--', lw=1.0, color='#2166AC', alpha=0.95)
    axins.set_xlim(1, 20)
    y_zoom_min = float(min(np.nanmin(m_tot[:20]), np.nanmin(m_q[:20])))
    y_zoom_max = float(max(np.nanmax(m_tot[:20]), np.nanmax(m_q[:20])))
    pad = max((y_zoom_max - y_zoom_min) * 0.10, 1e-9)
    axins.set_ylim(y_zoom_min - pad, y_zoom_max + pad)
    axins.set_xticks([1, 10, 20])
    axins.tick_params(axis='both', labelsize=6.8, pad=1, length=0)
    axins.grid(False)
    axins.set_facecolor((1, 1, 1, 0.92))
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)

    # Dashed zoom connectors: inset UL/LR corners -> zoom boundary points.
    con_ul = ConnectionPatch(
        xyA=(0.0, 1.0), coordsA=axins.transAxes,
        xyB=(1.0, y_zoom_max), coordsB=ax.transData,
        color='#666666', lw=0.7, ls='--', alpha=0.65
    )
    con_lr = ConnectionPatch(
        xyA=(1.0, 0.0), coordsA=axins.transAxes,
        xyB=(20.0, y_zoom_min), coordsB=ax.transData,
        color='#666666', lw=0.7, ls='--', alpha=0.65
    )
    ax.add_artist(con_ul)
    ax.add_artist(con_lr)

    # Panel (a): concise inset annotation.
    if idx == 0:
        axins.text(0.98, 0.93, 'first 20 iters',
                   transform=axins.transAxes, ha='right', va='top',
                   fontsize=6.9, color='#444')

    # Mark mean convergence iteration with a thin dashed vertical.
    conv_arr = np.asarray(conv_iters[k], dtype=float)
    t_conv_m = conv_arr.mean()
    t_conv_s = conv_arr.std()
    rt_arr = np.asarray(rt_seconds[k], dtype=float)
    rt_m = np.nanmean(rt_arr)
    rt_s = np.nanstd(rt_arr)
    ax.axvline(t_conv_m, color='#444', lw=0.8, ls=':', alpha=0.75, zorder=4)

    ax.set_xlabel('EM iteration', fontsize=12)
    ax.set_ylabel('Loss' if idx in (0, 2) else '', fontsize=12)
    ax.set_xlim(1, EM_ITERS)
    if idx == 1:  # panel (b): keep a single shared legend
        ax.legend(fontsize=10.2, loc='upper right', framealpha=0.92, edgecolor='none')

    # Place convergence annotation in axes coordinates so it stays inside panel.
    # and never collides with the legend (top-right) or the loss curve (top-left).
    tx, ty = (0.97, 0.52) if idx == 1 else (0.97, 0.90)
    va = 'center' if idx == 1 else 'top'
    rt_line = f"RT={rt_m:.2f}$\\pm${rt_s:.2f}s" if np.isfinite(rt_m) else "RT=N/A"
    ax.text(tx, ty,
            rf"$\hat{{t}}_{{\mathrm{{conv}}}}={t_conv_m:.0f}\pm{t_conv_s:.0f}$" + "\n" + rt_line,
            transform=ax.transAxes, ha='right', va=va,
            fontsize=9.6, color='#333')
    ax.grid(True, alpha=0.22, lw=0.42)
    ax.text(0.5, -0.40, DS_LABEL[k], transform=ax.transAxes,
            ha='center', fontsize=12.5, fontweight='bold')

for ext in ('pdf', 'png'):
    path = os.path.join(FIG_DIR, f'fig5_convergence.{ext}')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {path}")
plt.close(fig)

# Console summary for the appendix paragraph.
print("\n[Summary] Convergence stats (per dataset, across 5 seeds):")
for k in DS_KEYS:
    arr = np.asarray(conv_iters[k], dtype=float)
    L_total = stacks['loss'][k]
    red = 1.0 - L_total[:, -1].mean() / L_total[:, 0].mean()
    print(f"  {k:10s}  t_hat_conv = {arr.mean():.1f} +/- {arr.std():.1f} iters  "
          f"loss reduction = {red*100:.1f}%")
print("[Done] fig5_convergence (5 seeds) complete.")
