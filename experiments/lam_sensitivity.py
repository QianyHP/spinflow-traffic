"""
SpinFlow appendix sensitivity analysis figure generator.

Current layout:
- 3x3 panels (lambda_smooth / lambda_phys / FD-point fraction) x
  (RMSE_q / Phys.Res. / |Delta T.MAE|).
- Curves are 5-seed mean trajectories with confidence bands.
- Red stars mark the panel-specific default settings:
  lambda_smooth = 0.02, lambda_phys = 0.10, FD fraction = 100%.

Outputs:
  Figures/fig6_sensitivity.{pdf,png}
Cache:
  fig6_sensitivity_cache.npz
"""

import os
import sys
import io
import argparse
import contextlib

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'spin_inversion'))
FIG_DIR    = os.path.join(SCRIPT_DIR, '..', 'Figures')
CACHE_FILE = os.path.join(SCRIPT_DIR, 'fig6_sensitivity_cache.npz')
os.makedirs(FIG_DIR, exist_ok=True)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--plot-only', action='store_true')
args, _ = parser.parse_known_args()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.5,
})

DS_KEYS = ['YTDJ', 'RML', 'HighD', 'NGSIM-I80']
DATASETS = {
    'YTDJ':      os.path.join(SCRIPT_DIR, '..', '..', 'results', 'YTDJ',      'YTDJ_inverse_init.npz'),
    'RML':       os.path.join(SCRIPT_DIR, '..', '..', 'results', 'RML',       'RML_inverse_init.npz'),
    'HighD':     os.path.join(SCRIPT_DIR, '..', '..', 'results', 'HighD',     'HighD_inverse_init.npz'),
    'NGSIM-I80': os.path.join(SCRIPT_DIR, '..', '..', 'results', 'NGSIM-I80', 'NGSIM-I80_inverse_init.npz'),
}
X_GT = {'YTDJ': 66.0, 'RML': 192.0, 'HighD': 87.9, 'NGSIM-I80': 392.0}
# Harmonized with the paper's Fig.3--6 palette style (blue/green/red + purple accent).
DS_COLORS  = {'YTDJ': '#2166AC', 'RML': '#4DAF4A', 'HighD': '#D6604D', 'NGSIM-I80': '#7B2D8E'}
DS_MARKERS = {'YTDJ': 'o',       'RML': 's',       'HighD': '^',       'NGSIM-I80': 'D'}

SMO_VALS  = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
PHYS_VALS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
FRAC_VALS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
SEEDS     = [42, 137, 256, 314, 628]

DEFAULT_LAM_SMO  = 0.02
DEFAULT_LAM_PHYS = 0.10
SMO_REF_IDX  = SMO_VALS.index(DEFAULT_LAM_SMO)
PHYS_REF_IDX = PHYS_VALS.index(DEFAULT_LAM_PHYS)
FRAC_REF_IDX = len(FRAC_VALS) - 1

XLBL_SMO  = ['0', '0.005', '0.01', '0.02*', '0.05', '0.1', '0.2', '0.35', '0.5', '0.75', '1.0']
XLBL_PHYS = ['0', '0.005', '0.01', '0.02',  '0.05', '0.1*', '0.2', '0.35', '0.5', '0.75', '1.0']
XLBL_FRAC = ['10%', '20%', '30%', '40%', '50%', '65%', '80%', '100%*']


def load_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    from fd_model import TriangularFD
    prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) for p in data['prototypes']]
    meta = data['meta'].item()
    return dict(
        fd_points=data['fd_points'].item(),
        prototypes=prototypes,
        dx=meta['dx'],
        cells=meta['cells'],
        rho_obs=data['rho0'],
        rho_start=data['rho_start'] if 'rho_start' in data else None,
        rho_end=data['rho_end'] if 'rho_end' in data else None,
        dt_window=meta.get('T_sample', None),
    )


def _run(ds, seed, lam_smo, lam_phys, fd_points_override=None, x_gt=None):
    from solver import em_inverse_fd
    from phase_utils import spin_to_mixture_weights_softmax
    from baselines import _forward_metrics, _detect_transition_ped, _phys_residual

    np.random.seed(seed)
    fd_pts = fd_points_override if fd_points_override is not None else ds['fd_points']
    rho_obs = ds['rho_obs']
    cells = ds['cells']
    dx = ds['dx']

    v_obs = np.zeros(cells)
    mask = rho_obs > 1e-6
    for g in range(len(ds['prototypes'])):
        q_g = ds['prototypes'][g].flow(rho_obs)
        v_obs[mask] += q_g[mask] / rho_obs[mask] / len(ds['prototypes'])

    with contextlib.redirect_stdout(io.StringIO()):
        sx, sy, sz, protos_out, _, _ = em_inverse_fd(
            fd_pts, dx, M=cells, rho_obs=rho_obs, v_obs=v_obs,
            rho_start=ds['rho_start'], rho_end=ds['rho_end'], dt_window=ds['dt_window'],
            n_prototypes=3, em_iters=80, inner_iters=20,
            lam_fd_q=1.0, lam_phys=lam_phys, lam_smo=lam_smo,
            convergence_tol=5e-4, learning_rate=0.05, seed=seed,
        )

    pi = spin_to_mixture_weights_softmax(sx, sy, sz, 3)
    k = fd_pts['k'] / 1000.0
    q = fd_pts['q'] / 3600.0
    v = fd_pts['v'] / 3.6
    x = fd_pts['x_center']
    rmse_q, _, _, _, _, _ = _forward_metrics(k, q, v, x, pi, protos_out, dx, cells)

    tmae = float('nan')
    if x_gt is not None:
        try:
            x_pos = np.arange(cells) * dx
            x_star = _detect_transition_ped(pi, protos_out, k, q, x, x_pos, dx, cells,
                                            rho_obs=rho_obs)
            tmae = float(abs(x_star - x_gt))
        except Exception:
            tmae = float('nan')

    phys_res = _phys_residual(
        pi, protos_out, rho_obs, ds['rho_start'], ds['rho_end'], dx, ds['dt_window']
    )
    if phys_res is None:
        phys_res = float('nan')
    return rmse_q, tmae, float(phys_res)


def _subsample_fdpoints(fd_pts, frac, rng):
    n = fd_pts['n_points']
    idx = rng.choice(n, size=max(10, int(n * frac)), replace=False)
    sub = {
        key: fd_pts[key][idx]
        for key in ('k', 'q', 'v', 'x_center', 'score', 'weight')
        if key in fd_pts
    }
    sub['n_points'] = len(idx)
    return sub


def _normalize_per_seed(mat, ref_idx):
    arr = np.asarray(mat, dtype=float)
    ref = arr[:, [ref_idx]]
    ref = np.where(np.isfinite(ref) & (ref > 0), ref, np.nan)
    return arr / ref * 100.0


def _delta_tmae_per_seed(mat, ref_idx):
    arr = np.asarray(mat, dtype=float)
    ref = arr[:, [ref_idx]]
    return np.abs(arr - ref)


def _mean_std(mat):
    arr = np.asarray(mat, dtype=float)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


need_sweep = (not args.plot_only) and (args.rerun or not os.path.exists(CACHE_FILE))

if need_sweep:
    print("Loading datasets for sweep...")
    DS = {k: load_dataset(v) for k, v in DATASETS.items()}

    smo_rm_seed = {}
    smo_tm_seed = {}
    smo_pr_seed = {}
    phys_rm_seed = {}
    phys_tm_seed = {}
    phys_pr_seed = {}
    frac_rm_seed = {}
    frac_tm_seed = {}
    frac_pr_seed = {}

    for k in DS_KEYS:
        ds = DS[k]
        print(f"\n[Dataset] {k}")

        # lambda_smo sweep (hold lambda_phys at paper default 0.10)
        rm_mat = np.zeros((len(SEEDS), len(SMO_VALS)), dtype=float)
        tm_mat = np.zeros((len(SEEDS), len(SMO_VALS)), dtype=float)
        pr_mat = np.zeros((len(SEEDS), len(SMO_VALS)), dtype=float)
        for si, seed in enumerate(SEEDS):
            for vi, val in enumerate(SMO_VALS):
                rm, tm, pr = _run(
                    ds, seed, lam_smo=val, lam_phys=DEFAULT_LAM_PHYS, x_gt=X_GT.get(k)
                )
                rm_mat[si, vi] = rm
                tm_mat[si, vi] = tm
                pr_mat[si, vi] = pr
        smo_rm_seed[k] = rm_mat
        smo_tm_seed[k] = tm_mat
        smo_pr_seed[k] = pr_mat
        print(f"  lam_smo done ({len(SEEDS)} seeds x {len(SMO_VALS)} values)")

        # lambda_phys sweep (hold lambda_smo at paper default 0.02)
        rm_mat = np.zeros((len(SEEDS), len(PHYS_VALS)), dtype=float)
        tm_mat = np.zeros((len(SEEDS), len(PHYS_VALS)), dtype=float)
        pr_mat = np.zeros((len(SEEDS), len(PHYS_VALS)), dtype=float)
        for si, seed in enumerate(SEEDS):
            for vi, val in enumerate(PHYS_VALS):
                rm, tm, pr = _run(
                    ds, seed, lam_smo=DEFAULT_LAM_SMO, lam_phys=val, x_gt=X_GT.get(k)
                )
                rm_mat[si, vi] = rm
                tm_mat[si, vi] = tm
                pr_mat[si, vi] = pr
        phys_rm_seed[k] = rm_mat
        phys_tm_seed[k] = tm_mat
        phys_pr_seed[k] = pr_mat
        print(f"  lam_phys done ({len(SEEDS)} seeds x {len(PHYS_VALS)} values)")

        # data-fraction sweep (hold defaults)
        rm_mat = np.zeros((len(SEEDS), len(FRAC_VALS)), dtype=float)
        tm_mat = np.zeros((len(SEEDS), len(FRAC_VALS)), dtype=float)
        pr_mat = np.zeros((len(SEEDS), len(FRAC_VALS)), dtype=float)
        for si, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed)
            for vi, frac in enumerate(FRAC_VALS):
                sub = _subsample_fdpoints(ds['fd_points'], frac, rng)
                rm, tm, pr = _run(
                    ds, seed, lam_smo=DEFAULT_LAM_SMO, lam_phys=DEFAULT_LAM_PHYS,
                    fd_points_override=sub, x_gt=X_GT.get(k)
                )
                rm_mat[si, vi] = rm
                tm_mat[si, vi] = tm
                pr_mat[si, vi] = pr
        frac_rm_seed[k] = rm_mat
        frac_tm_seed[k] = tm_mat
        frac_pr_seed[k] = pr_mat
        print(f"  data fraction done ({len(SEEDS)} seeds x {len(FRAC_VALS)} values)")

    np.savez(
        CACHE_FILE,
        ds_keys=np.array(DS_KEYS),
        seeds=np.array(SEEDS),
        smo_vals=np.array(SMO_VALS),
        phys_vals=np.array(PHYS_VALS),
        frac_vals=np.array(FRAC_VALS),
        default_lam_smo=DEFAULT_LAM_SMO,
        default_lam_phys=DEFAULT_LAM_PHYS,
        **{f'smo_rm_seed_{k}': smo_rm_seed[k] for k in DS_KEYS},
        **{f'smo_tm_seed_{k}': smo_tm_seed[k] for k in DS_KEYS},
        **{f'smo_pr_seed_{k}': smo_pr_seed[k] for k in DS_KEYS},
        **{f'phys_rm_seed_{k}': phys_rm_seed[k] for k in DS_KEYS},
        **{f'phys_tm_seed_{k}': phys_tm_seed[k] for k in DS_KEYS},
        **{f'phys_pr_seed_{k}': phys_pr_seed[k] for k in DS_KEYS},
        **{f'frac_rm_seed_{k}': frac_rm_seed[k] for k in DS_KEYS},
        **{f'frac_tm_seed_{k}': frac_tm_seed[k] for k in DS_KEYS},
        **{f'frac_pr_seed_{k}': frac_pr_seed[k] for k in DS_KEYS},
    )
    print(f"\nCache saved -> {CACHE_FILE}")

cache = np.load(CACHE_FILE, allow_pickle=True)

for req in (
    f'smo_rm_seed_{DS_KEYS[0]}',
    f'phys_rm_seed_{DS_KEYS[0]}',
    f'frac_tm_seed_{DS_KEYS[0]}',
    f'frac_pr_seed_{DS_KEYS[0]}'
):
    if req not in cache.files:
        raise RuntimeError("Sensitivity cache is old-format. Please rerun with --rerun.")

smo_rm_seed = {k: cache[f'smo_rm_seed_{k}'] for k in DS_KEYS}
smo_tm_seed = {k: cache[f'smo_tm_seed_{k}'] for k in DS_KEYS}
phys_rm_seed = {k: cache[f'phys_rm_seed_{k}'] for k in DS_KEYS}
phys_tm_seed = {k: cache[f'phys_tm_seed_{k}'] for k in DS_KEYS}
frac_rm_seed = {k: cache[f'frac_rm_seed_{k}'] for k in DS_KEYS}
frac_tm_seed = {k: cache[f'frac_tm_seed_{k}'] for k in DS_KEYS}
frac_pr_seed = {k: cache[f'frac_pr_seed_{k}'] for k in DS_KEYS}
smo_pr_seed = {k: cache[f'smo_pr_seed_{k}'] for k in DS_KEYS}
phys_pr_seed = {k: cache[f'phys_pr_seed_{k}'] for k in DS_KEYS}

# Derived statistics for plotting.
smo_rm_norm = {k: _normalize_per_seed(smo_rm_seed[k], SMO_REF_IDX) for k in DS_KEYS}
phys_rm_norm = {k: _normalize_per_seed(phys_rm_seed[k], PHYS_REF_IDX) for k in DS_KEYS}
frac_rm_norm = {k: _normalize_per_seed(frac_rm_seed[k], FRAC_REF_IDX) for k in DS_KEYS}
smo_pr_norm = {k: _normalize_per_seed(smo_pr_seed[k], SMO_REF_IDX) for k in DS_KEYS}
phys_pr_norm = {k: _normalize_per_seed(phys_pr_seed[k], PHYS_REF_IDX) for k in DS_KEYS}
# For lambda_phys panels, visualize tiny sensitivity as delta from default (% points).
phys_rm_delta = {k: phys_rm_norm[k] - 100.0 for k in DS_KEYS}
phys_pr_delta = {k: phys_pr_norm[k] - 100.0 for k in DS_KEYS}
phys_rm_delta_scaled = {k: phys_rm_delta[k] * 1e4 for k in DS_KEYS}
phys_pr_delta_scaled = {k: phys_pr_delta[k] * 1e4 for k in DS_KEYS}

frac_pr_norm = {k: _normalize_per_seed(frac_pr_seed[k], FRAC_REF_IDX) for k in DS_KEYS}

smo_dt = {k: _delta_tmae_per_seed(smo_tm_seed[k], SMO_REF_IDX) for k in DS_KEYS}
phys_dt = {k: _delta_tmae_per_seed(phys_tm_seed[k], PHYS_REF_IDX) for k in DS_KEYS}
frac_dt = {k: _delta_tmae_per_seed(frac_tm_seed[k], FRAC_REF_IDX) for k in DS_KEYS}

smo_rm_mean = {k: _mean_std(smo_rm_norm[k])[0] for k in DS_KEYS}
smo_rm_std = {k: _mean_std(smo_rm_norm[k])[1] for k in DS_KEYS}
phys_rm_mean = {k: _mean_std(phys_rm_norm[k])[0] for k in DS_KEYS}
phys_rm_std = {k: _mean_std(phys_rm_norm[k])[1] for k in DS_KEYS}
phys_rm_delta_mean = {k: _mean_std(phys_rm_delta_scaled[k])[0] for k in DS_KEYS}
phys_rm_delta_std = {k: _mean_std(phys_rm_delta_scaled[k])[1] for k in DS_KEYS}
frac_rm_mean = {k: _mean_std(frac_rm_norm[k])[0] for k in DS_KEYS}
frac_rm_std = {k: _mean_std(frac_rm_norm[k])[1] for k in DS_KEYS}
smo_pr_mean = {k: _mean_std(smo_pr_norm[k])[0] for k in DS_KEYS}
smo_pr_std = {k: _mean_std(smo_pr_norm[k])[1] for k in DS_KEYS}
phys_pr_mean = {k: _mean_std(phys_pr_norm[k])[0] for k in DS_KEYS}
phys_pr_std = {k: _mean_std(phys_pr_norm[k])[1] for k in DS_KEYS}
phys_pr_delta_mean = {k: _mean_std(phys_pr_delta_scaled[k])[0] for k in DS_KEYS}
phys_pr_delta_std = {k: _mean_std(phys_pr_delta_scaled[k])[1] for k in DS_KEYS}
frac_pr_mean = {k: _mean_std(frac_pr_norm[k])[0] for k in DS_KEYS}
frac_pr_std = {k: _mean_std(frac_pr_norm[k])[1] for k in DS_KEYS}

smo_dt_mean = {k: _mean_std(smo_dt[k])[0] for k in DS_KEYS}
smo_dt_std = {k: _mean_std(smo_dt[k])[1] for k in DS_KEYS}
phys_dt_mean = {k: _mean_std(phys_dt[k])[0] for k in DS_KEYS}
phys_dt_std = {k: _mean_std(phys_dt[k])[1] for k in DS_KEYS}
frac_dt_mean = {k: _mean_std(frac_dt[k])[0] for k in DS_KEYS}
frac_dt_std = {k: _mean_std(frac_dt[k])[1] for k in DS_KEYS}


def _plot_with_band_and_errorbar(ax, x_idx, y_mean_dict, y_std_dict, *,
                                 x_ticks, x_label, y_label, title, zero_line=None,
                                 default_idx=None):
    for k in DS_KEYS:
        c = DS_COLORS[k]
        m = DS_MARKERS[k]
        y = np.asarray(y_mean_dict[k], dtype=float)
        s = np.asarray(y_std_dict[k], dtype=float)
        ax.plot(x_idx, y, marker=m, ms=2.8, color=c, lw=1.35, label=k, zorder=3)
        if default_idx is not None and 0 <= default_idx < len(y) and np.isfinite(y[default_idx]):
            # Mark the paper-default hyperparameter point directly on each curve.
            ax.plot(
                x_idx[default_idx], y[default_idx],
                marker='*', ms=8.4, mfc='#D62728', mec='#D62728', mew=0.9,
                color='#D62728', zorder=5
            )
        ax.fill_between(x_idx, y - s, y + s, color=c, alpha=0.10, zorder=1)
    if zero_line is not None:
        ax.axhline(zero_line, color='#888', lw=0.7, ls='--', alpha=0.55)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_ticks, rotation=35, ha='right', fontsize=8)
    ax.set_xlabel(x_label, fontsize=10, labelpad=2)
    ax.set_ylabel(y_label, fontsize=9.5)
    ax.grid(True, alpha=0.13, lw=0.4)
    ax.text(0.5, -0.50, title, transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontweight='bold')


def _tight_plain_yaxis(ax, y_mean_dict):
    """Zoom to data range while keeping default tick formatting."""
    all_y = np.concatenate([np.asarray(y_mean_dict[k], dtype=float) for k in DS_KEYS])
    ymin = float(np.nanmin(all_y))
    ymax = float(np.nanmax(all_y))
    pad = max((ymax - ymin) * 0.25, 1e-5)
    ax.set_ylim(ymin - pad, ymax + pad)


fig, axes = plt.subplots(3, 3, figsize=(11.2, 6.2))
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.10, top=0.98,
                    wspace=0.30, hspace=0.70)

# Top row: RMSE_q norm (%)
_plot_with_band_and_errorbar(
    axes[0, 0], list(range(len(SMO_VALS))), smo_rm_mean, smo_rm_std,
    x_ticks=XLBL_SMO, x_label=r'$\lambda_\mathrm{smooth}$',
    y_label=r'RMSE$_q$ (norm., \%)',
    title=r'(a) RMSE$_q$ vs. $\lambda_\mathrm{smooth}$',
    zero_line=100.0,
    default_idx=SMO_REF_IDX,
)
axes[0, 0].legend(fontsize=6.5, ncol=2, loc='upper left',
                  framealpha=0.85, edgecolor='none')

_plot_with_band_and_errorbar(
    axes[0, 1], list(range(len(PHYS_VALS))), phys_rm_delta_mean, phys_rm_delta_std,
    x_ticks=XLBL_PHYS, x_label=r'$\lambda_\mathrm{phys}$',
    y_label=r'$\Delta$RMSE$_q$ ($\times 10^{-4}$\%)',
    title=r'(b) $\Delta$RMSE$_q$ vs. $\lambda_\mathrm{phys}$',
    zero_line=0.0,
    default_idx=PHYS_REF_IDX,
)
_tight_plain_yaxis(axes[0, 1], phys_rm_delta_mean)
axes[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

_plot_with_band_and_errorbar(
    axes[0, 2], list(range(len(FRAC_VALS))), frac_rm_mean, frac_rm_std,
    x_ticks=XLBL_FRAC, x_label='Fraction of FD points used',
    y_label=r'RMSE$_q$ (norm., \%)',
    title=r'(c) RMSE$_q$ vs. data fraction',
    zero_line=100.0,
    default_idx=FRAC_REF_IDX,
)

# Bottom row: Phys. Residual (normalized %)
_plot_with_band_and_errorbar(
    axes[1, 0], list(range(len(SMO_VALS))), smo_pr_mean, smo_pr_std,
    x_ticks=XLBL_SMO, x_label=r'$\lambda_\mathrm{smooth}$',
    y_label=r'Phys.Res. (norm., \%)',
    title=r'(d) Phys.Res. vs. $\lambda_\mathrm{smooth}$',
    zero_line=100.0,
    default_idx=SMO_REF_IDX,
)
_plot_with_band_and_errorbar(
    axes[1, 1], list(range(len(PHYS_VALS))), phys_pr_delta_mean, phys_pr_delta_std,
    x_ticks=XLBL_PHYS, x_label=r'$\lambda_\mathrm{phys}$',
    y_label=r'$\Delta$Phys.Res. ($\times 10^{-4}$\%)',
    title=r'(e) $\Delta$Phys.Res. vs. $\lambda_\mathrm{phys}$',
    zero_line=0.0,
    default_idx=PHYS_REF_IDX,
)
_tight_plain_yaxis(axes[1, 1], phys_pr_delta_mean)
axes[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
_plot_with_band_and_errorbar(
    axes[1, 2], list(range(len(FRAC_VALS))), frac_pr_mean, frac_pr_std,
    x_ticks=XLBL_FRAC, x_label='Fraction of FD points used',
    y_label=r'Phys.Res. (norm., \%)',
    title=r'(f) Phys.Res. vs. data fraction',
    zero_line=100.0,
    default_idx=FRAC_REF_IDX,
)

# Third row: |Delta T.MAE| (m)
_plot_with_band_and_errorbar(
    axes[2, 0], list(range(len(SMO_VALS))), smo_dt_mean, smo_dt_std,
    x_ticks=XLBL_SMO, x_label=r'$\lambda_\mathrm{smooth}$',
    y_label=r'$|\Delta$T.MAE$|$ (m)',
    title=r'(g) $|\Delta$T.MAE$|$ vs. $\lambda_\mathrm{smooth}$',
    zero_line=0.0,
    default_idx=SMO_REF_IDX,
)
_plot_with_band_and_errorbar(
    axes[2, 1], list(range(len(PHYS_VALS))), phys_dt_mean, phys_dt_std,
    x_ticks=XLBL_PHYS, x_label=r'$\lambda_\mathrm{phys}$',
    y_label=r'$|\Delta$T.MAE$|$ (m)',
    title=r'(h) $|\Delta$T.MAE$|$ vs. $\lambda_\mathrm{phys}$',
    zero_line=0.0,
    default_idx=PHYS_REF_IDX,
)
axes[2, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
_plot_with_band_and_errorbar(
    axes[2, 2], list(range(len(FRAC_VALS))), frac_dt_mean, frac_dt_std,
    x_ticks=XLBL_FRAC, x_label='Fraction of FD points used',
    y_label=r'$|\Delta$T.MAE$|$ (m)',
    title=r'(i) $|\Delta$T.MAE$|$ vs. data fraction',
    zero_line=0.0,
    default_idx=FRAC_REF_IDX,
)

for ext in ('pdf', 'png'):
    out = os.path.join(FIG_DIR, f'fig6_sensitivity.{ext}')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved: {out}")
plt.close(fig)
print("[Done] fig6_sensitivity complete.")
