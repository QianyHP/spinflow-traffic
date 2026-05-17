"""
SpinFlow robustness analysis (Fig 6).

Panel (a): RMSE_q vs lam_smo sweep.
Panel (b): RMSE_q vs lam_phys sweep.
Panel (c): Data efficiency — RMSE_q vs fraction of FD points used.

Sweep results are cached to fig6_sensitivity_cache.npz so that layout/style
changes can be made without re-running the experiments (pass --plot-only).

Usage:
  python experiments/run_regularization_sweep.py            # sweeps + plot (skips if cache exists)
  python experiments/run_regularization_sweep.py --rerun    # force re-run sweeps
  python experiments/run_regularization_sweep.py --plot-only  # figure only from cache
"""
import os, sys, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from paths import FIG_DIR, RESULTS_DIR, SPIN_SRC

sys.path.insert(0, SPIN_SRC)
CACHE_FILE = os.path.join(SCRIPT_DIR, 'fig6_sensitivity_cache.npz')
os.makedirs(FIG_DIR, exist_ok=True)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--rerun',     action='store_true')
parser.add_argument('--plot-only', action='store_true')
args, _ = parser.parse_known_args()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.5,
})

DS_KEYS    = ['YTDJ', 'RML', 'HighD', 'NGSIM-I80']
DATASETS   = {
    'YTDJ':      os.path.join(RESULTS_DIR, 'YTDJ', 'YTDJ_inverse_init.npz'),
    'RML':       os.path.join(RESULTS_DIR, 'RML', 'RML_inverse_init.npz'),
    'HighD':     os.path.join(RESULTS_DIR, 'HighD', 'HighD_inverse_init.npz'),
    'NGSIM-I80': os.path.join(RESULTS_DIR, 'NGSIM-I80', 'NGSIM-I80_inverse_init.npz'),
}
DS_COLORS  = {'YTDJ':'#2166AC','RML':'#D6604D','HighD':'#4DAF4A','NGSIM-I80':'#984EA3'}
DS_MARKERS = {'YTDJ':'o','RML':'s','HighD':'^','NGSIM-I80':'D'}

SMO_VALS    = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
PHYS_VALS   = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
FRAC_VALS   = [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
N_SEEDS_EFF = 5
SEED_BASE   = 42


def load_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    proto_data = data['prototypes']
    from spinflow.fd_model import TriangularFD
    prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0'])
                  for p in proto_data]
    meta      = data['meta'].item()
    fd_points = data['fd_points'].item()
    dx        = meta['dx']; cells = meta['cells']
    rho_obs   = data['rho0']
    rho_start = data['rho_start'] if 'rho_start' in data else None
    rho_end   = data['rho_end']   if 'rho_end'   in data else None
    dt_window = meta.get('T_sample', None)
    return dict(fd_points=fd_points, prototypes=prototypes, dx=dx, cells=cells,
                rho_obs=rho_obs, rho_start=rho_start,
                rho_end=rho_end, dt_window=dt_window)


def _run(ds, seed, lam_smo=0.02, lam_phys=0.02, fd_points_override=None):
    import io, contextlib
    from spinflow.solver import em_inverse_fd
    from phase_utils import spin_to_mixture_weights_softmax
    from paper_benchmark_baselines import _forward_metrics, _entropy_profile
    np.random.seed(seed)
    fd_pts  = fd_points_override if fd_points_override is not None else ds['fd_points']
    rho_obs = ds['rho_obs']; cells = ds['cells']; dx = ds['dx']
    v_obs   = np.zeros(cells)
    mask    = rho_obs > 1e-6
    for g in range(len(ds['prototypes'])):
        q_g = ds['prototypes'][g].flow(rho_obs)
        v_obs[mask] += q_g[mask] / rho_obs[mask] / len(ds['prototypes'])
    with contextlib.redirect_stdout(io.StringIO()):
        sx, sy, sz, protos_out, _, _ = em_inverse_fd(
            fd_pts, dx, M=cells, rho_obs=rho_obs, v_obs=v_obs,
            rho_start=ds['rho_start'], rho_end=ds['rho_end'],
            dt_window=ds['dt_window'],
            n_prototypes=3, em_iters=80, inner_iters=20,
            lam_fd_q=1.0, lam_phys=lam_phys, lam_smo=lam_smo,
            convergence_tol=5e-4, learning_rate=0.05, seed=seed,
        )
    pi = spin_to_mixture_weights_softmax(sx, sy, sz, 3)
    k  = fd_pts['k']/1000; q = fd_pts['q']/3600
    v  = fd_pts['v']/3.6;  x = fd_pts['x_center']
    rq, r2q, rv, r2v, _, _ = _forward_metrics(k, q, v, x, pi, protos_out, dx, cells)
    H  = _entropy_profile(pi)
    return rq, r2q, H.std()


def _subsample_fdpoints(fd_pts, frac, rng):
    n   = fd_pts['n_points']
    idx = rng.choice(n, size=max(10, int(n * frac)), replace=False)
    sub = {key: fd_pts[key][idx] for key in ('k', 'q', 'v', 'x_center', 'score', 'weight')
           if key in fd_pts}
    sub['n_points'] = len(idx)
    return sub


need_sweep = (not args.plot_only) and (args.rerun or not os.path.exists(CACHE_FILE))

if need_sweep:
    print("Loading datasets for sweep...")
    DS = {k: load_dataset(v) for k, v in DATASETS.items()}

    print("\n[A] lam_smo sweep...")
    smo_r = {k: [] for k in DS_KEYS}
    for k, ds in DS.items():
        for val in SMO_VALS:
            rq, _, _ = _run(ds, SEED_BASE, lam_smo=val, lam_phys=0.02)
            smo_r[k].append(rq)
            print(f"  {k}  smo={val:.3f}  RMSE={rq:.1f}")

    print("\n[A] lam_phys sweep...")
    phy_r = {k: [] for k in DS_KEYS}
    for k, ds in DS.items():
        for val in PHYS_VALS:
            rq, _, _ = _run(ds, SEED_BASE, lam_smo=0.02, lam_phys=val)
            phy_r[k].append(rq)
            print(f"  {k}  phys={val:.3f}  RMSE={rq:.1f}")

    print("\n[B] Data-efficiency sweep...")
    eff_mean = {k: [] for k in DS_KEYS}
    eff_std  = {k: [] for k in DS_KEYS}
    for k, ds in DS.items():
        n_full = ds['fd_points']['n_points']
        for frac in FRAC_VALS:
            vals = []
            for s in range(N_SEEDS_EFF):
                rng = np.random.default_rng(SEED_BASE + s)
                sub = _subsample_fdpoints(ds['fd_points'], frac, rng)
                rq, _, _ = _run(ds, SEED_BASE + s, fd_points_override=sub)
                vals.append(rq)
            m, sd = float(np.mean(vals)), float(np.std(vals))
            eff_mean[k].append(m); eff_std[k].append(sd)
            print(f"  {k}  frac={frac:.0%}  n={int(n_full*frac)}  RMSE={m:.1f}+/-{sd:.1f}")

    # Save all sweep results to cache
    np.savez(CACHE_FILE,
             ds_keys=DS_KEYS,
             smo_vals=SMO_VALS, phys_vals=PHYS_VALS, frac_vals=FRAC_VALS,
             **{f'smo_{k}':  np.array(smo_r[k])       for k in DS_KEYS},
             **{f'phy_{k}':  np.array(phy_r[k])        for k in DS_KEYS},
             **{f'effm_{k}': np.array(eff_mean[k])     for k in DS_KEYS},
             **{f'effs_{k}': np.array(eff_std[k])      for k in DS_KEYS},
    )
    print(f"\nCache saved → {CACHE_FILE}")

else:
    # Load from cache (supports current keys smo_*/phy_* and legacy smom_*/phym_*)
    print(f"Loading cached sweep results from {CACHE_FILE}")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    SMO_VALS  = cache['smo_vals'].tolist()
    PHYS_VALS = cache['phys_vals'].tolist()
    FRAC_VALS = cache['frac_vals'].tolist()

    def _cache_vec(prefix, k, legacy_prefixes=()):
        """Return 1D cache column as list; try ``prefix_k`` then legacy key stems."""
        for stem in (prefix,) + tuple(legacy_prefixes):
            name = f'{stem}_{k}'
            if name in cache.files:
                return cache[name].tolist()
        raise KeyError(
            f"No cache column for dataset {k!r} (tried {prefix}_{k}"
            + ''.join(f', {p}_{k}' for p in legacy_prefixes)
            + f"). Archive keys: {sorted(cache.files)}"
        )

    smo_r = {k: _cache_vec('smo', k, ('smom',)) for k in DS_KEYS}
    phy_r = {k: _cache_vec('phy', k, ('phym',)) for k in DS_KEYS}
    eff_mean = {k: _cache_vec('effm', k) for k in DS_KEYS}
    eff_std  = {k: _cache_vec('effs', k) for k in DS_KEYS}

fig, axes = plt.subplots(3, 1, figsize=(4, 5.8))
fig.subplots_adjust(left=0.16, right=0.92, bottom=0.06, top=0.98, hspace=0.78)

xlbls    = ['0','0.005','0.01','0.02*','0.05','0.1','0.2','0.5']
pct_lbls = ['10%','20%','30%','40%','50%','65%','80%','100%*']

def _norm(vals, ref_idx=3):
    base = vals[ref_idx] if vals[ref_idx] > 0 else 1.0
    return [v / base * 100.0 for v in vals]

# Panel (a): lam_smo
ax = axes[0]
for k in DATASETS:
    c = DS_COLORS[k]; m = DS_MARKERS[k]
    ax.plot(range(len(SMO_VALS)), _norm(smo_r[k]),
            marker=m, ms=4, color=c, lw=1.4, label=k)
ax.axhline(100, color='#888', lw=0.7, ls='--', alpha=0.5)
ax.set_xticks(range(len(SMO_VALS)))
ax.set_xticklabels(xlbls, rotation=35, ha='right', fontsize=6)
ax.set_ylabel(r'RMSE$_q$ (norm., \%)')
ax.set_xlabel(r'$\lambda_\mathrm{smo}$')
ax.legend(fontsize=5.5, loc='upper left', framealpha=0.85, edgecolor='none')
ax.grid(True, alpha=0.13, lw=0.4)
ax.text(0.5, -0.5, r'(a) RMSE$_q$ vs.\ $\lambda_\mathrm{smo}$',
        transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold')

# Panel (b): lam_phys
ax = axes[1]
for k in DATASETS:
    c = DS_COLORS[k]; m = DS_MARKERS[k]
    ax.plot(range(len(PHYS_VALS)), _norm(phy_r[k]),
            marker=m, ms=4, color=c, lw=1.4, label=k)
ax.axhline(100, color='#888', lw=0.7, ls='--', alpha=0.5)
ax.set_xticks(range(len(PHYS_VALS)))
ax.set_xticklabels(xlbls, rotation=35, ha='right', fontsize=6)
ax.set_ylabel(r'RMSE$_q$ (norm., \%)')
ax.set_xlabel(r'$\lambda_\mathrm{phys}$')
ax.legend(fontsize=5.5, loc='upper left', framealpha=0.85, edgecolor='none')
ax.grid(True, alpha=0.13, lw=0.4)
ax.text(0.5, -0.5, r'(b) RMSE$_q$ vs.\ $\lambda_\mathrm{phys}$',
        transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold')

# Panel (c): data efficiency
ax = axes[2]
xs = list(range(len(FRAC_VALS)))
for k in DATASETS:
    c = DS_COLORS[k]; m = DS_MARKERS[k]
    ms_arr = np.array(eff_mean[k]); ss_arr = np.array(eff_std[k])
    base = ms_arr[-1] if ms_arr[-1] > 0 else 1.0
    ms_n = ms_arr / base * 100.0; ss_n = ss_arr / base * 100.0
    ax.plot(xs, ms_n, marker=m, ms=4, color=c, lw=1.4, label=k)
    ax.fill_between(xs, ms_n - ss_n, ms_n + ss_n, color=c, alpha=0.12)
ax.axhline(100, color='#888', lw=0.7, ls='--', alpha=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(pct_lbls, rotation=35, ha='right', fontsize=6)
ax.set_ylabel(r'RMSE$_q$ (norm., \%)')
ax.set_xlabel('Fraction of FD points used')
ax.legend(fontsize=5.5, loc='lower right', framealpha=0.85, edgecolor='none')
ax.grid(True, alpha=0.13, lw=0.4)
ax.text(0.5, -0.5, r'(c) RMSE$_q$ vs.\ data fraction',
        transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold')

for ext in ('pdf', 'png'):
    path = os.path.join(FIG_DIR, f'fig6_sensitivity.{ext}')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved: {path}")
plt.close(fig)
print("[Done] fig6_sensitivity complete.")
