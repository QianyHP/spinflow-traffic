"""
Multi-seed paper benchmark runner: SpinFlow, baselines, and ablations.

Grid: four trajectory-derived datasets x eight model keys x five random seeds.
Writes ``results/paper_figures/experiment_results.csv``, a companion ``.npz`` of phase
weights (seed 42), and prints LaTeX-ready tables with paired RMSE_q t-tests.

Hyperparameter policy documented here is frozen for the manuscript: physics weight
``lam_phys=0.1``, EM convergence tolerance ``5e-4``, and PI-DeepONet trained with
rolling-window loss convergence (no patience) while exporting a pseudo ``pi_x`` so
all models share the same metric columns.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, time, platform, json, csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from paths import FIG_DIR, RESULTS_DIR, SPIN_SRC

sys.path.insert(0, SPIN_SRC)
sys.path.insert(0, SCRIPT_DIR)

import torch
torch.set_num_threads(1)

from spinflow.fd_model import TriangularFD, PHASES, PHASE_INDEX
from spinflow.solver import em_inverse_fd
from spinflow.phase_utils import spin_to_mixture_weights_softmax
from paper_benchmark_baselines import (
    run_pwa_ctm, run_vbgmm, run_pi_deeponet,
    _forward_metrics, _entropy_profile, _detect_transition_ped, _phys_residual,
)
from paper_benchmark_ablations import (
    run_ablation_mapping, run_ablation_spin_norm,
    run_ablation_no_physics, run_ablation_single_phase,
)
from scipy import stats


def print_env_info():
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'torch_threads': torch.get_num_threads(),
        'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
    }
    print("=" * 72)
    print("HARDWARE / ENV ALIGNMENT")
    print("=" * 72)
    for k, v in info.items():
        print(f"  {k:20s}: {v}")
    print("=" * 72)
    return info


SEEDS = [42, 137, 256, 314, 628]

DATASETS = {
    'YTDJ': {
        'npz': os.path.join(RESULTS_DIR, 'YTDJ', 'YTDJ_inverse_init.npz'),
        'x_gt': 66.0,
        'lam_phys': 0.1, 'lam_smo': 0.02, 'lam_fd_q': 1.0,
        'lr': 0.05, 'em_iters': 80, 'inner_iters': 20,
    },
    'RML': {
        'npz': os.path.join(RESULTS_DIR, 'RML', 'RML_inverse_init.npz'),
        'x_gt': 192.0,
        'lam_phys': 0.1, 'lam_smo': 0.02, 'lam_fd_q': 1.0,
        'lr': 0.05, 'em_iters': 80, 'inner_iters': 20,
    },
    'HighD': {
        'npz': os.path.join(RESULTS_DIR, 'HighD', 'HighD_inverse_init.npz'),
        'x_gt': 87.9,
        'lam_phys': 0.1, 'lam_smo': 0.02, 'lam_fd_q': 1.0,
        'lr': 0.05, 'em_iters': 80, 'inner_iters': 20,
    },
    'NGSIM-I80': {
        'npz': os.path.join(RESULTS_DIR, 'NGSIM-I80', 'NGSIM-I80_inverse_init.npz'),
        'x_gt': 392.0,
        'lam_phys': 0.1, 'lam_smo': 0.02, 'lam_fd_q': 1.0,
        'lr': 0.05, 'em_iters': 80, 'inner_iters': 20,
    },
}

MODELS = [
    ('SpinFlow',          'spinflow'),
    ('PWA-CTM',           'pwa_ctm'),
    ('VBGMM+KDE',        'vbgmm'),
    ('PI-DeepONet',       'deeponet'),
    ('Abl: No Comp.',     'abl_mapping'),
    ('Abl: Unit Norm',    'abl_spin'),
    ('Abl: No Phys.',     'abl_physics'),
    ('Abl: Single FD',    'abl_single'),
]


def load_dataset(key):
    cfg = DATASETS[key]
    data = np.load(cfg['npz'], allow_pickle=True)
    pi_final = data['pi_final']
    proto_data = data['prototypes']
    fd_points = data['fd_points'].item()
    meta = data['meta'].item()
    prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) for p in proto_data]
    dx = meta['dx']
    cells = meta['cells']
    x_pos = np.arange(cells) * dx
    rho_obs = data['rho0']
    rho_start = data.get('rho_start', None)
    rho_end = data.get('rho_end', None)
    dt_window = meta.get('T_sample', None)
    return {
        'fd_points': fd_points, 'prototypes': prototypes,
        'dx': dx, 'cells': cells, 'x_pos': x_pos,
        'rho_obs': rho_obs, 'rho_start': rho_start, 'rho_end': rho_end,
        'dt_window': dt_window, 'pi_final': pi_final,
        'sx0': data['sx0'], 'sy0': data['sy0'], 'sz0': data['sz0'],
        'meta': meta,
    }


# SpinFlow full pipeline: forward observation model → em_inverse_fd → metrics.
def run_spinflow(fd_points, prototypes, dx, cells, x_pos, rho_obs,
                 seed=42, rho_start=None, rho_end=None, dt_window=None,
                 lam_fd_q=1.0, lam_phys=0.1, lam_smo=0.02,
                 lr=0.05, em_iters=80, inner_iters=20, **kw):
    t0 = time.perf_counter()
    np.random.seed(seed)

    v_obs = np.zeros(cells)
    mask = rho_obs > 1e-6
    for g in range(len(prototypes)):
        q_g = prototypes[g].flow(rho_obs)
        v_obs[mask] += q_g[mask] / rho_obs[mask] / len(prototypes)

    sx, sy, sz, protos_out, _, history = em_inverse_fd(
        fd_points, dx, M=cells, rho_obs=rho_obs, v_obs=v_obs,
        rho_start=rho_start, rho_end=rho_end, dt_window=dt_window,
        n_prototypes=3, em_iters=em_iters, inner_iters=inner_iters,
        lam_fd_q=lam_fd_q, lam_phys=lam_phys, lam_smo=lam_smo,
        convergence_tol=5e-4, learning_rate=lr, seed=seed,
    )
    pi_final = spin_to_mixture_weights_softmax(sx, sy, sz, 3)

    k_pts = fd_points['k'] / 1000.0
    q_pts = fd_points['q'] / 3600.0
    v_pts = fd_points['v'] / 3.6
    x_pts = fd_points['x_center']

    rmse_q, r2_q, rmse_v, r2_v, _, _ = _forward_metrics(
        k_pts, q_pts, v_pts, x_pts, pi_final, protos_out, dx, cells)
    x_star = _detect_transition_ped(pi_final, protos_out, k_pts, q_pts, x_pts, x_pos, dx, cells,
                                    rho_obs=rho_obs)
    H = _entropy_profile(pi_final)
    phys_res = _phys_residual(pi_final, protos_out, rho_obs, rho_start, rho_end, dx, dt_window)
    runtime = time.perf_counter() - t0

    return {
        'rmse_q': rmse_q, 'r2_q': r2_q, 'rmse_v': rmse_v, 'r2_v': r2_v,
        'x_star': x_star, 'pi_x': pi_final,
        'phys_residual': phys_res, 'entropy_std': float(H.std()),
        'runtime_s': runtime, 'convergence_steps': len(history['loss']),
    }


# Maps CLI / table model names to runner callables (main text + ablations).
DISPATCH = {
    'spinflow':     run_spinflow,
    'pwa_ctm':      run_pwa_ctm,
    'vbgmm':        run_vbgmm,
    'deeponet':     run_pi_deeponet,
    'abl_mapping':  run_ablation_mapping,
    'abl_spin':     run_ablation_spin_norm,
    'abl_physics':  run_ablation_no_physics,
    'abl_single':   run_ablation_single_phase,
}


def main():
    env_info = print_env_info()
    all_results = []

    for ds_key in DATASETS:
        print(f"\n{'#'*72}")
        print(f"# Dataset: {ds_key}")
        print(f"{'#'*72}")
        ds = load_dataset(ds_key)
        cfg = DATASETS[ds_key]
        x_gt = cfg['x_gt']

        common_kw = dict(
            fd_points=ds['fd_points'], prototypes=ds['prototypes'],
            dx=ds['dx'], cells=ds['cells'], x_pos=ds['x_pos'],
            rho_obs=ds['rho_obs'], rho_start=ds['rho_start'],
            rho_end=ds['rho_end'], dt_window=ds['dt_window'],
            lam_fd_q=cfg['lam_fd_q'], lam_phys=cfg['lam_phys'],
            lam_smo=cfg['lam_smo'], lr=cfg['lr'],
            em_iters=cfg['em_iters'], inner_iters=cfg['inner_iters'],
        )

        for model_name, model_key in MODELS:
            fn = DISPATCH[model_key]
            print(f"\n  >> {model_name} ({len(SEEDS)} seeds) ...", end='', flush=True)
            for si, seed in enumerate(SEEDS):
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    res = fn(seed=seed, **common_kw)

                trans_mae = abs(res['x_star'] - x_gt) if res['x_star'] is not None else None
                row = {
                    'dataset': ds_key, 'model': model_name, 'model_key': model_key,
                    'seed': seed,
                    'rmse_q': res['rmse_q'], 'r2_q': res['r2_q'],
                    'rmse_v': res['rmse_v'], 'r2_v': res['r2_v'],
                    'x_star': res['x_star'], 'trans_mae': trans_mae,
                    'phys_residual': res['phys_residual'],
                    'entropy_std': res['entropy_std'],
                    'runtime_s': res['runtime_s'],
                    'convergence_steps': res['convergence_steps'],
                }
                all_results.append(row)
                print(f" s{si}", end='', flush=True)
            print(" done.")

    # Persist per-seed rows for auditing and downstream plotting.
    csv_path = os.path.join(FIG_DIR, 'experiment_results.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = list(all_results[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved raw CSV: {csv_path}")

    # Also save pi_x for figure generation (re-run seed=42)
    npz_path = os.path.join(FIG_DIR, 'experiment_results.npz')
    pi_x_store = {}
    for ds_key in DATASETS:
        ds = load_dataset(ds_key)
        cfg = DATASETS[ds_key]
        common_kw2 = dict(
            fd_points=ds['fd_points'], prototypes=ds['prototypes'],
            dx=ds['dx'], cells=ds['cells'], x_pos=ds['x_pos'],
            rho_obs=ds['rho_obs'], rho_start=ds['rho_start'],
            rho_end=ds['rho_end'], dt_window=ds['dt_window'],
            lam_fd_q=cfg['lam_fd_q'], lam_phys=cfg['lam_phys'],
            lam_smo=cfg['lam_smo'], lr=cfg['lr'],
            em_iters=cfg['em_iters'], inner_iters=cfg['inner_iters'],
        )
        for model_name, model_key in MODELS:
            fn = DISPATCH[model_key]
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = fn(seed=42, **common_kw2)
            if res.get('pi_x') is not None:
                pi_x_store[f'{ds_key}_{model_key}'] = res['pi_x']

    np.savez(npz_path, **pi_x_store)

    import pandas as pd
    df = pd.DataFrame(all_results)

    print("\n\n" + "=" * 72)
    print("AGGREGATED RESULTS (mean +/- std over 5 seeds)")
    print("=" * 72)

    agg_metrics = ['rmse_q', 'r2_q', 'trans_mae', 'phys_residual',
                   'entropy_std', 'runtime_s', 'convergence_steps']

    tables = {}

    for ds_key in DATASETS:
        print(f"\n--- {ds_key} ---")
        for model_name, model_key in MODELS:
            sub = df[(df['dataset'] == ds_key) & (df['model_key'] == model_key)]
            row = {}
            for m in agg_metrics:
                vals = sub[m].dropna()
                if len(vals) > 0:
                    row[f'{m}_mean'] = float(vals.mean())
                    row[f'{m}_std'] = float(vals.std())
                else:
                    row[f'{m}_mean'] = None
                    row[f'{m}_std'] = None
            tables[(ds_key, model_name)] = row
            line = f"  {model_name:18s}"
            for m in ['rmse_q', 'r2_q', 'trans_mae']:
                mn = row.get(f'{m}_mean')
                sd = row.get(f'{m}_std')
                if mn is not None:
                    line += f"  {m}={mn:.2f}+/-{sd:.2f}"
                else:
                    line += f"  {m}=N/A"
            print(line)

    print("\n\n" + "=" * 72)
    print("PAIRED T-TESTS (SpinFlow vs baselines, on RMSE_q)")
    print("=" * 72)
    sig_marks = {}

    for ds_key in DATASETS:
        sf_vals = df[(df['dataset'] == ds_key) & (df['model_key'] == 'spinflow')]['rmse_q'].values
        for model_name, model_key in MODELS:
            if model_key == 'spinflow':
                sig_marks[(ds_key, model_name)] = False
                continue
            other_vals = df[(df['dataset'] == ds_key) & (df['model_key'] == model_key)]['rmse_q'].values
            if len(sf_vals) == len(other_vals) and len(sf_vals) >= 2:
                t_stat, p_val = stats.ttest_rel(sf_vals, other_vals)
                sig = p_val < 0.05
                sig_marks[(ds_key, model_name)] = sig
                print(f"  {ds_key} | SpinFlow vs {model_name:18s} | t={t_stat:+.3f} p={p_val:.4f} {'*' if sig else ''}")
            else:
                sig_marks[(ds_key, model_name)] = False

    print("\n\n" + "=" * 72)
    print("LATEX TABLE A: Performance & Topological Accuracy")
    print("=" * 72)

    def _fmt(val, fmt_str, bold=False):
        if val is None:
            return "--"
        s = f"{val:{fmt_str}}"
        if bold:
            return r"\textbf{" + s + "}"
        return s

    def _fmt_pm(mean, std, fmt_str, bold=False):
        if mean is None:
            return "--"
        s = f"{mean:{fmt_str}}" + r"$\pm$" + f"{std:{fmt_str}}"
        if bold:
            return r"\textbf{" + s + "}"
        return s

    for ds_key in DATASETS:
        best = {}
        for m in ['rmse_q', 'trans_mae', 'phys_residual', 'entropy_std']:
            vals = [(tables[(ds_key, mn)][f'{m}_mean'], mn) for mn, _ in MODELS
                    if tables[(ds_key, mn)].get(f'{m}_mean') is not None]
            if vals:
                best[m] = min(vals, key=lambda x: x[0])[1]
        vals_r2 = [(tables[(ds_key, mn)]['r2_q_mean'], mn) for mn, _ in MODELS
                   if tables[(ds_key, mn)].get('r2_q_mean') is not None]
        if vals_r2:
            best['r2_q'] = max(vals_r2, key=lambda x: x[0])[1]

        print(f"\n% --- {ds_key} ---")
        print(r"\begin{table}[t]")
        print(f"\\caption{{{ds_key} performance comparison.}}")
        print(f"\\label{{tab:{ds_key.lower()}_perf}}")
        print(r"\centering\footnotesize")
        print(r"\resizebox{\columnwidth}{!}{%")
        print(r"\begin{tabular}{l c c c c c}")
        print(r"\toprule")
        print(r"Model & RMSE$_q$ & $R_q^2$ & Trans.\ MAE (m) & Phys.\ Res. & Entr.\ Std \\")
        print(r"\midrule")

        for model_name, model_key in MODELS:
            t = tables[(ds_key, model_name)]
            star = "$^*$" if sig_marks.get((ds_key, model_name), False) else ""

            cols = []
            cols.append(model_name + star)
            cols.append(_fmt_pm(t['rmse_q_mean'], t['rmse_q_std'], '.1f',
                                bold=(best.get('rmse_q') == model_name)))
            cols.append(_fmt(t['r2_q_mean'], '.3f',
                             bold=(best.get('r2_q') == model_name)))
            cols.append(_fmt_pm(t['trans_mae_mean'], t['trans_mae_std'], '.1f',
                                bold=(best.get('trans_mae') == model_name)))
            cols.append(_fmt(t['phys_residual_mean'], '.2e',
                             bold=(best.get('phys_residual') == model_name)))
            cols.append(_fmt(t['entropy_std_mean'], '.3f',
                             bold=(best.get('entropy_std') == model_name)))
            print(" & ".join(cols) + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular}%")
        print(r"}")
        print(r"\end{table}")

    print("\n\n" + "=" * 72)
    print("LATEX TABLE B: Computational Efficiency")
    print("=" * 72)

    for ds_key in DATASETS:
        print(f"\n% --- {ds_key} ---")
        print(r"\begin{table}[t]")
        print(f"\\caption{{{ds_key} computational efficiency.}}")
        print(f"\\label{{tab:{ds_key.lower()}_efficiency}}")
        print(r"\centering\footnotesize")
        print(r"\begin{tabular}{l c c c}")
        print(r"\toprule")
        print(r"Model & Runtime (s) & Conv.\ Steps & Interpretable \\")
        print(r"\midrule")

        for model_name, model_key in MODELS:
            t = tables[(ds_key, model_name)]
            interp = "Yes" if model_key not in ('deeponet',) else "No"
            if model_key in ('pwa_ctm', 'vbgmm'):
                interp = "Partial"
            rt = _fmt_pm(t['runtime_s_mean'], t['runtime_s_std'], '.2f')
            cs = _fmt(t['convergence_steps_mean'], '.0f')
            print(f"  {model_name} & {rt} & {cs} & {interp}" + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    print("\n\n[Done] All experiments completed.")


if __name__ == '__main__':
    main()
