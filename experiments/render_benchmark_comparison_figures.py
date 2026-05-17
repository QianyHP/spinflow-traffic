"""
Generate comparison figures for SpinFlow baseline & ablation experiments.

Fig 6: Grouped bar chart of key metrics (RMSE_q, Trans.MAE) + radar/panel
Fig 7: Phase weight profiles — SpinFlow vs all baselines (2x2: F+S phases x 2 datasets)
Fig 8: Pareto Efficiency Tradeoff (Runtime vs RMSE_q, with interpretability annotation)

Reads results from experiment_results.csv and experiment_results.npz.
"""

import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from paths import FIG_DIR, RESULTS_DIR, SPIN_SRC

sys.path.insert(0, SPIN_SRC)
from spinflow.fd_model import PHASES, PHASE_INDEX, PHASE_DISPLAY, TriangularFD

os.makedirs(FIG_DIR, exist_ok=True)

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
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

T10 = {
    'spinflow':    '#1f77b4',
    'vbgmm':       '#ff7f0e',
    'abl_mapping': '#2ca02c',
    'pwa_ctm':     '#d62728',
    'deeponet':    '#9467bd',
    'abl_spin':    '#8c564b',
    'abl_physics': '#e377c2',
    'abl_single':  '#7f7f7f',
}

LABEL_MAP = {
    'spinflow': 'SpinFlow',
    'pwa_ctm': 'PWA-CTM',
    'vbgmm': 'VBGMM+KDE',
    'deeponet': 'PI-DeepONet',
    'abl_mapping': 'Abl: No Comp.',
    'abl_spin': 'Abl: Unit Norm',
    'abl_physics': 'Abl: No Phys.',
    'abl_single': 'Abl: Single FD',
}

SHORT_LABEL = {
    'spinflow': 'SpinFlow',
    'pwa_ctm': 'PWA\nCTM',
    'vbgmm': 'VB\nGMM',
    'deeponet': 'Deep\nONet',
    'abl_mapping': 'No\nComp.',
    'abl_spin': 'Unit\nNorm',
    'abl_physics': 'No\nPhys.',
    'abl_single': 'Single\nFD',
}

DATASETS_CFG = {
    'YTDJ': {'npz': os.path.join(RESULTS_DIR, 'YTDJ', 'YTDJ_inverse_init.npz'),
             'x_gt': 66.0, 'label': 'YTDJ'},
    'RML':  {'npz': os.path.join(RESULTS_DIR, 'RML', 'RML_inverse_init.npz'),
             'x_gt': 192.0, 'label': 'RML'},
}

MODEL_ORDER = ['spinflow', 'pwa_ctm', 'vbgmm', 'deeponet',
               'abl_mapping', 'abl_spin', 'abl_physics', 'abl_single']


def load_csv():
    csv_path = os.path.join(FIG_DIR, 'experiment_results.csv')
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ['rmse_q', 'r2_q', 'rmse_v', 'r2_v', 'x_star', 'trans_mae',
                       'phys_residual', 'entropy_std', 'runtime_s', 'convergence_steps', 'seed']:
                if r[k] and r[k] != 'None' and r[k] != '':
                    r[k] = float(r[k])
                else:
                    r[k] = None
            rows.append(r)
    return rows


def load_pi_x():
    npz_path = os.path.join(FIG_DIR, 'experiment_results.npz')
    return dict(np.load(npz_path, allow_pickle=True))


def load_ds_grid(ds_key):
    data = np.load(DATASETS_CFG[ds_key]['npz'], allow_pickle=True)
    meta = data['meta'].item()
    dx = meta['dx']
    cells = meta['cells']
    x_pos = np.arange(cells) * dx
    fd_points = data['fd_points'].item()
    prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) for p in data['prototypes']]
    return x_pos, dx, cells, fd_points, prototypes


def _agg(rows_csv, ds_key, mk, metric):
    vals = [r[metric] for r in rows_csv
            if r['dataset'] == ds_key and r['model_key'] == mk and r[metric] is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


# ============================================================================
# Fig 6: Grouped Bar Chart — RMSE_q + Transition MAE side by side
# ============================================================================
def gen_fig6(rows_csv):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.2))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.93,
                        wspace=0.30, hspace=0.45)

    metrics = [('rmse_q', r'RMSE$_q$ (veh/h)'), ('trans_mae', 'Transition MAE (m)')]

    for row, (metric, ylabel) in enumerate(metrics):
        for col, ds_key in enumerate(['YTDJ', 'RML']):
            ax = axes[row, col]
            means, stds, colors, labels = [], [], [], []
            for mk in MODEL_ORDER:
                m, s = _agg(rows_csv, ds_key, mk, metric)
                means.append(m if m is not None else 0)
                stds.append(s if s is not None else 0)
                colors.append(T10[mk])
                labels.append(SHORT_LABEL[mk])

            x = np.arange(len(MODEL_ORDER))
            bars = ax.bar(x, means, yerr=stds, width=0.65,
                          color=colors, edgecolor='white', lw=0.5,
                          capsize=2, error_kw={'lw': 0.8})

            # highlight SpinFlow bar
            bars[0].set_edgecolor('#333333')
            bars[0].set_linewidth(1.2)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(ds_key, fontweight='bold')
            ax.grid(axis='y', alpha=0.2, lw=0.4)

            # mark missing values
            for i, mk in enumerate(MODEL_ORDER):
                m, _ = _agg(rows_csv, ds_key, mk, metric)
                if m is None:
                    ax.text(i, ax.get_ylim()[1] * 0.05, 'N/A', ha='center',
                            fontsize=5, color='grey')

    fig.savefig(os.path.join(FIG_DIR, 'fig6_metric_comparison.png'),
                bbox_inches='tight', pad_inches=0.02)
    print("Saved: fig6_metric_comparison.png")
    plt.close(fig)


# ============================================================================
# Fig 7: Phase Weight Profiles — SpinFlow vs key models (2x2)
# ============================================================================
def gen_fig7(pi_x_data):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.93,
                        wspace=0.22, hspace=0.38)

    phases_show = ['free', 'critical']
    phase_labels = [r'$\pi_F(x)$', r'$\pi_S(x)$']
    models_to_show = ['spinflow', 'pwa_ctm', 'vbgmm', 'deeponet', 'abl_mapping']

    for row, (phase, plabel) in enumerate(zip(phases_show, phase_labels)):
        g = PHASE_INDEX[phase]
        for col, ds_key in enumerate(['YTDJ', 'RML']):
            ax = axes[row, col]
            x_pos, _, _, _, _ = load_ds_grid(ds_key)
            x_gt = DATASETS_CFG[ds_key]['x_gt']

            for mk in models_to_show:
                key = f'{ds_key}_{mk}'
                if key in pi_x_data:
                    pi = pi_x_data[key]
                    ax.plot(x_pos, pi[:, g], color=T10[mk], lw=1.3,
                            label=LABEL_MAP[mk], alpha=0.85)

            ax.axvline(x_gt, color='grey', lw=0.8, ls='--', alpha=0.6,
                       label=r'$x_{\mathrm{gt}}$' if row == 0 and col == 0 else None)
            ax.set_xlim(0, x_pos.max())
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.15, lw=0.4)

            if row == 1:
                ax.set_xlabel('Position (m)')
            if col == 0:
                ax.set_ylabel(plabel)
            if row == 0:
                ax.set_title(ds_key, fontweight='bold')
            if row == 0 and col == 1:
                ax.legend(fontsize=5.5, loc='upper right', framealpha=0.9,
                          edgecolor='none', ncol=2)

    fig.savefig(os.path.join(FIG_DIR, 'fig7_phase_profiles.png'),
                bbox_inches='tight', pad_inches=0.02)
    print("Saved: fig7_phase_profiles.png")
    plt.close(fig)


# ============================================================================
# Fig 8: Pareto Efficiency — Runtime vs RMSE_q (both datasets combined)
# ============================================================================
def gen_fig8(rows_csv):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.88,
                        wspace=0.30)

    markers = {'spinflow': 'o', 'pwa_ctm': 's', 'vbgmm': '^', 'deeponet': 'D',
               'abl_mapping': 'v', 'abl_spin': '<', 'abl_physics': '>'}

    for col, ds_key in enumerate(['YTDJ', 'RML']):
        ax = axes[col]
        for mk in MODEL_ORDER:
            sub = [r for r in rows_csv
                   if r['dataset'] == ds_key and r['model_key'] == mk]
            rts = [r['runtime_s'] for r in sub if r['runtime_s'] is not None]
            rmses = [r['rmse_q'] for r in sub if r['rmse_q'] is not None]

            if rts and rmses:
                rt_m, rt_s = np.mean(rts), np.std(rts)
                rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
                ax.errorbar(rt_m, rmse_m, xerr=rt_s, yerr=rmse_s,
                            fmt=markers.get(mk, 'o'), color=T10[mk], ms=7,
                            capsize=2, lw=1.0, label=LABEL_MAP[mk], zorder=5)

        ax.set_xscale('log')
        ax.set_xlabel('Runtime (s)')
        ax.set_ylabel(r'RMSE$_q$ (veh/h)')
        ax.set_title(ds_key, fontsize=8.5, fontweight='bold')
        ax.grid(True, alpha=0.18, lw=0.4)
        if col == 1:
            ax.legend(fontsize=5, loc='upper right', framealpha=0.9,
                      edgecolor='none', handlelength=1.5, ncol=2)

    fig.savefig(os.path.join(FIG_DIR, 'fig8_pareto_tradeoff.png'),
                bbox_inches='tight', pad_inches=0.02)
    print("Saved: fig8_pareto_tradeoff.png")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Loading experiment results...")
    rows_csv = load_csv()
    pi_x_data = load_pi_x()

    gen_fig6(rows_csv)
    gen_fig7(pi_x_data)
    gen_fig8(rows_csv)
    print("\n[Done] All comparison figures generated.")
