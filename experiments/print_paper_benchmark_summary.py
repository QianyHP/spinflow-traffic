"""
Summarize experiment_results.csv: console tables and LaTeX snippets for the paper.

Run from repo root with PYTHONPATH including ``experiments`` (or ``cd experiments``).
Requires ``experiment_results.csv`` under FIG_DIR (written by run_paper_benchmarks).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from paths import FIG_DIR

csv_path = os.path.join(FIG_DIR, 'experiment_results.csv')
df = pd.read_csv(csv_path)

MODELS = [
    ('SpinFlow',       'spinflow'),
    ('PWA-CTM',        'pwa_ctm'),
    ('VBGMM+KDE',     'vbgmm'),
    ('PI-DeepONet',    'deeponet'),
    ('Abl-Mapping',   'abl_mapping'),
    ('Abl-Spin',      'abl_spin'),
    ('Abl-Physics',   'abl_physics'),
]

rows_by_ds = {}
sig_by_ds = {}

for ds_key in ['YTDJ', 'RML']:
    sf_vals = df[(df['dataset']==ds_key) & (df['model_key']=='spinflow')]['rmse_q'].values
    rows_by_ds[ds_key] = {}
    sig_by_ds[ds_key] = {}
    print(f'\n=== {ds_key} ===')
    for mn, mk in MODELS:
        sub = df[(df['dataset']==ds_key) & (df['model_key']==mk)]
        row = {}
        for col in ['rmse_q','r2_q','trans_mae','phys_residual','entropy_std','runtime_s','convergence_steps']:
            vals = sub[col].dropna()
            row[col+'_m'] = float(vals.mean()) if len(vals) else None
            row[col+'_s'] = float(vals.std()) if len(vals) else None
        rows_by_ds[ds_key][mk] = row
        sig = False
        if mk != 'spinflow':
            ov = sub['rmse_q'].values
            if len(sf_vals)==len(ov) and len(sf_vals)>=2:
                _,p = stats.ttest_rel(sf_vals, ov)
                sig = bool(p < 0.05)
        sig_by_ds[ds_key][mk] = sig
        star = '*' if sig else ''
        rm = row['rmse_q_m']
        rs = row['rmse_q_s']
        r2 = row['r2_q_m']
        mae_m = row['trans_mae_m']
        rt = row['runtime_s_m']
        steps = row['convergence_steps_m']
        rms_str = f"{rm:.1f}+/-{rs:.2f}" if rm is not None else "N/A"
        mae_str = f"{mae_m:.1f}" if mae_m is not None else "N/A"
        r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
        rt_str = f"{rt:.2f}s" if rt is not None else "N/A"
        steps_str = f"{steps:.0f}" if steps is not None else "N/A"
        print(f"  {mn+star:20s}  RMSE_q={rms_str}  R2={r2_str}  MAE={mae_str}  RT={rt_str}  Steps={steps_str}")

print("\n\n=== LaTeX Table A: Performance ===")
for ds_key in ['YTDJ', 'RML']:
    print(f"\n% {ds_key}")
    print(r"\begin{table}[t]")
    print(r"\caption{" + f"{ds_key} performance. Best in bold. $^*$: $p<0.05$ vs. best baseline (paired $t$-test)." + r"}")
    print(r"\label{tab:" + ds_key.lower() + r"_perf}\centering\footnotesize")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{l c c c c c}")
    print(r"\toprule")
    print(r"Model & RMSE$_q$ (veh/h) & $R_q^2$ & Trans.\ MAE (m) & Phys.\ Res. & Entr.\ Std \\")
    print(r"\midrule")
    T = rows_by_ds[ds_key]
    best_rmse = min((T[mk]['rmse_q_m'] for _, mk in MODELS if T[mk]['rmse_q_m'] is not None))
    best_r2   = max((T[mk]['r2_q_m']   for _, mk in MODELS if T[mk]['r2_q_m']   is not None))
    best_mae  = min((T[mk]['trans_mae_m'] for _, mk in MODELS if T[mk]['trans_mae_m'] is not None))
    best_phys = min((T[mk]['phys_residual_m'] for _, mk in MODELS if T[mk]['phys_residual_m'] is not None))
    best_ent  = min((T[mk]['entropy_std_m'] for _, mk in MODELS if T[mk]['entropy_std_m'] is not None))
    for mn, mk in MODELS:
        r = T[mk]
        star = r"$^*$" if sig_by_ds[ds_key].get(mk) else ""
        def b(v, ref, fmt, higher=False):
            if v is None: return "N/A"
            s = f"{v:{fmt}}"
            if (higher and abs(v-ref)<1e-9) or (not higher and abs(v-ref)<1e-9):
                return r"\textbf{" + s + "}"
            return s
        rmse_m = r['rmse_q_m']; rmse_s = r['rmse_q_s']
        r2_m   = r['r2_q_m']
        mae_m  = r['trans_mae_m']; mae_s = r['trans_mae_s']
        phys_m = r['phys_residual_m']
        ent_m  = r['entropy_std_m']
        rmse_str = (r"\textbf{" + f"{rmse_m:.1f}" + r"}$\pm$" + f"{rmse_s:.1f}" if rmse_m is not None and abs(rmse_m - best_rmse) < 0.05 else (f"{rmse_m:.1f}$\\pm${rmse_s:.1f}" if rmse_m is not None else "N/A"))
        r2_str   = (r"\textbf{" + f"{r2_m:.3f}" + "}" if r2_m is not None and abs(r2_m - best_r2) < 1e-4 else (f"{r2_m:.3f}" if r2_m is not None else "N/A"))
        mae_str  = (r"\textbf{" + f"{mae_m:.1f}" + r"}$\pm$" + f"{mae_s:.1f}" if mae_m is not None and abs(mae_m - best_mae) < 0.05 else (f"{mae_m:.1f}$\\pm${mae_s:.1f}" if mae_m is not None else "--"))
        phys_str = (r"\textbf{" + f"{phys_m:.2e}" + "}" if phys_m is not None and abs(phys_m - best_phys) < 1e-15 else (f"{phys_m:.2e}" if phys_m is not None else "--"))
        ent_str  = (r"\textbf{" + f"{ent_m:.3f}" + "}" if ent_m is not None and abs(ent_m - best_ent) < 1e-6 else (f"{ent_m:.3f}" if ent_m is not None else "--"))
        print(f"  {mn}{star} & {rmse_str} & {r2_str} & {mae_str} & {phys_str} & {ent_str}" + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")

print("\n\n=== LaTeX Table B: Efficiency ===")
INTERP = {'spinflow': 'Yes', 'pwa_ctm': 'Partial', 'vbgmm': 'Partial',
          'deeponet': 'No', 'abl_mapping': 'Yes', 'abl_spin': 'Yes', 'abl_physics': 'Yes'}
for ds_key in ['YTDJ', 'RML']:
    print(f"\n% {ds_key}")
    print(r"\begin{table}[t]")
    print(r"\caption{" + f"{ds_key} computational efficiency." + r"}")
    print(r"\label{tab:" + ds_key.lower() + r"_eff}\centering\footnotesize")
    print(r"\begin{tabular}{l c c c}")
    print(r"\toprule")
    print(r"Model & Runtime (s) & Conv.\ Steps & $\pi(x)$ \\")
    print(r"\midrule")
    T = rows_by_ds[ds_key]
    for mn, mk in MODELS:
        r = T[mk]
        rt_m = r['runtime_s_m']; rt_s = r['runtime_s_s']
        cs_m = r['convergence_steps_m']
        rt_str = f"{rt_m:.2f}$\\pm${rt_s:.2f}" if rt_m is not None else "N/A"
        cs_str = f"{cs_m:.0f}" if cs_m is not None else "N/A"
        print(f"  {mn} & {rt_str} & {cs_str} & {INTERP.get(mk,'?')}" + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
