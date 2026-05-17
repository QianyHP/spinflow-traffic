"""
Post-inversion diagnostics: FD panels and initial-condition sanity checks.

Reads ``*_inverse_init.npz``, rebuilds Edie fields from the recorded CSV path in ``meta``,
and writes FD scatter / initial-field figures next to the NPZ.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib.pyplot as plt
import os
from spinflow.repo_paths import repo_root
from spinflow.preprocessing import load_trajectory_csv, select_direction, filter_lanes, parse_lanes_arg
from spinflow.observation import compute_edie_sle

_DEFAULT_NPZ = str(repo_root() / "results" / "YTDJ" / "YTDJ_inverse_init.npz")
npz_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_NPZ
output_dir = os.path.dirname(npz_path)
save_prefix = os.path.splitext(os.path.basename(npz_path))[0].replace('_inverse_init', '')

print("="*80)
print("Generating FD Analysis Plots")
print("="*80)
print(f"Output Directory: {output_dir}")

# Load inversion NPZ
print("\n[1/6] Loading FD inversion results...")
data = np.load(npz_path, allow_pickle=True)
rho0 = data['rho0']
sx0 = data['sx0']
sy0 = data['sy0']
sz0 = data['sz0']
fd_points = data['fd_points'].item()
prototypes_data = data['prototypes']
meta = data['meta'].item()

dx = meta['dx']
cells = meta['cells']

print(f"  - Method: {meta.get('method', 'FD-EM')}")
print(f"  - FD points: {fd_points['n_points']}")
print(f"  - Cells: {cells}, dx={dx:.2f}m")

# Reload trajectories and Edie field
print("\n[2/6] Loading and gridding observation data...")
csv_path = meta.get('csv_path', None)
if not csv_path:
    raise ValueError("meta.csv_path missing in npz; please re-run main.py to regenerate results.")
print(f"  - CSV Path: {csv_path}")
fps = float(meta.get('fps', 24.0))
road_length = float(meta.get('road_length', 280.0))
road_length_raw = float(meta.get('road_length_raw', road_length))
direction = meta.get('direction', 'eb')
lanes = parse_lanes_arg(meta.get('lanes', 'all'))
t0 = float(meta.get('t0', 0.0))
# Keep visualization window consistent with the inversion window saved in meta
T_obs = float(meta.get('T_sample', 60.0))
dt_obs = float(meta.get('dt', 0.25))
x_offset = float(meta.get('x_offset', meta.get('road_begin', 0.0)) or 0.0)

df = load_trajectory_csv(csv_path, fps=fps)
df = select_direction(df, direction, road_length_raw)
df = filter_lanes(df, lanes)
if x_offset > 0:
    df = df[(df["s"] >= x_offset) & (df["s"] <= road_length_raw)].copy()
    df["s"] = df["s"] - x_offset

edie = compute_edie_sle(
    df,
    dx=dx,
    dt=dt_obs,
    t_start=t0,
    t_end=t0 + T_obs,
    road_length=road_length,
    fps=fps,
    lane_count_threshold=int(meta.get('lane_count_threshold', 10)),
)
rho_sle = edie.rho_sle
L = edie.lanes

# normalize for plotting
rho_jam_est = float(data['rho_jam_est']) if 'rho_jam_est' in data else np.quantile(rho_sle[rho_sle > 0], 0.99)
rho_norm = np.clip(rho_sle / max(rho_jam_est, 1e-6), 0.0, 1.0)

K_obs, M_obs = rho_norm.shape
print(f"  - Observation data: {K_obs} steps x {M_obs} cells")
print(f"  - rho_obs range: [{rho_norm.min():.4f}, {rho_norm.max():.4f}]")

# Mixture weights (for overlays)
print("\n[3/6] Computing phase mixture parameters...")
from spinflow.fd_model import TriangularFD
from spinflow.phase_utils import spin_to_mixture_weights_softmax, mixture_fd_params

prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) 
              for p in prototypes_data]

pi = spin_to_mixture_weights_softmax(sx0, sy0, sz0, len(prototypes))
vf, Qcap, w, rho_jam = mixture_fd_params(pi, prototypes)

print(f"  - Phase weights π computed")
print(f"  - vf range: [{vf.min()*3.6:.1f}, {vf.max()*3.6:.1f}] km/h")
print(f"  - Qcap range: [{Qcap.min()*3600:.0f}, {Qcap.max()*3600:.0f}] veh/h")

# phase_analysis merged into phase_map figure
print("\n[4/6] Skipping phase_analysis (merged into phase_map)...")

# FD scatter + prototype panel
print("\n[5/6] Generating FD analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# FD scatter + prototypes
ax = axes[0]
k_data = fd_points['k']
q_data = fd_points['q']
score = fd_points['score']

scatter = ax.scatter(k_data, q_data, c=score, s=60, alpha=0.7, 
                    cmap='viridis', edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax, label='Score (lower=better)')

rho_range = np.linspace(0, 0.3, 200)
k_range = rho_range * 1000
colors = ['blue', 'green', 'red']
for i, (proto, color) in enumerate(zip(prototypes, colors)):
    q_curve = proto.flow(rho_range) * 3600
    ax.plot(k_range, q_curve, color=color, linewidth=2.5, linestyle='--',
            label=f'Prototype {i}: vf={proto.vf*3.6:.1f}, Q={proto.Q0*3600:.0f}')
    ax.plot(proto.rho_c*1000, proto.Q0*3600, 'o', color=color, markersize=10)

ax.set_xlabel('Density (veh/km)', fontsize=12)
ax.set_ylabel('Flow (veh/h)', fontsize=12)
ax.set_title(f'FD: {fd_points["n_points"]} Points + 3 Prototypes', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(k_data.max()*1.2, 250))
ax.set_ylim(0, max(q_data.max()*1.2, 2500))

# Score histogram
ax = axes[1]
ax.hist(score, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(score.mean(), color='red', linestyle='--', linewidth=2, 
          label=f'Mean={score.mean():.3f}')
ax.set_xlabel('Score', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Score Distribution (CV/NAE based)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_path_analysis = os.path.join(output_dir, f'{save_prefix}_analysis.png')
plt.savefig(save_path_analysis, dpi=200, bbox_inches='tight')
print(f"  --> {save_path_analysis}")
plt.close()

# initial_analysis multi-panel
# NOTE: Text reports are generated by `evaluator.py` as the single source of truth.
print("\n[6/6] Generating initial analysis...")

x_pos = np.arange(cells) * dx
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# rho(x) observation
ax = axes[0, 0]
rho0_norm = np.clip(rho0 / max(rho_jam_est, 1e-6), 0.0, 1.0)
ax.plot(
    x_pos,
    rho0_norm,
    linewidth=2.5,
    alpha=0.8,
    color='#1f77b4',
    label='Observed Density (Edie, normalized)',
)
# ax.axvspan(100, 200, alpha=0.2, color='red', label='Bottleneck', zorder=1)
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('Initial Density Field ρ0(x) - From Observation', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, road_length)
ax.set_ylim(-0.05, 1.05)

# Spin components
ax = axes[0, 1]
spin_mag = np.sqrt(sx0**2 + sy0**2 + sz0**2)
ax.plot(x_pos, sx0, label='sx', alpha=0.7, linewidth=1.5)
ax.plot(x_pos, sy0, label='sy', alpha=0.7, linewidth=1.5)
ax.plot(x_pos, sz0, label='sz', alpha=0.7, linewidth=1.5)
# ax.plot(x_pos, spin_mag, 'k--', label='|s|', linewidth=2)
# ax.axvspan(100, 200, alpha=0.2, color='red')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Spin Components', fontsize=11)
ax.set_title('Spin Field Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, road_length)

# Lane count
ax = axes[1, 0]
ax.bar(x_pos, L, width=dx*0.8, alpha=0.6, color='steelblue', edgecolor='navy')
# ax.axvspan(100, 200, alpha=0.2, color='red')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Number of Lanes', fontsize=11)
ax.set_title('Lane Count Profile L(x)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(0, road_length)
ax.set_ylim(0, 6)

# drho/dx
ax = axes[1, 1]
grad_rho = np.gradient(rho0_norm)
ax.plot(x_pos, grad_rho, color='darkblue', linewidth=2)
# # ax.axvspan(100, 200, alpha=0.2, color='red')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('d(rho)/dx', fontsize=11)
ax.set_title('Density Gradient', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, road_length)

plt.tight_layout()
save_path_initial = os.path.join(output_dir, f'{save_prefix}_initial_analysis.png')
plt.savefig(save_path_initial, dpi=200, bbox_inches='tight')
print(f"  --> {save_path_initial}")
plt.close()


print("\n" + "="*80)
print("All FD analysis plots generated successfully!")
print(f"Generated {3} common plots + {2} FD-specific plots = 5 plots total")
print("="*80)