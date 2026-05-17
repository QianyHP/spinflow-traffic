"""
Spacetime (t, x) diagrams: Edie density heatmap with phase overlays.

Uses inversion outputs plus trajectory replay to highlight dominant-phase corridors
and optional zoom panels with sampling evidence.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import argparse
import os

from spinflow.repo_paths import repo_root
from spinflow.fd_model import PHASE_INDEX, PHASE_DISPLAY, PHASE_COLORS
from spinflow.preprocessing import load_trajectory_csv, select_direction, filter_lanes, parse_lanes_arg
from spinflow.observation import compute_edie_sle

parser = argparse.ArgumentParser()
_DEFAULT_NPZ = str(repo_root() / "results" / "YTDJ" / "YTDJ_inverse_init.npz")
parser.add_argument("npz_path", type=str, nargs='?', default=_DEFAULT_NPZ)
args = parser.parse_args()

npz_path = args.npz_path
output_dir = os.path.dirname(npz_path)
base_name = os.path.basename(npz_path).replace('_inverse_init.npz', '')

data = np.load(npz_path, allow_pickle=True)
rho0 = data['rho0']
pi_final = data['pi_final']
fd_points = data['fd_points'].item()
meta = data['meta'].item()

dx = meta['dx']
cells = meta['cells']
x_pos = np.arange(cells) * dx

csv_path = meta.get('csv_path', None)
if not csv_path:
    raise ValueError("meta.csv_path missing in npz; please re-run main.py to regenerate results.")
print(f"Loading observation data from: {csv_path}")

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

# Windowed trajectories for overlay (shared by zoom panels + main panel)
df_window = df[(df['t'] >= t0) & (df['t'] < t0 + T_obs)].copy()

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

rho_jam_est = float(data['rho_jam_est'])
rho_norm = np.clip(rho_sle / max(rho_jam_est, 1e-6), 0.0, 1.0)

# grid sizes
K, M = rho_norm.shape

# Zoom window for the three top panels (seconds).
# Rationale: top panels are meant to show fine spacetime structures; using the full T_obs
# can overplot trajectories and hide details, especially for long windows.
T_zoom = min(60.0, float(T_obs))

# Dominant-phase corridors along x
dominant_phase = np.argmax(pi_final, axis=1)

def get_dynamic_region(indices, total_M, padding=5):
    if len(indices) < 3:
        return (0, min(20, total_M))
    start = np.min(indices)
    end = np.max(indices)
    plot_start = max(0, start - padding)
    plot_end = min(total_M, end + padding)
    return (plot_start, plot_end)

free_cells = np.where(dominant_phase == PHASE_INDEX["free"])[0]
free_region = get_dynamic_region(free_cells, M)

critical_cells = np.where(dominant_phase == PHASE_INDEX["critical"])[0]
critical_region = get_dynamic_region(critical_cells, M)

congested_cells = np.where(dominant_phase == PHASE_INDEX["congested"])[0]
congested_region = get_dynamic_region(congested_cells, M)

print(f"Identified Regions:")
print(f"  {PHASE_DISPLAY['free']}:    x={free_region[0]*dx:.0f}-{free_region[1]*dx:.0f}m")
print(f"  {PHASE_DISPLAY['critical']}:     x={critical_region[0]*dx:.0f}-{critical_region[1]*dx:.0f}m")
print(f"  {PHASE_DISPLAY['congested']}:    x={congested_region[0]*dx:.0f}-{congested_region[1]*dx:.0f}m")

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.5], hspace=0.25, wspace=0.3)

# Convention: x-axis = Time (s), y-axis = Position (m)
extent = [t0, t0 + T_obs, 0, road_length]

# Top row: zoomed spacetime panels
regions = [
    (free_region, f"{PHASE_DISPLAY['free']} Region", PHASE_INDEX["free"]),
    (critical_region, f"{PHASE_DISPLAY['critical']} Region", PHASE_INDEX["critical"]),
    (congested_region, f"{PHASE_DISPLAY['congested']} Region (Bottleneck)", PHASE_INDEX["congested"])
]

for idx, (region, title, col_idx) in enumerate(regions):
    ax = fig.add_subplot(gs[0, col_idx])
    
    x_start, x_end = region
    rho_region_full = rho_norm[:, x_start:x_end]  # [K, x_width]

    # Pick a representative time window around the peak mean density in this region
    # to avoid "everything overlaps" in long sequences.
    mean_rho_t = rho_region_full.mean(axis=1)  # [K]
    k_peak = int(np.argmax(mean_rho_t))
    t_peak = t0 + (k_peak + 0.5) * dt_obs
    t_start_zoom = float(np.clip(t_peak - 0.5 * T_zoom, t0, t0 + T_obs - T_zoom))
    t_end_zoom = t_start_zoom + T_zoom
    k0 = int(np.clip((t_start_zoom - t0) / dt_obs, 0, K - 1))
    k1 = int(np.clip((t_end_zoom - t0) / dt_obs, 1, K))

    rho_region = rho_norm[k0:k1, x_start:x_end]
    extent_region = [t_start_zoom, t_end_zoom, x_start * dx, x_end * dx]
    
    im = ax.imshow(rho_region.T, aspect='auto', origin='lower', cmap='jet',
                   vmin=0, vmax=0.8, extent=extent_region, interpolation='bilinear')
    
    for p_data in fd_points['x_center']:
        if x_start*dx <= p_data <= x_end*dx:
            pass
    
    df_region = df_window[
        (df_window['t'] >= t_start_zoom) & (df_window['t'] < t_end_zoom) &
        (df_window['s'] >= x_start * dx) & (df_window['s'] <= x_end * dx)
    ]
    if len(df_region) > 0:
        sample_ratio = max(1, len(df_region) // 2000)
        df_sample = df_region.iloc[::sample_ratio]
        ax.scatter(df_sample['t'], df_sample['s'], s=0.5, c='white', alpha=0.6, rasterized=True)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position (m)', fontsize=11)
    ax.set_title(f'{title}\n({x_start*dx:.0f}-{x_end*dx:.0f}m)', fontsize=12, fontweight='bold')
    ax.set_xlim(t_start_zoom, t_end_zoom)
    ax.set_ylim(x_start * dx, x_end * dx)
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', rotation=270, labelpad=15, fontsize=10)

# Bottom: full spacetime + FD inset
ax_main = fig.add_subplot(gs[1, :])

im_main = ax_main.imshow(rho_norm.T, aspect='auto', origin='lower', cmap='jet',
                         vmin=0, vmax=0.8, extent=extent, interpolation='bilinear')

sample_ratio_main = max(1, len(df_window) // 10000)
df_sample_main = df_window.iloc[::sample_ratio_main]
ax_main.scatter(df_sample_main['t'], df_sample_main['s'],
               s=0.8, c='white', alpha=0.8, rasterized=True)

for region, title, color in [
    (free_region, PHASE_DISPLAY["free"], PHASE_COLORS["free"]),
    (critical_region, PHASE_DISPLAY["critical"], PHASE_COLORS["critical"]),
    (congested_region, PHASE_DISPLAY["congested"], PHASE_COLORS["congested"])
]:
    x_start, x_end = region
    rect = patches.Rectangle((t0, x_start * dx), T_obs, (x_end - x_start) * dx,
                             linewidth=2.5, edgecolor=color, facecolor='none',
                             linestyle='--', alpha=0.8, label=title)
    ax_main.add_patch(rect)

ax_main.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
ax_main.set_ylabel('Position (m)', fontsize=13, fontweight='bold')
ax_main.set_title('Spacetime Diagram with Phase Regions', fontsize=14, fontweight='bold')
ax_main.set_xlim(t0, t0 + T_obs)
ax_main.set_ylim(0, road_length)
ax_main.legend(loc='upper right', fontsize=11)

# colorbar
cbar_main = plt.colorbar(im_main, ax=ax_main)
cbar_main.set_label('Normalized Density', rotation=270, labelpad=20, fontsize=12)

# FD inset on main axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_inset = inset_axes(ax_main, width="25%", height="30%", loc='upper left',
                      bbox_to_anchor=(0.02, 0.02, 0.96, 0.96), bbox_transform=ax_main.transAxes)

from spinflow.fd_model import TriangularFD
prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) 
              for p in data['prototypes']]

k_data = fd_points['k']
q_data = fd_points['q']
score = fd_points['score']

ax_inset.scatter(k_data, q_data, c=score, s=20, alpha=0.6, 
                cmap='viridis_r', edgecolors='black', linewidth=0.3)

rho_range = np.linspace(0, 0.25, 100)
k_range = rho_range * 1000
colors = ['blue', 'green', 'red']
for i, (proto, color) in enumerate(zip(prototypes, colors)):
    q_curve = proto.flow(rho_range) * 3600
    ax_inset.plot(k_range, q_curve, color=color, linewidth=1.5, linestyle='--', alpha=0.7)

ax_inset.set_xlabel('k (veh/km)', fontsize=9)
ax_inset.set_ylabel('q (veh/h)', fontsize=9)
ax_inset.set_title(f'FD: {len(k_data)} points', fontsize=10, fontweight='bold')
ax_inset.grid(True, alpha=0.3, linewidth=0.5)
ax_inset.tick_params(labelsize=8)
ax_inset.set_xlim(0, 250)
ax_inset.set_ylim(0, 2500)

save_path = os.path.join(output_dir, f'{base_name}_spacetime_regions.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved Spacetime Regions Plot: {save_path}")
plt.close()

print("\n[OK] Spacetime regions visualization generated!")
print(f"  - Zoomed views of 3 phase regions")
print(f"  - Complete spacetime diagram with trajectories")
print(f"  - Inset FD scatter plot")

