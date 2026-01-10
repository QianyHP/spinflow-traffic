"""
时空演化可视化模块 (Spacetime Visualization)

功能描述：
    生成高精度的交通流时空演化图谱 (Time-Space Diagram)。
    直观展示反演识别出的不同交通相态在时空二维平面上的分布与传播特征。

主要功能：
    1. 宏观密度场热力图: 基于 Edie 方法绘制的时空密度分布。
    2. 动态相态边界识别: 自动标记 Free、Critical、Congested 主导的时空区域。
    3. 细节放大展示: 对关键相态区域进行缩放展示，叠加微观轨迹采样证据，验证识别准确性。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from matplotlib.gridspec import GridSpec

import argparse
import os
import sys

# 从命令行获取npz文件路径
parser = argparse.ArgumentParser()
parser.add_argument("npz_path", type=str, nargs='?', default='../results/YTDJ/YTDJ_inverse_init.npz')
args = parser.parse_args()

npz_path = args.npz_path
output_dir = os.path.dirname(npz_path)
base_name = os.path.basename(npz_path).replace('_inverse_init.npz', '')

# 加载数据
data = np.load(npz_path, allow_pickle=True)
rho0 = data['rho0']
pi_final = data['pi_final']
fd_points = data['fd_points'].item()
meta = data['meta'].item()

# 从fd_points获取平行四边形列表（需要重新生成）
# 或者加载轨迹数据重新计算
dx = meta['dx']
cells = meta['cells']
x_pos = np.arange(cells) * dx

# 加载观测数据重新计算时空场
csv_path = meta.get('csv_path', '../data/raw_data/YTDJ/VTDJ_6-10.csv')
print(f"Loading observation data from: {csv_path}")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    # Fallback
    alt_path = '../data/processed_data/YTDJ/VTDJ_6-10.csv'
    print(f"Warning: CSV not found at {csv_path}, trying {alt_path}")
    df = pd.read_csv(alt_path)

df = df.rename(columns={
    'vehicleID': 'veh',
    'time(s)': 't',
    'longitudinalDistance(m)': 'x',
    'lateralDistance(m)': 'y',
    'laneID': 'lane'
})
df['s'] = df['x']
t0 = meta['t0']
T_obs = 60
df_window = df[(df['t'] >= t0) & (df['t'] < t0 + T_obs)].copy()

# 计算速度
fps = 24.0
dt_frame = 1.0/fps
df_window = df_window.sort_values(['veh','t'])
for vid, g in df_window.groupby('veh'):
    dx_diff = g['x'].diff().fillna(0.0)
    df_window.loc[g.index, 'vx'] = dx_diff/dt_frame
df_window['vs'] = df_window['vx']

# Edie栅格化
road_length = 280.0
dt_obs = 0.25
M = cells
t_grid = np.arange(t0, t0 + T_obs, dt_obs)
K = len(t_grid)
Theta = np.zeros((K, M))
Xi = np.zeros((K, M))

k_idx = np.clip(((df_window['t'] - t0) / dt_obs).values.astype(int), 0, K - 1)
i_idx = np.clip(((df_window['s'] - 0.0) / dx).values.astype(int), 0, M - 1)
vs = df_window['vs'].values

np.add.at(Theta, (k_idx, i_idx), dt_frame)
np.add.at(Xi, (k_idx, i_idx), np.abs(vs) * dt_frame)

# 估计车道数
lane_ids = sorted(df_window["lane"].dropna().unique().tolist())
L = np.zeros(M, dtype=int)
for ln in lane_ids:
    di = df_window[df_window["lane"] == ln]
    if len(di) > 0:
        i_idx_ln = np.clip(((di["s"] - 0.0) / dx).values.astype(int), 0, M - 1)
        counts = np.bincount(i_idx_ln, minlength=M)
        L += (counts >= 10).astype(int)
L = np.maximum(L, 1)

# 计算密度和速度场
area = dx * dt_obs
rho_all = Theta / area
q_all = Xi / area
Lx = L[np.newaxis, :]
rho_sle = rho_all / Lx

# 归一化
rho_jam_est = data['rho_jam_est']
rho_norm = np.clip(rho_sle / max(rho_jam_est, 1e-6), 0.0, 1.0)

# 识别三个典型区域 (动态识别物理边界)
dominant_phase = np.argmax(pi_final, axis=1)

def get_dynamic_region(indices, total_M, padding=5):
    if len(indices) < 3:
        # 如果样本太少，返回一个默认范围
        return (0, min(20, total_M))
    
    # 获取实际分布的边界
    start = np.min(indices)
    end = np.max(indices)
    
    # 为了绘图好看，增加少量 Padding，但不超过路网边界
    plot_start = max(0, start - padding)
    plot_end = min(total_M, end + padding)
    return (plot_start, plot_end)

# Free主导区域
free_cells = np.where(dominant_phase == 0)[0]
free_region = get_dynamic_region(free_cells, M)

# Critical主导区域
critical_cells = np.where(dominant_phase == 1)[0]
critical_region = get_dynamic_region(critical_cells, M)

# Congested主导区域（瓶颈）
congested_cells = np.where(dominant_phase == 2)[0]
congested_region = get_dynamic_region(congested_cells, M)

print(f"Identified Regions:")
print(f"  Free Flow:    x={free_region[0]*dx:.0f}-{free_region[1]*dx:.0f}m")
print(f"  Critical:     x={critical_region[0]*dx:.0f}-{critical_region[1]*dx:.0f}m")
print(f"  Congested:    x={congested_region[0]*dx:.0f}-{congested_region[1]*dx:.0f}m")

# 创建图形（模仿参考图布局）
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.5], hspace=0.25, wspace=0.3)

extent = [0, road_length, t0, t0 + T_obs]

# === 上面三个缩放图 ===
regions = [
    (free_region, "Free Flow Region", 0),
    (critical_region, "Critical Region", 1),
    (congested_region, "Congested Region (Bottleneck)", 2)
]

for idx, (region, title, col_idx) in enumerate(regions):
    ax = fig.add_subplot(gs[0, col_idx])
    
    # 提取区域数据
    x_start, x_end = region
    rho_region = rho_norm[:, x_start:x_end]
    extent_region = [x_start*dx, x_end*dx, t0, t0 + T_obs]
    
    # 绘制时空图
    im = ax.imshow(rho_region, aspect='auto', origin='lower', cmap='jet',
                   vmin=0, vmax=1, extent=extent_region, interpolation='bilinear')
    
    # 叠加平行四边形（找到该区域内的）
    for p_data in fd_points['x_center']:
        if x_start*dx <= p_data <= x_end*dx:
            # 绘制平行四边形中心点
            # 注意：这里简化为点，完整版需要从parallelogram对象获取corner
            pass
    
    # 叠加轨迹点（稀疏采样）
    df_region = df_window[(df_window['s'] >= x_start*dx) & (df_window['s'] <= x_end*dx)]
    if len(df_region) > 0:
        sample_ratio = max(1, len(df_region) // 3000)  # 最多3000个点
        df_sample = df_region.iloc[::sample_ratio]
        ax.scatter(df_sample['s'], df_sample['t'], s=0.8, c='white', alpha=0.8, rasterized=True)
    
    ax.set_xlabel('Position (m)', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_title(f'{title}\n({x_start*dx:.0f}-{x_end*dx:.0f}m)', fontsize=12, fontweight='bold')
    ax.set_xlim(x_start*dx, x_end*dx)
    ax.set_ylim(t0, t0 + T_obs)
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', rotation=270, labelpad=15, fontsize=10)

# === 下面：完整时空图 + FD插图 ===
ax_main = fig.add_subplot(gs[1, :])

# 绘制完整密度场
im_main = ax_main.imshow(rho_norm, aspect='auto', origin='lower', cmap='jet',
                         vmin=0, vmax=1, extent=extent, interpolation='bilinear')

# 叠加轨迹点（稀疏）
sample_ratio_main = max(1, len(df_window) // 10000)
df_sample_main = df_window.iloc[::sample_ratio_main]
ax_main.scatter(df_sample_main['s'], df_sample_main['t'], 
               s=0.8, c='white', alpha=0.8, rasterized=True)

# 标记三个区域
for region, title, color in [
    (free_region, 'Free', 'blue'),
    (critical_region, 'Critical', 'green'),
    (congested_region, 'Congested', 'red')
]:
    x_start, x_end = region
    rect = patches.Rectangle((x_start*dx, t0), (x_end-x_start)*dx, T_obs,
                             linewidth=2.5, edgecolor=color, facecolor='none',
                             linestyle='--', alpha=0.8, label=title)
    ax_main.add_patch(rect)

ax_main.set_xlabel('Position (m)', fontsize=13, fontweight='bold')
ax_main.set_ylabel('Time (s)', fontsize=13, fontweight='bold')
ax_main.set_title('Spacetime Diagram with Phase Regions', fontsize=14, fontweight='bold')
ax_main.set_xlim(0, road_length)
ax_main.set_ylim(t0, t0 + T_obs)
ax_main.legend(loc='upper right', fontsize=11)

# colorbar
cbar_main = plt.colorbar(im_main, ax=ax_main)
cbar_main.set_label('Normalized Density', rotation=270, labelpad=20, fontsize=12)

# === 插图：FD散点图 ===
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_inset = inset_axes(ax_main, width="25%", height="30%", loc='upper left',
                      bbox_to_anchor=(0.02, 0.02, 0.96, 0.96), bbox_transform=ax_main.transAxes)

# 从prototypes重建
from fd_model import TriangularFD
prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) 
              for p in data['prototypes']]

k_data = fd_points['k']
q_data = fd_points['q']
score = fd_points['score']

ax_inset.scatter(k_data, q_data, c=score, s=20, alpha=0.6, 
                cmap='viridis_r', edgecolors='black', linewidth=0.3)

# 绘制原型曲线
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

