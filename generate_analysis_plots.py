"""
生成标准实验的综合分析图表
包括密度对比、时空演化、自旋场分析等
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 确保results文件夹存在
os.makedirs('results', exist_ok=True)

print("="*80)
print("Generating Analysis Plots")
print("="*80)

# 1. 加载反演结果
print("\n[1/6] Loading inversion results...")
data = np.load('results/optimal_inverse_init.npz', allow_pickle=True)
rho0 = data['rho0']
sx0 = data['sx0']
sy0 = data['sy0']
sz0 = data['sz0']
meta = data['meta'].item()

dx = meta['dx']
cells = meta['cells']
dt_model = meta['dt_model']

print(f"  - Spatial resolution: {dx:.2f}m ({cells} cells)")
print(f"  - Time step: {dt_model:.4f}s")

# 2. 重新计算观测数据用于对比
print("\n[2/6] Recomputing observation data...")
df = pd.read_csv('data/VTDJ_6-10.csv')
df = df.rename(columns={
    'vehicleID': 'veh',
    'time(s)': 't',
    'longitudinalDistance(m)': 'x',
    'lateralDistance(m)': 'y',
    'laneID': 'lane'
})

# 东向
df['s'] = df['x']
t0, T_obs = 60, 45
df_window = df[(df['t'] >= t0) & (df['t'] < t0 + T_obs)].copy()

# 估算速度
fps = 24.0
dt_frame = 1.0/fps
df_window = df_window.sort_values(['veh','t'])
for vid, g in df_window.groupby('veh'):
    dx_diff = g['x'].diff().fillna(0.0)
    df_window.loc[g.index, 'vx'] = dx_diff/dt_frame
df_window['vs'] = df_window['vx']

# Edie栅格化
# 实际数据长度约283m，使用280m以覆盖主要区域
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
        L += (counts >= 20).astype(int)
L = np.maximum(L, 1)

# 计算密度（SLE）
area = dx * dt_obs
rho_all = Theta / area
q_all = Xi / area
Lx = L[np.newaxis, :]
rho_sle = rho_all / Lx
q_sle = q_all / Lx

# 归一化
rho_jam_est = np.quantile(rho_sle[rho_sle>0], 0.99) if np.any(rho_sle>0) else 1.0
rho_norm = np.clip(rho_sle / max(rho_jam_est, 1e-6), 0.0, 1.0)

print(f"  - Observation data: {K} time steps, {M} spatial cells")
print(f"  - Density range: [{rho_norm.min():.4f}, {rho_norm.max():.4f}]")

# 3. 生成图1：密度对比（使用jet colormap）
print("\n[3/6] Generating density comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 设置extent让x轴显示实际位置（米）
extent = [0, road_length, t0, t0 + T_obs]

# 原始SLE密度（使用jet colormap）
im1 = axes[0].imshow(rho_sle, aspect='auto', origin='lower', cmap='jet', 
                     extent=extent, interpolation='bilinear')
axes[0].set_title(f'Raw SLE Density (veh/m)\nMax={rho_sle.max():.3f}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Position (m)', fontsize=11)
axes[0].set_ylabel('Time (s)', fontsize=11)
axes[0].set_xlim(0, road_length)
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('Density (veh/m)', rotation=270, labelpad=15)

# 归一化密度（使用jet colormap）
im2 = axes[1].imshow(rho_norm, aspect='auto', origin='lower', cmap='jet', 
                     vmin=0, vmax=1, extent=extent, interpolation='bilinear')
axes[1].set_title(f'Normalized Density (rho/rho_jam)\nJam={rho_jam_est:.3f}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Position (m)', fontsize=11)
axes[1].set_ylabel('Time (s)', fontsize=11)
axes[1].set_xlim(0, road_length)
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label('Normalized Density', rotation=270, labelpad=15)

# 速度场（使用jet colormap，强调速度差异）
velocity = np.zeros_like(rho_sle)
valid_mask = rho_sle > 1e-6
velocity[valid_mask] = q_sle[valid_mask] / rho_sle[valid_mask]
# 速度单位转换：m/s 转为 km/h 方便理解
velocity_kmh = velocity * 3.6
im3 = axes[2].imshow(velocity_kmh, aspect='auto', origin='lower', cmap='jet',
                     extent=extent, interpolation='bilinear', vmin=0, vmax=80)
axes[2].set_title('Space-Mean Speed (km/h)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Position (m)', fontsize=11)
axes[2].set_ylabel('Time (s)', fontsize=11)
axes[2].set_xlim(0, road_length)
cbar3 = plt.colorbar(im3, ax=axes[2])
cbar3.set_label('Speed (km/h)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('results/density_comparison.png', dpi=200, bbox_inches='tight')
print("  --> results/density_comparison.png")
plt.close()

# 4. 生成图2：初始密度剖面详细对比
print("\n[4/6] Generating initial density analysis...")
x_pos = np.arange(cells) * dx

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 子图1：初始密度对比
ax = axes[0, 0]
ax.plot(x_pos, rho_norm[0], label='Observed (t=60s)', linewidth=2, alpha=0.8)
ax.plot(x_pos, rho0, '--', label='Inverted rho0', linewidth=2, alpha=0.8)
ax.axvspan(100, 200, alpha=0.2, color='red', label='Bottleneck')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('Initial Density Profile Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 280)

# 子图2：自旋场分量
ax = axes[0, 1]
spin_mag = np.sqrt(sx0**2 + sy0**2 + sz0**2)
ax.plot(x_pos, sx0, label='sx', alpha=0.7, linewidth=1.5)
ax.plot(x_pos, sy0, label='sy', alpha=0.7, linewidth=1.5)
ax.plot(x_pos, sz0, label='sz', alpha=0.7, linewidth=1.5)
ax.plot(x_pos, spin_mag, 'k--', label='|s|', linewidth=2)
ax.axvspan(100, 200, alpha=0.2, color='red')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Spin Components', fontsize=11)
ax.set_title('Spin Field Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 280)

# 子图3：车道数分布
ax = axes[1, 0]
ax.bar(x_pos, L, width=dx*0.8, alpha=0.6, color='steelblue', edgecolor='navy')
ax.axvspan(100, 200, alpha=0.2, color='red')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Number of Lanes', fontsize=11)
ax.set_title('Lane Count Profile L(x)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(0, 280)
ax.set_ylim(0, 6)

# 子图4：密度梯度
ax = axes[1, 1]
grad_rho = np.gradient(rho0)
ax.plot(x_pos, grad_rho, color='darkblue', linewidth=2)
ax.axvspan(100, 200, alpha=0.2, color='red')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('d(rho)/dx', fontsize=11)
ax.set_title('Density Gradient', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 280)

plt.tight_layout()
plt.savefig('results/initial_analysis.png', dpi=200, bbox_inches='tight')
print("  --> results/initial_analysis.png")
plt.close()

# 5. 生成图3：瓶颈区域特写
print("\n[5/6] Generating bottleneck analysis...")
bottleneck_start_idx = int(100 / dx)
bottleneck_end_idx = int(200 / dx)
bottleneck_range = range(bottleneck_start_idx, bottleneck_end_idx)
x_bottleneck = x_pos[bottleneck_range]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 瓶颈区密度演化（使用jet colormap，显示位置坐标）
ax = axes[0]
rho_bottleneck = rho_norm[:, bottleneck_start_idx:bottleneck_end_idx]
extent_bottleneck = [100, 200, t0, t0 + T_obs]
im = ax.imshow(rho_bottleneck, aspect='auto', origin='lower', cmap='jet', 
               vmin=0, vmax=1, extent=extent_bottleneck, interpolation='bilinear')
ax.set_title('Bottleneck Density Evolution (100-200m)', fontsize=12, fontweight='bold')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Time (s)', fontsize=11)
ax.set_xlim(100, 200)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Density', rotation=270, labelpad=15)

# 瓶颈区初始剖面
ax = axes[1]
ax.plot(x_bottleneck, rho_norm[0, bottleneck_range], 'o-', label='Observed', markersize=4, linewidth=2)
ax.plot(x_bottleneck, rho0[bottleneck_range], 's--', label='Inverted', markersize=4, linewidth=2)
ax.fill_between(x_bottleneck, 0, 1, alpha=0.1, color='red')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('Bottleneck Initial Density Profile', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('results/bottleneck_analysis.png', dpi=200, bbox_inches='tight')
print("  --> results/bottleneck_analysis.png")
plt.close()

# 6. 生成统计报告
print("\n[6/6] Generating statistics report...")
report = []
report.append("="*80)
report.append("Experiment Results - Statistical Report")
report.append("="*80)
report.append("")
report.append("1. Spatial Configuration")
report.append(f"   - Road length: {road_length}m")
report.append(f"   - Spatial cells: {cells}")
report.append(f"   - Spatial resolution: {dx:.2f}m/cell")
report.append(f"   - Bottleneck region: 100-200m ({bottleneck_end_idx - bottleneck_start_idx} cells)")
report.append("")
report.append("2. Temporal Configuration")
report.append(f"   - Start time: {t0}s")
report.append(f"   - Observation duration: {T_obs}s")
report.append(f"   - Time resolution: {dt_obs}s")
report.append(f"   - Time steps: {K}")
report.append("")
report.append("3. Density Field Statistics")
report.append(f"   - Initial density mean: {rho0.mean():.4f}")
report.append(f"   - Initial density std: {rho0.std():.4f}")
report.append(f"   - Initial density range: [{rho0.min():.4f}, {rho0.max():.4f}]")
report.append(f"   - Bottleneck density mean: {rho0[bottleneck_range].mean():.4f}")
report.append(f"   - Bottleneck density std: {rho0[bottleneck_range].std():.4f}")
report.append("")
report.append("4. Spin Field Statistics")
spin_mag = np.sqrt(sx0**2 + sy0**2 + sz0**2)
report.append(f"   - Average magnitude: {spin_mag.mean():.4f}")
report.append(f"   - Magnitude std: {spin_mag.std():.4f}")
report.append(f"   - Magnitude range: [{spin_mag.min():.4f}, {spin_mag.max():.4f}]")
report.append("")
report.append("5. Observation Data Statistics")
report.append(f"   - Raw density max: {rho_sle.max():.4f} veh/m")
report.append(f"   - Estimated jam density: {rho_jam_est:.4f} veh/m")
report.append(f"   - Normalized density mean: {rho_norm.mean():.4f}")
report.append(f"   - Lane count range: {L.min()}-{L.max()}")
report.append("")
report.append("6. Generated Files")
report.append("   - results/optimal_inverse_init.npz (inversion data)")
report.append("   - results/density_comparison.png (density analysis)")
report.append("   - results/initial_analysis.png (comprehensive analysis)")
report.append("   - results/bottleneck_analysis.png (bottleneck details)")
report.append("   - results/experiment_report.txt (this report)")
report.append("")
report.append("="*80)

# 保存报告
with open('results/experiment_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

# 打印报告
for line in report:
    print(line)

print("\n" + "="*80)
print("All analysis plots generated successfully!")
print("="*80)

