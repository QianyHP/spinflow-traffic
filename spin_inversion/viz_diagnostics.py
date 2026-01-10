"""
反演诊断可视化模块 (Inversion Diagnostics Visualization)

功能描述：
    生成用于诊断反演算法性能的详细图表。
    重点关注 FD 空间的匹配情况和初始物理场的数据特征。

图表内容：
    1. FD 拟合分析图: 展示观测数据点云、拟合的原型曲线以及点的质量评分分布。
    2. 初始场分析图: 展示观测密度场 ρ(x)、车道分布 L(x) 以及预估的密度梯度，用于验证物理守恒约束的输入数据质量。
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# 从命令行获取npz文件路径
npz_path = sys.argv[1] if len(sys.argv) > 1 else '../results/YTDJ/YTDJ_inverse_init.npz'
output_dir = os.path.dirname(npz_path)
save_prefix = os.path.splitext(os.path.basename(npz_path))[0].replace('_inverse_init', '')

print("="*80)
print("Generating FD Analysis Plots")
print("="*80)
print(f"Output Directory: {output_dir}")

# 1. 加载反演结果
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
dt = meta['dt']
cells = meta['cells']

print(f"  - Method: {meta.get('method', 'FD-EM')}")
print(f"  - FD points: {fd_points['n_points']}")
print(f"  - Cells: {cells}, dx={dx:.2f}m")

# 2. 加载观测数据并计算时空场
print("\n[2/6] Loading and gridding observation data...")
csv_path = meta.get('csv_path', '../data/raw_data/YTDJ/VTDJ_6-10.csv')
print(f"  - CSV Path: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    # Fallback for old path or relative path issues
    alt_path = '../data/processed_data/YTDJ/VTDJ_6-10.csv'
    print(f"  Warning: CSV not found at {csv_path}, trying {alt_path}")
    df = pd.read_csv(alt_path)
df = df.rename(columns={
    'vehicleID': 'veh',
    'time(s)': 't',
    'longitudinalDistance(m)': 'x',
    'lateralDistance(m)': 'y',
    'laneID': 'lane'
})
df['s'] = df['x']
t0, T_obs = 60, 60
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

print(f"  - Observation data: {K} steps x {M} cells")
print(f"  - rho_obs range: [{rho_norm.min():.4f}, {rho_norm.max():.4f}]")

# 3. 计算相混合参数
print("\n[3/6] Computing phase mixture parameters...")
from fd_model import TriangularFD
from phase_utils import spin_to_mixture_weights_softmax, mixture_fd_params

prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) 
              for p in prototypes_data]

pi = spin_to_mixture_weights_softmax(sx0, sy0, sz0, len(prototypes))
vf, Qcap, w, rho_jam = mixture_fd_params(pi, prototypes)

print(f"  - Phase weights π computed")
print(f"  - vf range: [{vf.min()*3.6:.1f}, {vf.max()*3.6:.1f}] km/h")
print(f"  - Qcap range: [{Qcap.min()*3600:.0f}, {Qcap.max()*3600:.0f}] veh/h")

# 4. 跳过phase_analysis图的生成（内容已合并到phase_map中）
print("\n[4/6] Skipping phase_analysis (merged into phase_map)...")

# 5. 生成图2：FD点云可视化
print("\n[5/6] Generating FD analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：FD散点+原型曲线
ax = axes[0]
k_data = fd_points['k']
q_data = fd_points['q']
score = fd_points['score']

scatter = ax.scatter(k_data, q_data, c=score, s=60, alpha=0.7, 
                    cmap='viridis', edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax, label='Score (lower=better)')

# 绘制原型曲线
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

# 子图2：Score分布
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
plt.tight_layout()
save_path_analysis = os.path.join(output_dir, f'{save_prefix}_analysis.png')
plt.savefig(save_path_analysis, dpi=200, bbox_inches='tight')
print(f"  --> {save_path_analysis}")
plt.close()

# 6. 生成报告
print("\n[6/6] Generating statistics report...")
spin_mag = np.sqrt(sx0**2 + sy0**2 + sz0**2)

report = []
report.append("="*80)
report.append("FD Inversion Results - Statistical Report")
report.append("="*80)
report.append("")
report.append("1. Method")
report.append(f"   - Approach: {meta.get('method', 'FD-EM')}")
report.append(f"   - Prototypes: {meta.get('n_prototypes', 3)}")
report.append("")
report.append("2. Sampling Quality")
report.append(f"   - FD points: {fd_points['n_points']}")
report.append(f"   - k range: [{k_data.min():.1f}, {k_data.max():.1f}] veh/km")
report.append(f"   - q range: [{q_data.min():.0f}, {q_data.max():.0f}] veh/h")
report.append(f"   - Mean score: {score.mean():.3f}")
report.append("")
report.append("3. Prototype Parameters")
for i, proto in enumerate(prototypes):
    report.append(f"   Prototype {i}:")
    report.append(f"     vf = {proto.vf*3.6:.2f} km/h")
    report.append(f"     w = {proto.w*3.6:.2f} km/h")
    report.append(f"     Q0 = {proto.Q0*3600:.0f} veh/h")
    report.append(f"     ρ_jam = {proto.rho_jam*1000:.1f} veh/km")
report.append("")
report.append("4. Initial Field Statistics")
report.append(f"   - rho0 mean: {rho0.mean():.4f}")
report.append(f"   - rho0 std: {rho0.std():.4f}")
report.append(f"   - Spin magnitude: {spin_mag.mean():.4f} ± {spin_mag.std():.4f}")
report.append("")
report.append("5. Generated Files")
report.append("5. Generated Files")
report.append(f"   - {output_dir}/{save_prefix}_analysis.png")
report.append(f"   - {output_dir}/{save_prefix}_experiment_report.txt")
report.append("")
report.append("="*80)

with open(f'{output_dir}/{save_prefix}_experiment_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

# for line in report:
#     print(line)

# 7. 跳过 density_comparison 图（已精简）
print("\n[7/9] Skipping density comparison (redundant)...")

# 8. 生成通用图2：initial_analysis
print("\n[8/9] Generating initial analysis...")

x_pos = np.arange(cells) * dx
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 子图1：初始密度场（观测）
ax = axes[0, 0]
ax.plot(x_pos, rho0, linewidth=2.5, alpha=0.8, color='#1f77b4', label='Observed Density (Edie)')
# ax.axvspan(100, 200, alpha=0.2, color='red', label='Bottleneck', zorder=1)
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('Initial Density Field ρ0(x) - From Observation', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 280)
ax.set_ylim(-0.05, 1.05)

# 子图2：自旋场分量
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
ax.set_xlim(0, 280)

# 子图3：车道数分布
ax = axes[1, 0]
ax.bar(x_pos, L, width=dx*0.8, alpha=0.6, color='steelblue', edgecolor='navy')
# ax.axvspan(100, 200, alpha=0.2, color='red')
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
# # ax.axvspan(100, 200, alpha=0.2, color='red')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('d(rho)/dx', fontsize=11)
ax.set_title('Density Gradient', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 280)

plt.tight_layout()
plt.tight_layout()
save_path_initial = os.path.join(output_dir, f'{save_prefix}_initial_analysis.png')
plt.savefig(save_path_initial, dpi=200, bbox_inches='tight')
print(f"  --> {save_path_initial}")
plt.close()


print("\n" + "="*80)
print("All FD analysis plots generated successfully!")
print(f"Generated {3} common plots + {2} FD-specific plots = 5 plots total")
print("="*80)