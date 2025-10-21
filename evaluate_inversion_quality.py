"""
反演质量定量评估脚本
计算反演结果还原实际观测数据的准确度
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 手动实现评估指标
def r2_score(y_true, y_pred):
    """R² 决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def mean_squared_error(y_true, y_pred):
    """均方误差"""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

print("="*80)
print("反演质量定量评估")
print("="*80)

# 1. 加载反演结果和观测数据
print("\n[1/5] 加载反演结果...")
data = np.load('results/optimal_inverse_init.npz', allow_pickle=True)
rho0_inverted = data['rho0']
rho0_observed = data['rho_obs_t0']  # 直接使用反演程序保存的观测数据
meta = data['meta'].item()
dx = meta['dx']
cells = meta['cells']

print(f"  反演和观测数据已从npz文件加载")

# 2. 加载完整观测时空数据用于可视化
print("\n[2/5] 加载完整观测数据用于可视化...")
df = pd.read_csv('data/VTDJ_6-10.csv')
df = df.rename(columns={
    'vehicleID': 'veh',
    'time(s)': 't',
    'longitudinalDistance(m)': 'x',
    'lateralDistance(m)': 'y',
    'laneID': 'lane'
})

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

# 估计车道数并计算SLE密度
lane_ids = sorted(df_window["lane"].dropna().unique().tolist())
L = np.zeros(M, dtype=int)
for ln in lane_ids:
    di = df_window[df_window["lane"] == ln]
    if len(di) > 0:
        i_idx_ln = np.clip(((di["s"] - 0.0) / dx).values.astype(int), 0, M - 1)
        counts = np.bincount(i_idx_ln, minlength=M)
        L += (counts >= 20).astype(int)
L = np.maximum(L, 1)

area = dx * dt_obs
rho_all = Theta / area
Lx = L[np.newaxis, :]
rho_sle = rho_all / Lx

# 归一化（用于后续可视化）
rho_jam_est = np.quantile(rho_sle[rho_sle>0], 0.99) if np.any(rho_sle>0) else 1.0
rho_norm = np.clip(rho_sle / max(rho_jam_est, 1e-6), 0.0, 1.0)

# rho0_observed已从npz文件加载，与反演使用的完全一致

print(f"  观测密度(t=60s)范围: [{rho0_observed.min():.4f}, {rho0_observed.max():.4f}]")
print(f"  观测密度(t=60s)均值: {rho0_observed.mean():.4f}")
print(f"  反演密度范围: [{rho0_inverted.min():.4f}, {rho0_inverted.max():.4f}]")
print(f"  反演密度均值: {rho0_inverted.mean():.4f}")

# 3. 计算全局指标
print("\n[3/5] 计算全局质量指标...")

# R² (决定系数) - 越接近1越好
r2 = r2_score(rho0_observed, rho0_inverted)

# RMSE (均方根误差) - 越小越好
rmse = np.sqrt(mean_squared_error(rho0_observed, rho0_inverted))

# MAE (平均绝对误差) - 越小越好
mae = mean_absolute_error(rho0_observed, rho0_inverted)

# MAPE (平均绝对百分比误差) - 考虑相对误差
# 避免除零，只在观测值>0.01时计算
mask = rho0_observed > 0.01
if mask.sum() > 0:
    mape = np.mean(np.abs((rho0_observed[mask] - rho0_inverted[mask]) / rho0_observed[mask])) * 100
else:
    mape = np.nan

# 相关系数
correlation = np.corrcoef(rho0_observed, rho0_inverted)[0, 1]

# 解释方差（1 - 残差方差/观测方差）
explained_variance = 1 - np.var(rho0_observed - rho0_inverted) / np.var(rho0_observed)

print(f"\nGlobal Quality Metrics:")
print(f"  R-squared:                {r2:.4f}  (1.0=perfect)")
print(f"  Correlation:              {correlation:.4f}  (1.0=perfect)")
print(f"  Explained Variance:       {explained_variance:.4f}  ({explained_variance*100:.2f}%)")
print(f"  RMSE:                     {rmse:.4f}")
print(f"  MAE:                      {mae:.4f}")
if not np.isnan(mape):
    print(f"  MAPE:                     {mape:.2f}%")

# 4. 分区域评估
print("\n[4/5] Regional evaluation...")

# 定义区域
bottleneck_start = int(100 / dx)
bottleneck_end = int(200 / dx)

regions = {
    'Full Road (0-280m)': slice(0, cells),
    'Upstream (0-100m)': slice(0, bottleneck_start),
    'Bottleneck (100-200m)': slice(bottleneck_start, bottleneck_end),
    'Downstream (200-280m)': slice(bottleneck_end, cells)
}

regional_metrics = {}
for name, region_slice in regions.items():
    obs = rho0_observed[region_slice]
    inv = rho0_inverted[region_slice]
    
    if len(obs) > 1:
        r2_region = r2_score(obs, inv)
        rmse_region = np.sqrt(mean_squared_error(obs, inv))
        mae_region = mean_absolute_error(obs, inv)
        
        regional_metrics[name] = {
            'R²': r2_region,
            'RMSE': rmse_region,
            'MAE': mae_region,
            'n_cells': len(obs)
        }

print("\nRegional Quality Metrics:")
print(f"{'Region':<25} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'Cells':>6}")
print("-" * 65)
for name, metrics in regional_metrics.items():
    print(f"{name:<25} {metrics['R²']:>8.4f} {metrics['RMSE']:>8.4f} "
          f"{metrics['MAE']:>8.4f} {metrics['n_cells']:>6d}")

# 5. 生成评估可视化
print("\n[5/5] 生成评估可视化...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

x_pos = np.arange(cells) * dx

# 子图1: 直接对比
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(x_pos, rho0_observed, 'o-', label='Observed', markersize=3, linewidth=1.5, alpha=0.7)
ax1.plot(x_pos, rho0_inverted, 's--', label='Inverted', markersize=3, linewidth=1.5, alpha=0.7)
ax1.axvspan(100, 200, alpha=0.15, color='red', label='Bottleneck')
ax1.set_xlabel('Position (m)', fontsize=11)
ax1.set_ylabel('Normalized Density', fontsize=11)
ax1.set_title('Direct Comparison: Observed vs Inverted', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 280)

# 子图2: 散点图 (观测 vs 反演)
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(rho0_observed, rho0_inverted, alpha=0.6, s=30)
# 1:1线
max_val = max(rho0_observed.max(), rho0_inverted.max())
ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect fit')
ax2.set_xlabel('Observed Density', fontsize=10)
ax2.set_ylabel('Inverted Density', fontsize=10)
ax2.set_title(f'Scatter Plot\nR²={r2:.3f}', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# 子图3: 残差分布
ax3 = fig.add_subplot(gs[1, 0])
residuals = rho0_inverted - rho0_observed
ax3.plot(x_pos, residuals, 'o-', markersize=3, linewidth=1.5, color='darkred')
ax3.axhline(0, color='k', linestyle='--', linewidth=1)
ax3.axhline(rmse, color='orange', linestyle='--', linewidth=1, label=f'RMSE={rmse:.3f}')
ax3.axhline(-rmse, color='orange', linestyle='--', linewidth=1)
ax3.axvspan(100, 200, alpha=0.15, color='red')
ax3.set_xlabel('Position (m)', fontsize=10)
ax3.set_ylabel('Residual (Inv - Obs)', fontsize=10)
ax3.set_title('Residual Distribution', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 280)

# 子图4: 残差直方图
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='navy')
ax4.axvline(0, color='r', linestyle='--', linewidth=2)
ax4.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=2, 
            label=f'Mean={residuals.mean():.4f}')
ax4.set_xlabel('Residual', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title('Residual Histogram', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# 子图5: 累积误差
ax5 = fig.add_subplot(gs[1, 2])
cumulative_error = np.cumsum(np.abs(residuals))
ax5.plot(x_pos, cumulative_error, linewidth=2, color='darkgreen')
ax5.axvspan(100, 200, alpha=0.15, color='red')
ax5.set_xlabel('Position (m)', fontsize=10)
ax5.set_ylabel('Cumulative Absolute Error', fontsize=10)
ax5.set_title('Cumulative Error', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 280)

# 子图6: 相对误差（百分比）
ax6 = fig.add_subplot(gs[2, 0])
relative_error = np.where(rho0_observed > 0.01, 
                         (rho0_inverted - rho0_observed) / rho0_observed * 100, 
                         0)
ax6.plot(x_pos, relative_error, 'o-', markersize=3, linewidth=1.5, color='purple')
ax6.axhline(0, color='k', linestyle='--', linewidth=1)
ax6.axhline(20, color='orange', linestyle=':', linewidth=1, label='±20%')
ax6.axhline(-20, color='orange', linestyle=':', linewidth=1)
ax6.axvspan(100, 200, alpha=0.15, color='red')
ax6.set_xlabel('Position (m)', fontsize=10)
ax6.set_ylabel('Relative Error (%)', fontsize=10)
ax6.set_title('Relative Error Distribution', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 280)
ax6.set_ylim(-100, 100)

# 子图7: 分区域指标对比
ax7 = fig.add_subplot(gs[2, 1:])
region_names = list(regional_metrics.keys())
r2_values = [regional_metrics[r]['R²'] for r in region_names]
rmse_values = [regional_metrics[r]['RMSE'] for r in region_names]

x_bar = np.arange(len(region_names))
width = 0.35
bars1 = ax7.bar(x_bar - width/2, r2_values, width, label='R²', alpha=0.8, color='steelblue')
ax7_twin = ax7.twinx()
bars2 = ax7_twin.bar(x_bar + width/2, rmse_values, width, label='RMSE', alpha=0.8, color='coral')

ax7.set_xlabel('Region', fontsize=10)
ax7.set_ylabel('R² Score', fontsize=10, color='steelblue')
ax7_twin.set_ylabel('RMSE', fontsize=10, color='coral')
ax7.set_title('Regional Quality Metrics', fontsize=11, fontweight='bold')
ax7.set_xticks(x_bar)
ax7.set_xticklabels([r.split('(')[0].strip() for r in region_names], rotation=15, ha='right')
ax7.tick_params(axis='y', labelcolor='steelblue')
ax7_twin.tick_params(axis='y', labelcolor='coral')
ax7.axhline(0.8, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax7.legend(loc='upper left', fontsize=9)
ax7_twin.legend(loc='upper right', fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')
ax7.set_ylim(0, 1)

plt.savefig('results/inversion_quality_evaluation.png', dpi=200, bbox_inches='tight')
print("  --> results/inversion_quality_evaluation.png")
plt.close()

# 生成文本报告
print("\n[6/6] 生成评估报告...")
report = []
report.append("="*80)
report.append("Inversion Quality Evaluation Report")
report.append("="*80)
report.append("")
report.append("1. Global Quality Metrics")
report.append(f"   R-squared:                {r2:.6f}")
report.append(f"   Correlation:              {correlation:.6f}")
report.append(f"   Explained Variance:       {explained_variance:.6f} ({explained_variance*100:.2f}%)")
report.append(f"   RMSE:                     {rmse:.6f}")
report.append(f"   MAE:                      {mae:.6f}")
if not np.isnan(mape):
    report.append(f"   MAPE:                     {mape:.2f}%")
report.append("")
report.append("   Interpretation:")
report.append(f"   - Inversion captured {explained_variance*100:.2f}% of observed variance")
if r2 > 0.9:
    quality_text = "Excellent"
elif r2 > 0.7:
    quality_text = "Good"
elif r2 > 0.5:
    quality_text = "Fair"
else:
    quality_text = "Poor"
report.append(f"   - R2={r2:.3f} indicates {quality_text} fit")
report.append("")
report.append("2. Regional Quality Metrics")
report.append(f"   {'Region':<30} {'R2':>10} {'RMSE':>10} {'MAE':>10}")
report.append("   " + "-"*65)
for name, metrics in regional_metrics.items():
    report.append(f"   {name:<30} {metrics['R²']:>10.4f} {metrics['RMSE']:>10.4f} {metrics['MAE']:>10.4f}")
report.append("")
report.append("3. Residual Statistics")
report.append(f"   Mean residual:       {residuals.mean():>10.6f}  (close to 0 = no bias)")
report.append(f"   Residual std:        {residuals.std():>10.6f}")
report.append(f"   Max positive:        {residuals.max():>10.6f}  (at {x_pos[residuals.argmax()]:.1f}m)")
report.append(f"   Max negative:        {residuals.min():>10.6f}  (at {x_pos[residuals.argmin()]:.1f}m)")
report.append("")
report.append("4. Overall Grade")
if r2 > 0.9 and rmse < 0.1:
    grade = "Excellent (A)"
elif r2 > 0.7 and rmse < 0.15:
    grade = "Good (B)"
elif r2 > 0.5 and rmse < 0.2:
    grade = "Fair (C)"
else:
    grade = "Needs Improvement (D)"
report.append(f"   Grade: {grade}")
report.append("")
report.append("="*80)

with open('results/inversion_quality_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print("\nEvaluation Report saved to: results/inversion_quality_report.txt")
print("Evaluation plot saved to: results/inversion_quality_evaluation.png")

print("\n" + "="*80)
print("QUALITY SUMMARY")
print("="*80)
print(f"R-squared: {r2:.4f}")
print(f"Explained Variance: {explained_variance*100:.2f}%")
print(f"Grade: {grade}")
print("="*80)

