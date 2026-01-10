"""
反演质量评估模块 (Inversion Quality Evaluator)

功能描述：
    对 SpinFlow 反演结果进行定量分析和质量评级。
    生成详细的统计报告，验证反演结果的物理合理性与数据拟合精度。

主要指标：
    1. 拟合优度: FD 曲线的 R² 和 RMSE (流量/速度)。
    2. 数理统计: 物理熵 H(x) (相态纯度) 和 结构正则性 G(x) (空间平滑度)。
    3. 质量评级: 基于上述指标综合给出的 A/B/C/D 评级。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, nargs='?', default='../results/YTDJ/YTDJ_inverse_init.npz')
args_eval = parser.parse_args()

# 加载反演结果
data = np.load(args_eval.data_path, allow_pickle=True)
rho0 = data['rho0']
sx0 = data['sx0']
sy0 = data['sy0']
sz0 = data['sz0']
pi_final = data['pi_final']
prototypes_data = data['prototypes']
fd_points = data['fd_points'].item()
history = data['history'].item()
meta = data['meta'].item()

# 根据输入文件路径确定报告保存名
import os
output_dir = os.path.dirname(args_eval.data_path)
base_name = os.path.basename(args_eval.data_path).replace('_inverse_init.npz', '')
report_txt_path = os.path.join(output_dir, f'{base_name}_report.txt')
report_img_path = os.path.join(output_dir, f'{base_name}_quality.png')

from fd_model import TriangularFD
prototypes = [TriangularFD(p['vf'], p['w'], p['rho_jam'], p['Q0']) 
              for p in prototypes_data]

dx = meta['dx']
cells = meta['cells']
x_pos = np.arange(cells) * dx

# === 核心评估1：FD匹配质量 ===
print("="*80)
print("Spin Field Inversion Quality Assessment")
print("="*80)
print(f"Method: {meta.get('method', 'Unknown')}")
print(f"Inversion Target: {meta.get('inversion_target', 'Unknown')}")

# 计算每个FD点的预测误差
k_points = fd_points['k'] / 1000.0  # veh/m
q_points = fd_points['q'] / 3600.0  # veh/s
v_points = fd_points['v'] / 3.6     # m/s
x_points = fd_points['x_center']
n_points = len(k_points)

q_pred_list = []
v_pred_list = []
q_errors = []
v_errors = []

for p_idx in range(n_points):
    k_p = k_points[p_idx]
    q_p = q_points[p_idx]
    v_p = v_points[p_idx]
    x_p = x_points[p_idx]
    
    # 找到对应cell
    cell_idx = int(np.clip(x_p / dx, 0, cells-1))
    
    # 混合FD预测
    q_pred = 0.0
    for g, proto in enumerate(prototypes):
        q_g = proto.flow(np.array([k_p]))[0]
        q_pred += pi_final[cell_idx, g] * q_g
    
    v_pred = q_pred / k_p if k_p > 0 else 0.0
    
    q_pred_list.append(q_pred)
    v_pred_list.append(v_pred)
    q_errors.append((q_pred - q_p) * 3600)  # veh/h
    v_errors.append((v_pred - v_p) * 3.6)   # km/h

q_pred_list = np.array(q_pred_list)
v_pred_list = np.array(v_pred_list)
q_errors = np.array(q_errors)
v_errors = np.array(v_errors)

# 计算指标
rmse_q = np.sqrt(np.mean(q_errors**2))
rmse_v = np.sqrt(np.mean(v_errors**2))
mae_q = np.mean(np.abs(q_errors))
mae_v = np.mean(np.abs(v_errors))

# R2
ss_tot_q = np.sum((q_points*3600 - (q_points*3600).mean())**2)
ss_res_q = np.sum(q_errors**2)
r2_q = 1 - ss_res_q / (ss_tot_q + 1e-12)

ss_tot_v = np.sum((v_points*3.6 - (v_points*3.6).mean())**2)
ss_res_v = np.sum(v_errors**2)
r2_v = 1 - ss_res_v / (ss_tot_v + 1e-12)



# === 评估2：相权重合理性 ===
dominant_phase = np.argmax(pi_final, axis=1)
phase_names = ['Free', 'Critical', 'Congested']

# Calculate Entropy H(x) (Section 7.5)
epsilon = 1e-10
pi_safe = np.clip(pi_final, epsilon, 1.0)
H_x = -np.sum(pi_final * np.log(pi_safe), axis=1) # [cells]
mean_H = np.mean(H_x)
std_H = np.std(H_x)

# Calculate Structure Regularity G_pi (Section 7.5)
# G_pi = sum ||pi_{i+1} - pi_i||^2
diff_pi = pi_final[1:] - pi_final[:-1] # [cells-1, 3]
G_pi_local = np.sum(diff_pi**2, axis=1) # [cells-1]
G_pi_total = np.sum(G_pi_local)
G_pi_mean = np.mean(G_pi_local) # normalized per cell interface


# === 评估3：原型参数合理性 ===


# === 评估4：收敛质量 ===


print("="*80)

# === 可视化 ===
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# === Row 1: Flow Analysis ===
# 图1：FD流量预测对比
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(q_points*3600, q_pred_list*3600, c=x_points, s=60, alpha=0.7,
           cmap='viridis', edgecolors='black', linewidth=0.5)
ax1.plot([0, 2500], [0, 2500], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Observed Flow (veh/h)', fontsize=12)
ax1.set_ylabel('Predicted Flow (veh/h)', fontsize=12)
ax1.set_title(f'Flow Prediction Quality\nRMSE={rmse_q:.1f} veh/h, R2={r2_q:.4f}', 
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 图2：预测误差分布（流量）
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(q_errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(q_errors.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={q_errors.mean():.1f}')
ax2.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero')
ax2.set_xlabel('Flow Error (veh/h)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title(f'Flow Error Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 图3：FD点在(k,q)空间的拟合
ax3 = fig.add_subplot(gs[0, 2])
k_data = fd_points['k']
q_data = fd_points['q']
scatter = ax3.scatter(k_data, q_data, c=fd_points['score'], s=60, alpha=0.7,
                     cmap='viridis_r', edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax3, label='Score (lower=better)')

# 绘制原型曲线
rho_range = np.linspace(0, 0.35, 300)
k_range = rho_range * 1000
colors = ['blue', 'green', 'red']
for i, (proto, color) in enumerate(zip(prototypes, colors)):
    q_curve = proto.flow(rho_range) * 3600
    ax3.plot(k_range, q_curve, color=color, linewidth=2.5, linestyle='--',
            label=f'Prototype {i}', alpha=0.8)

ax3.set_xlabel('Density (veh/km)', fontsize=12)
ax3.set_ylabel('Flow (veh/h)', fontsize=12)
ax3.set_title(f'FD Points + Prototypes (N={n_points})', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9, loc='lower center')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, max(k_data.max()*1.1, 350))
ax3.set_ylim(0, max(q_data.max()*1.1, 2500))

# === Row 2: Velocity Analysis ===
# 图4：FD速度预测对比
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(v_points*3.6, v_pred_list*3.6, c=x_points, s=60, alpha=0.7,
           cmap='viridis', edgecolors='black', linewidth=0.5)
ax4.plot([0, 50], [0, 50], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Observed Velocity (km/h)', fontsize=12)
ax4.set_ylabel('Predicted Velocity (km/h)', fontsize=12)
ax4.set_title(f'Velocity Prediction Quality\nRMSE={rmse_v:.2f} km/h, R2={r2_v:.4f}', 
             fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

# 图5：预测误差分布（速度）
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(v_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
ax5.axvline(v_errors.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={v_errors.mean():.2f}')
ax5.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero')
ax5.set_xlabel('Velocity Error (km/h)', fontsize=12)
ax5.set_ylabel('Count', fontsize=12)
ax5.set_title(f'Velocity Error Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

# 图6：Loss收敛曲线
ax6 = fig.add_subplot(gs[1, 2])
iterations = np.arange(1, len(history['loss']) + 1)
ax6.plot(iterations, history['loss'], 'o-', label='Total Loss', 
        linewidth=2.5, markersize=6, color='purple')
ax6.plot(iterations, history['loss_fd_q'], 's--', label='Flow Loss', 
        linewidth=2, markersize=4, alpha=0.8, color='blue')
ax6.plot(iterations, history['loss_fd_v'], '^--', label='Velocity Loss', 
        linewidth=2, markersize=4, alpha=0.8, color='green')
ax6.set_xlabel('EM Iteration', fontsize=12)
ax6.set_ylabel('Normalized Loss', fontsize=12)
ax6.set_title(f'Convergence (iter {len(iterations)})', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

# === Row 3: Physical Interpretation ===
# 图7：相权重空间分布
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(x_pos, pi_final[:, 0], label=f'π_Free (vf={prototypes[0].vf*3.6:.1f})', 
        linewidth=2.5, alpha=0.8)
ax7.plot(x_pos, pi_final[:, 1], label=f'π_Critical (vf={prototypes[1].vf*3.6:.1f})', 
        linewidth=2.5, alpha=0.8)
ax7.plot(x_pos, pi_final[:, 2], label=f'π_Congested (vf={prototypes[2].vf*3.6:.1f})', 
        linewidth=2.5, alpha=0.8)
ax7.set_xlabel('Position (m)', fontsize=12)
ax7.set_ylabel('Phase Weight π_g(x)', fontsize=12)
ax7.set_title('Inverted Phase Distribution', fontsize=13, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)
ax7.set_xlim(0, 280)
ax7.set_ylim(0, 1)

# 图8：自旋场分布 (Removed |s|)
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(x_pos, sx0, label='sx', linewidth=2, alpha=0.7)
ax8.plot(x_pos, sy0, label='sy', linewidth=2, alpha=0.7)
ax8.plot(x_pos, sz0, label='sz', linewidth=2, alpha=0.7)
ax8.set_xlabel('Position (m)', fontsize=12)
ax8.set_ylabel('Spin Components', fontsize=12)
ax8.set_title('Inverted Spin Field', fontsize=13, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, 280)

# 图9: 误差与密度分布
ax9 = fig.add_subplot(gs[2, 2])
ax9.scatter(k_points*1000, q_errors, c=v_points*3.6, s=60, alpha=0.7,
           cmap='coolwarm', edgecolors='black', linewidth=0.5, vmin=0, vmax=40)
ax9.axhline(0, color='k', linestyle='--', linewidth=1)
ax9.set_xlabel('Density (veh/km)', fontsize=12)
ax9.set_ylabel('Flow Error (veh/h)', fontsize=12)
ax9.set_title('Flow Error vs Density', fontsize=13, fontweight='bold')
cbar = plt.colorbar(ax9.collections[0], ax=ax9)
cbar.set_label('Velocity (km/h)', fontsize=10)
ax9.grid(True, alpha=0.3)

fig.suptitle('Spin Field Inversion Quality Assessment', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(report_img_path, dpi=200, bbox_inches='tight')
print(f"\nSaved Assessment Plot: {report_img_path}")
plt.close()

# === 生成报告 ===
report = []
report.append("="*80)
report.append("Spin Field Inversion Quality Assessment Report")
report.append("="*80)
report.append(f"\nMethod: {meta.get('method', 'Unknown')}")
report.append(f"Inversion Target: s(x) ONLY")
report.append(f"Density Source: Observation Data (Edie Method)")

report.append(f"\n1. FD Matching Quality (Core Metrics)")
report.append(f"   FD Observation Points: {n_points}")
report.append(f"   ")
report.append(f"   Flow Prediction:")
report.append(f"     RMSE = {rmse_q:.1f} veh/h")
report.append(f"     MAE  = {mae_q:.1f} veh/h")
report.append(f"     R2   = {r2_q:.4f}")
report.append(f"   ")
report.append(f"   Velocity Prediction:")
report.append(f"     RMSE = {rmse_v:.2f} km/h")
report.append(f"     MAE  = {mae_v:.2f} km/h")
report.append(f"     R2   = {r2_v:.4f}")

report.append(f"\n2. Phase Weight Distribution")
for g in range(3):
    count = np.sum(dominant_phase == g)
    percent = count / cells * 100
    report.append(f"   {phase_names[g]:10s}: {count:3d} cells ({percent:5.1f}%)")

report.append(f"\n3. Prototype Parameters")
for g, proto in enumerate(prototypes):
    report.append(f"   Prototype {g}:")
    report.append(f"     vf={proto.vf*3.6:.1f} km/h, Q0={proto.Q0*3600:.0f} veh/h, ρ_c={proto.rho_c*1000:.1f} veh/km")

report.append(f"\n4. Convergence Performance")
report.append(f"   Iterations: {len(history['loss'])}")
report.append(f"   Loss Drop: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f} ({(1-history['loss'][-1]/history['loss'][0])*100:.1f}%)")

report.append(f"\n5. Parallelogram Sampling")
params = meta.get('parallelogram_params', {})
report.append(f"   Lw={params.get('Lw')}m, H={params.get('H')}m")
report.append(f"   FD Points: {n_points} (Baseline 125, +{(n_points-125)/125*100:.1f}%)")

report.append(f"\n6. Phase Structure Metrics (Section 7.5)")
report.append(f"   Phase Entropy H(x):")
report.append(f"     Mean = {mean_H:.4f} (Low=Pure, High=Mixed)")
report.append(f"     Std  = {std_H:.4f}")
report.append(f"   Structure Regularity G_pi:")
report.append(f"     Total = {G_pi_total:.4f}")
report.append(f"     Mean  = {G_pi_mean:.6f} (Low=Smooth, High=Oscillating)")

report.append(f"\nQuality Rating:")
if r2_q > 0.9 and rmse_v < 5.0:
    rating = "A (Excellent)"
elif r2_q > 0.8 and rmse_v < 8.0:
    rating = "B (Good)"
elif r2_q > 0.6 and rmse_v < 12.0:
    rating = "C (Fair)"
else:
    rating = "D (Poor)"
report.append(f"   FD Matching Quality: {rating}")
report.append(f"   Based on: R2_q={r2_q:.3f}, RMSE_v={rmse_v:.2f} km/h")

report.append("="*80)

with open(report_txt_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

# for line in report:
#     print(line)

print(f"\nSaved Report: {report_txt_path}")
print("\n[OK] Evaluation Completed!")

