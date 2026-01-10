"""
SpinFlow 主反演程序 (Main Pipeline)

功能描述：
    该脚本是 SpinFlow 框架的核心入口，负责编排整个从轨迹数据到自旋场反演的全流程。
    它集成数据加载、近稳态采样、物理模型校准、EM算法求解以及结果归档等步骤。

主要流程：
    1. 数据加载 (Data Loading): 读取并清洗轨迹数据 (支持 YTDJ, RML, XAM-N6 等数据集)。
    2. 采样 (Sampling): 利用平行四边形方法 (Parallelogram Method) 提取近稳态 FD 点。
    3. 建模 (Modeling): 校准三相基本图原型 (Free, Critical, Congested)。
    4. 求解 (Solver): 运行 Spin-CTM EM 算法，反演空间自旋场 s(x) 和相混合权重 π(x)。
    5. 归档 (Archiving): 保存 .npz 结果文件及初步可视化图表。

使用示例：
    python main.py --dataset YTDJ
    python main.py --dataset RML --direction eb
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict

# 导入新模块
from sampler import sample_parallelograms, extract_fd_points, visualize_parallelograms
from fd_model import calibrate_three_prototypes
from phase_utils import visualize_phase_map
from solver import em_inverse_fd

# 导入数据预处理函数
from preprocessing import load_trajectory_csv, select_direction, filter_lanes, parse_lanes_arg


def main():
    ap = argparse.ArgumentParser("YTDJ FD-based Inversion")
    
    # 数据参数
    ap.add_argument("--dataset", type=str, default="YTDJ", help="Dataset name (YTDJ, RML, XAM-N6)")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV file. If None, derived from dataset.")
    ap.add_argument("--direction", type=str, default="eb", choices=["eb","wb"])
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--road-length", type=float, default=280.0)
    ap.add_argument("--lanes", type=str, default="all")
    
    # 采样参数（优化后的最佳参数）
    ap.add_argument("--t0", type=float, default=60.0)
    ap.add_argument("--T-sample", type=float, default=485.0, help="sampling time window (default: to data end)")
    ap.add_argument("--wave-speed", type=float, default=-8.5, help="wave speed (km/h, negative, optimal: -8.5)")
    ap.add_argument("--Lw", type=float, default=30.0, help="parallelogram long edge (m)")
    ap.add_argument("--H", type=float, default=10.0, help="parallelogram height (m)")
    ap.add_argument("--n-select", type=int, default=200, help="parallelograms per speed (optimal: 200)")
    ap.add_argument("--min-points", type=int, default=3, help="minimum points per parallelogram")
    
    # 反演参数
    ap.add_argument("--cells", type=int, default=140)
    ap.add_argument("--em-iters", type=int, default=50, help="EM iterations (will stop early if converged)")
    ap.add_argument("--inner-iters", type=int, default=20, help="E-step inner iterations")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save-prefix", type=str, default=None)
    
    args = ap.parse_args()
    
    # Auto-configure paths based on dataset
    if args.csv is None:
        if args.dataset == "YTDJ":
            args.csv = "../data/processed_data/YTDJ/VTDJ_6-10.csv"
        elif args.dataset == "RML":
            args.csv = "../data/processed_data/RML/RML.csv" # Placeholder
        elif args.dataset == "XAM-N6":
            args.csv = "../data/processed_data/XAM-N6/XAM-N6.csv" # Placeholder
        else:
            args.csv = f"../data/processed_data/{args.dataset}/{args.dataset}.csv"

    if args.save_prefix is None:
        args.save_prefix = f"../results/{args.dataset}/{args.dataset}"

    np.random.seed(args.seed)
    
    print("="*80)
    print(f"{args.dataset} FD-Based Inversion Pipeline")
    print("="*80)
    
    # 1) 加载并预处理轨迹数据
    print("\n[Step 1/6] Loading and preprocessing trajectory data...")
    df = load_trajectory_csv(args.csv, fps=args.fps)
    df = select_direction(df, args.direction, args.road_length)
    lanes = parse_lanes_arg(args.lanes)
    df = filter_lanes(df, lanes)
    
    # 确保lane列存在（用于SLE折算）
    if 'lane' not in df.columns:
        print("  Warning: No lane info, assuming single lane")
        df['lane'] = 1
    
    # 时间窗口筛选
    t_max = args.t0 + args.T_sample
    df_window = df[(df['t'] >= args.t0) & (df['t'] < t_max)].copy()
    
    print(f"  Total samples: {len(df)}")
    print(f"  Window samples: {len(df_window)} (t={args.t0}-{t_max}s)")
    print(f"  Direction: {args.direction}")
    
    # 2) 平行四边形近稳态采样
    print("\n[Step 2/6] Quasi-stationary sampling with parallelograms...")
    parallelograms = sample_parallelograms(
        df_window,
        wave_speed=args.wave_speed,
        given_speeds=None,  # 使用默认范围（5-55 km/h）
        Lw=args.Lw,
        H=args.H,
        t_min=args.t0,
        t_max=t_max,
        x_min=0.0,
        x_max=args.road_length,
        n_select_per_speed=args.n_select,
        min_points=args.min_points,
        seed=args.seed
    )
    
    # 提取FD点云
    fd_points = extract_fd_points(parallelograms)
    
    if fd_points['n_points'] == 0:
        print("\n❌ Error: No valid parallelograms found!")
        print("Suggestions:")
        print("  1. Decrease parallelogram size (--Lw 80 --H 20)")
        print("  2. Adjust wave speed (--wave-speed -10 or -20)")
        print("  3. Lower minimum points (--min-points 3)")
        return
    
    print(f"\n  Extracted {fd_points['n_points']} FD points")
    print(f"  k range: [{fd_points['k'].min():.1f}, {fd_points['k'].max():.1f}] veh/km")
    print(f"  q range: [{fd_points['q'].min():.0f}, {fd_points['q'].max():.0f}] veh/h")
    
    # 可视化平行四边形
    visualize_parallelograms(df_window, parallelograms, 
                            save_path=f"{args.save_prefix}_parallelograms.png")
    
    # 3) 校准FD原型基线
    print("\n[Step 3/6] Calibrating FD prototypes...")
    prototypes = calibrate_three_prototypes(fd_points, n_prototypes=3)
    

    
    # 4) 准备空间网格并计算观测密度（用于反演初始化）
    print("\n[Step 4/6] Computing observed density field (Edie method)...")
    dx = args.road_length / args.cells
    x_pos = np.arange(args.cells) * dx
    dt_grid = 0.25
    
    # 计算在采样窗口内的密度变化 (用于CTM守恒约束)
    t_obs_start = args.t0
    t_obs_end = args.t0 + args.T_sample
    df_obs = df_window[(df_window['t'] >= t_obs_start) & (df_window['t'] < t_obs_end)].copy()
    
    # Edie栅格化
    K_obs = int((t_obs_end - t_obs_start) / dt_grid)
    M = args.cells
    Theta = np.zeros((K_obs, M))
    
    dt_frame = 1.0 / args.fps
    k_idx = np.clip(((df_obs['t'] - t_obs_start) / dt_grid).values.astype(int), 0, K_obs - 1)
    i_idx = np.clip((df_obs['s'] / dx).values.astype(int), 0, M - 1)
    
    np.add.at(Theta, (k_idx, i_idx), dt_frame)
    
    # 估计车道数
    L = np.ones(M, dtype=int)
    if 'lane' in df_obs.columns:
        lane_ids = sorted(df_obs["lane"].dropna().unique().tolist())
        L_temp = np.zeros(M, dtype=int)
        for ln in lane_ids:
            di = df_obs[df_obs["lane"] == ln]
            if len(di) > 0:
                i_idx_ln = np.clip((di["s"] / dx).values.astype(int), 0, M - 1)
                counts = np.bincount(i_idx_ln, minlength=M)
                L_temp += (counts >= 10).astype(int)
        L = np.maximum(L_temp, 1)
    
    # 计算SLE密度
    area = dx * dt_grid
    rho_all = Theta / area
    Lx = L[np.newaxis, :]
    rho_sle = rho_all / Lx
    
    # 计算t_start和t_end时刻的平均密度（利用滑动平均平滑瞬时波动）
    # 物理目的：利用CTM守恒定律解释相态识别
    rho_start_raw = np.mean(rho_sle[:4], axis=0) # 前1秒均值
    rho_end_raw = np.mean(rho_sle[-4:], axis=0)  # 后1秒均值
    rho_mean_raw = np.mean(rho_sle, axis=0)      # 全窗口均值
    
    # 归一化
    rho_jam_est = np.quantile(rho_sle[rho_sle > 0], 0.99) if np.any(rho_sle > 0) else 1.0
    rho_start_norm = np.clip(rho_start_raw / max(rho_jam_est, 1e-6), 0.0, 1.0)
    rho_end_norm = np.clip(rho_end_raw / max(rho_jam_est, 1e-6), 0.0, 1.0)
    rho_mean_norm = np.clip(rho_mean_raw / max(rho_jam_est, 1e-6), 0.0, 1.0)
    
    print(f"  Spatial grid: {args.cells} cells, dx={dx:.2f}m")
    print(f"  Observed density (60s window, Edie method):")
    print(f"    mean={rho_mean_norm.mean():.4f}, Δρ={ (rho_end_norm - rho_start_norm).mean():+.4f}")
    print(f"  Jam density estimate: {rho_jam_est:.4f} veh/m")
    
    # 5) 自旋场反演（ρ用观测，只反演s）
    print("\n[Step 5/6] Inverting spin field s(x) via FD matching...")
    
    sx0, sy0, sz0, prototypes_final, rho0_final, history = em_inverse_fd(
        fd_points,
        dx,
        M=args.cells,
        rho_obs=rho_mean_norm,  # 全窗口平均密度
        rho_start=rho_start_norm, # CTM守恒约束：起始密度
        rho_end=rho_end_norm,     # CTM守恒约束：结束密度
        dt_window=args.T_sample,  # 动态匹配采样时间跨度
        n_prototypes=3,       # 保持3个原型
        em_iters=args.em_iters,
        inner_iters=args.inner_iters,
        lam_fd_q=1.0,     # 流量匹配权重
        lam_fd_v=0.1,     # 引入微量速度约束 (Pull apart Total Loss and Flow Loss)
        convergence_tol=1e-4,
        seed=args.seed
    )
    
    # 计算最终的相混合权重
    from phase_utils import spin_to_mixture_weights_softmax
    pi_final = spin_to_mixture_weights_softmax(sx0, sy0, sz0, 3)
    
    # 6) 保存结果
    print("\n[Step 6/6] Saving results...")
    
    # 保存npz
    np.savez(
        f"{args.save_prefix}_inverse_init.npz",
        rho0=rho0_final,  # 观测密度（归一化）
        rho_obs_t0=rho_mean_norm,  # 观测密度（用于对比）
        rho_start=rho_start_norm,
        rho_end=rho_end_norm,
        rho_jam_est=rho_jam_est,  # jam密度估计
        sx0=sx0,
        sy0=sy0,
        sz0=sz0,
        pi_final=pi_final,  # 相混合权重
        prototypes=[p.to_dict() for p in prototypes_final],
        fd_points=fd_points,
        history=history,
        meta=dict(
            dx=dx,
            dt=dt_grid,
            cells=args.cells,
            t0=args.t0,
            road_length=args.road_length,
            dataset=args.dataset,
            csv_path=args.csv,
            method='FD-EM-SpinInversion',
            theory='MacroscopicPhaseTransition',
            inversion_target='spin_only_rho_observed',
            density_source='Edie_observation',
            n_prototypes=3,
            lam_fd_q=1.0,
            lam_fd_v=0.5,
            lam_smooth=0.0,
            final_loss=history['loss'][-1] if history['loss'] else None,
            final_loss_q=history['loss_fd_q'][-1] if history['loss_fd_q'] else None,
            final_loss_v=history['loss_fd_v'][-1] if history['loss_fd_v'] else None,
            parallelogram_params=dict(Lw=args.Lw, H=args.H, wave_speed=args.wave_speed)
        )
    )
    
    print(f"  Saved: {args.save_prefix}_inverse_init.npz")
    
    # 可视化相分布地图
    visualize_phase_map(sx0, sy0, sz0, x_pos, prototypes_final,
                       save_path=f"{args.save_prefix}_phase_map.png")
    
    print("\n" + "="*80)
    print("Pipeline Completed Successfully!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {args.save_prefix}_parallelograms.png")
    print(f"  - {args.save_prefix}_phase_map.png")
    print(f"  - {args.save_prefix}_inverse_init.npz")
    print("="*80)


if __name__ == "__main__":
    main()

