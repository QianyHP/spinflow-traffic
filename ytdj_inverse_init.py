# ytdj_inverse_init.py
# ------------------------------------------------------------
# 目的：
#   基于 YTDJ 轨迹（无人机 24fps），将同向车道数据栅格化为
#   ρ_obs(t,x), q_obs(t,x)（可选 SLE：按 L(x) 折算为单车道等效），
#   并用短窗反演（SPSA/FD）估计初始自旋场 s^0(x)=(sx,sy,sz)。
#
# 主要特性：
#   - 严格 Edie 近似：ρ=车辆-秒/(ΔxΔt)，q=车-米/(ΔxΔt)
#   - 车道筛选 (--lanes) 与 单车道等效 SLE (--use-sle) 的 L(x) 剖面
#   - 反演损失用“全场”L2：ρ(t,x)、q(t,x) 逐点对齐（而非仅均值）
#   - SPSA 近似梯度（每迭代仅 2 次 K 步仿真，速度快）
#   - 产出 inverse_init.npz：rho0, sx0, sy0, sz0, meta(dx, dt_model, cells, t0)
#   - 诊断图：观测时空 ρ（SLE 后）与 t0 切片对比
#
# 用法示例：
#   python ytdj_inverse_init.py \
#       --csv data/VTDJ_6-10.csv --direction wb \
#       --road-length 362 --cells 90 \
#       --dt-obs 0.5 --t0 0 --T-obs 60 \
#       --use-sle --lanes all \
#       --group-size 6 --K-inv 20 --iters 25 \
#       --save-prefix inv_ytdj
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List

# ---------------------- 基础工具 ----------------------

def sigmoid(x): 
    return 1.0/(1.0+np.exp(-x))

def project_ball(sx, sy, sz, r):
    """把 (sx,sy,sz) 投影到半径 r 的球内（逐元素）"""
    n = np.sqrt(sx*sx + sy*sy + sz*sz) + 1e-12
    return sx*(r/n), sy*(r/n), sz*(r/n)

def roll_lr(M: int):
    """周期边界左/右索引"""
    idx = np.arange(M)
    return np.roll(idx, 1), np.roll(idx, -1)

# ---------------------- 模型最小内核（与主模型一致） ----------------------

def spin_to_params(sx, sy, sz, r, v_limit, v_min, w_min, w_max, Q0,
                   a0, az_map, b0, bx_map, c0, by_map, d0, dy_map,
                   enable_map_sz=True, enable_map_sx=True, enable_map_sy=True):
    """自旋 → 三角基本图参数（逐 cell）"""
    zx, zy, zz = r*sx, r*sy, r*sz
    vf = v_min + (v_limit - v_min) * sigmoid(a0 + (az_map*zz if enable_map_sz else 0))
    cap_mult = 0.5 + 0.5 * sigmoid(b0 + (bx_map*zx if enable_map_sx else 0))
    Qcap = Q0 * cap_mult
    wloc = w_min + (w_max - w_min) * sigmoid(c0 + (by_map*zy if enable_map_sy else 0))
    rho_jam = 0.6 + 0.4 * (1 - sigmoid(d0 + (dy_map*zy if enable_map_sy else 0)))
    rho_c = (wloc/(vf+wloc))*rho_jam
    return vf, Qcap, wloc, rho_jam, rho_c

def ctm_step(rho, vf, Qcap, wloc, rho_jam, dx, dt):
    """单步 CTM（需求-供给，环路）
    注意：为兼容历史接口保留，但默认流程不再使用环路边界。
    """
    M = rho.shape[0]
    left, right = roll_lr(M)
    demand = np.minimum(vf*rho, Qcap)
    supply = np.minimum(Qcap, wloc*(rho_jam - rho))
    y = np.minimum(demand, supply[right])  # i -> i+1
    rho_next = rho + (dt/dx)*(y[left] - y)
    rho_next = np.minimum(np.maximum(rho_next, 0.0), rho_jam)
    return rho_next, y

def ctm_step_open(rho, vf, Qcap, wloc, rho_jam, dx, dt,
                  demand_in: float = 0.0,
                  supply_out: float = None):
    """单步 CTM（需求-供给，开边界）
    - 上游边界以恒定需求 demand_in 注入（veh/s），受 cell0 的 supply 限制
    - 下游边界以恒定供给上限 supply_out 抽出（veh/s），受 cell(M-1) 的 demand 限制
    - 当 supply_out=None 时，按自由出流（用 Qcap[-1] 作为上限）
    返回：rho_next, y，其中 y[i] 为 i->i+1 的通量，y[M-1] 为末端出流
    """
    M = rho.shape[0]
    demand = np.minimum(vf*rho, Qcap)
    supply = np.minimum(Qcap, wloc*(rho_jam - rho))

    # 内部通量（0..M-2）：i -> i+1
    y_internal = np.minimum(demand[:-1], supply[1:])

    # 上游入流
    y_in = np.minimum(max(0.0, float(demand_in)), supply[0])

    # 下游出流能力
    sup_out_cap = Qcap[-1] if supply_out is None else max(0.0, float(supply_out))
    y_out = np.minimum(demand[-1], sup_out_cap)

    # 更新密度（守恒形式）
    rho_next = rho.copy()
    if M == 1:
        # 退化情形
        rho_next[0] = rho[0] + (dt/dx)*(y_in - y_out)
    else:
        rho_next[0] = rho[0] + (dt/dx)*(y_in - y_internal[0])
        rho_next[1:-1] = rho[1:-1] + (dt/dx)*(y_internal[:-1] - y_internal[1:])
        rho_next[-1] = rho[-1] + (dt/dx)*(y_internal[-1] - y_out)

    rho_next = np.minimum(np.maximum(rho_next, 0.0), rho_jam)

    # 组装通量向量，长度 M，与环路版本保持一致
    y = np.zeros(M)
    if M == 1:
        y[0] = y_out
    else:
        y[:-1] = y_internal
        y[-1] = y_out
    return rho_next, y

# ---------------------- 轨迹读入与预处理 ----------------------

def load_ytdj_csv(path: str,
                  col_map: Optional[Dict[str,str]] = None,
                  fps: float = 24.0) -> pd.DataFrame:
    """
    读取 YTDJ CSV；可通过 col_map 指定列名映射：
      col_map = {'veh':'id','t':'time','x':'x','y':'y','vx':'vx','vy':'vy','lane':'lane'}
    若无速度列，将按轨迹差分估计（步长=1/fps）。
    """
    df = pd.read_csv(path)
    if col_map is None:
        col_map = {}
        # 车辆ID
        for k in ['veh','track','id','vehicle_id','vehicleID','object_id']:
            if k in df.columns: col_map['veh']=k; break
        # 时间
        for k in ['t','time','time(s)','timestamp','frame_time']:
            if k in df.columns: col_map['t']=k; break
        # 坐标
        for k in ['x','pos_x','cx','lon_x','longitudinalDistance(m)']:
            if k in df.columns: col_map['x']=k; break
        for k in ['y','pos_y','cy','lat_y','lateralDistance(m)']:
            if k in df.columns: col_map['y']=k; break
        # 速度
        for k in ['vx','vel_x','vx_mps']:
            if k in df.columns: col_map['vx']=k; break
        for k in ['vy','vel_y','vy_mps']:
            if k in df.columns: col_map['vy']=k; break
        # 车道
        for k in ['lane','lane_id','laneID','Lane','laneId']:
            if k in df.columns: col_map['lane']=k; break

    # 标准列
    df = df.rename(columns={
        col_map.get('veh','veh'): 'veh',
        col_map.get('t','t'): 't',
        col_map.get('x','x'): 'x',
        col_map.get('y','y'): 'y'
    })
    if 'vx' in col_map and col_map['vx'] in df.columns:
        df = df.rename(columns={col_map['vx']: 'vx'})
    else:
        df['vx'] = np.nan
    if 'vy' in col_map and col_map['vy'] in df.columns:
        df = df.rename(columns={col_map['vy']: 'vy'})
    else:
        df['vy'] = np.nan
    if 'lane' in col_map and col_map['lane'] in df.columns:
        df = df.rename(columns={col_map['lane']: 'lane'})
    else:
        df['lane'] = -1

    # 差分估计速度（若缺失）
    df = df.sort_values(['veh','t'])
    if df['vx'].isna().any():
        dt_est = 1.0/fps
        for vid, g in df.groupby('veh'):
            dx = g['x'].diff().fillna(0.0)
            dy = g['y'].diff().fillna(0.0)
            df.loc[g.index, 'vx'] = dx/dt_est
            df.loc[g.index, 'vy'] = dy/dt_est
    return df

def select_direction(df: pd.DataFrame, direction: str, road_len: float) -> pd.DataFrame:
    """
    选定行车方向：'eb'（向东，x 递增）或 'wb'（向西，x 递减）。
    若选择 wb，则用 s = road_len - x，使“沿行驶方向”坐标单调递增。
    """
    g = df.groupby('veh').agg(x0=('x','first'), x1=('x','last'))
    g['dir'] = np.sign(g['x1'] - g['x0']).replace(0, 1)  # 0 当作正向
    dir_map = g['dir'].to_dict()
    df['dir'] = df['veh'].map(dir_map)

    if direction.lower() == 'eb':
        df = df[df['dir'] >= 0].copy()
        df['s'] = df['x']               # 行驶方向坐标
        df['vs'] = df['vx']
    else:
        df = df[df['dir'] <= 0].copy()
        df['s'] = road_len - df['x']    # 翻转到“向前为正”
        df['vs'] = -df['vx']
    return df

def filter_lanes(df: pd.DataFrame, lanes: Optional[List[int]]):
    """按 lane-id 过滤；lanes=None 表示不过滤。"""
    if lanes is None:
        return df
    return df[df["lane"].isin(lanes)].copy()

# ---------------------- SLE：有效车道数 L(x) 与 Edie 栅格化 ----------------------

def estimate_lane_count_profile(df: pd.DataFrame, x_min: float, x_max: float, dx: float,
                                presence_thresh: int = 20) -> np.ndarray:
    """
    估计每个空间 cell 的有效车道数 L(x)：
    对每个 lane，若该 cell 内样本数 >= presence_thresh，则认为该 lane 在该 cell“有效”。
    """
    M = int(np.round((x_max - x_min) / dx))
    lane_ids = sorted(df["lane"].dropna().unique().tolist())
    L = np.zeros(M, dtype=int)
    for ln in lane_ids:
        di = df[df["lane"] == ln]
        if di.empty: 
            continue
        i_idx = np.clip(((di["s"] - x_min) / dx).values.astype(int), 0, M - 1)
        counts = np.bincount(i_idx, minlength=M)
        L += (counts >= presence_thresh).astype(int)
    L = np.maximum(L, 1)
    return L

def edie_grid_strict(df: pd.DataFrame, x_min: float, x_max: float, dx: float,
                     t0: float, T_obs: float, dt_obs: float, dt_frame: float,
                     lane_count: Optional[np.ndarray] = None):
    """
    严格 Edie：逐样本把 dt_frame 与 |v_s|*dt_frame 累入 (k,i) 盒子。
    若给 lane_count(x)，则将“所有同向车道”折算为“单车道等效”（SLE）。
    返回：rho_obs, q_obs, u_obs, t_grid
    """
    M = int(np.round((x_max - x_min) / dx))
    t_grid = np.arange(t0, t0 + T_obs, dt_obs)
    K = len(t_grid)
    Theta = np.zeros((K, M))  # 车辆-秒
    Xi    = np.zeros((K, M))  # 车-米

    k_idx = np.clip(((df['t'] - t0) / dt_obs).values.astype(int), 0, K - 1)
    i_idx = np.clip(((df['s'] - x_min) / dx).values.astype(int), 0, M - 1)
    vs    = df['vs'].values

    # 每个样本贡献的 dt_frame 与 |v_s|*dt_frame 累加
    np.add.at(Theta, (k_idx, i_idx), dt_frame)
    np.add.at(Xi,    (k_idx, i_idx), np.abs(vs) * dt_frame)

    area = dx * dt_obs
    rho_all = Theta / area
    q_all   = Xi    / area
    with np.errstate(divide='ignore', invalid='ignore'):
        u_all = np.where(rho_all > 1e-9, q_all / rho_all, 0.0)

    # 单车道等效（SLE）
    if lane_count is not None:
        Lx = lane_count[np.newaxis, :]
        rho_obs = rho_all / Lx
        q_obs   = q_all   / Lx
        u_obs   = np.where(rho_obs > 1e-9, q_obs / rho_obs, 0.0)
    else:
        rho_obs, q_obs, u_obs = rho_all, q_all, u_all

    # ρ 归一到 [0,1]（用 99% 分位作为 ρ_jam 近似）；q 保持物理量级
    rho_jam_est = np.quantile(rho_obs[rho_obs > 0], 0.99) if np.any(rho_obs > 0) else 1.0
    rho_obs = np.clip(rho_obs / max(rho_jam_est, 1e-6), 0.0, 1.0)
    return rho_obs, q_obs, u_obs, t_grid

# ---------------------- 边界流估计 ----------------------

def estimate_boundary_flows(rho_obs: np.ndarray, q_obs: np.ndarray, 
                           dx: float, dt_obs: float,
                           time_avg_window: int = 10) -> tuple:
    """
    从观测时空数据估计边界流条件
    - demand_in: 上游入流需求（前几个cell的时间平均流量）
    - supply_out: 下游出流供给（后几个cell的时间平均流量）
    
    Args:
        rho_obs: 观测密度场 [K, M]
        q_obs: 观测流量场 [K, M]
        dx: 空间步长
        dt_obs: 观测时间步长
        time_avg_window: 用于平均的时间步数
    
    Returns:
        (demand_in, supply_out): 边界流量（veh/s）
    """
    K, M = rho_obs.shape
    avg_window = min(time_avg_window, K)
    
    # 上游需求：取前3个cell的时间平均流量
    q_upstream = q_obs[:avg_window, :3].mean()
    demand_in = float(q_upstream)
    
    # 下游供给：取后3个cell的时间平均流量
    q_downstream = q_obs[:avg_window, -3:].mean()
    supply_out = float(q_downstream)
    
    return demand_in, supply_out

# ---------------------- 反演：分组 + SPSA/FD ----------------------

def inverse_spin_shortwindow(rho_obs: np.ndarray, q_obs: np.ndarray,
                             dx: float, dt_model: float, dt_obs: float,
                             group_size: int = 6, K_inv: int = 20, iters: int = 25,
                             # 边界流条件
                             demand_in: float = 0.0, supply_out: float = None,
                             # 映射与模型参数（需与主模型保持一致）
                             v_limit: float = 1.05, v_min: float = 0.4,
                             w_min: float = 0.4, w_max: float = 0.9, Q0: float = 0.30,
                             a0: float = 0.0, az_map: float = 2.2,
                             b0: float = 0.0, bx_map: float = 1.8,
                             c0: float = 0.0, by_map: float = 1.8,
                             d0: float = 0.0, dy_map: float = 1.8,
                             # 先验/正则与损失权重
                             lam_H: float = 0.01, lam_smooth: float = 0.05,
                             w_rho: float = 1.0, w_q: float = 0.5,
                             # 优化器
                             use_spsa: bool = True, spsa_c: float = 1e-2,
                             step: float = 0.5, r_min: float = 0.3, r_max: float = 0.9,
                             seed: int = 123) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    输入：rho_obs[T,K], q_obs[T,K]（用前 K_inv 帧）
    输出：rho0, sx0, sy0, sz0（逐 cell）
    """
    rng = np.random.default_rng(seed)
    T_obs, M = rho_obs.shape
    K = min(K_inv, T_obs-1)
    G = int(np.ceil(M / group_size))

    # 初值：rho0 用观测第 0 帧；自旋方向按“外场”启发式
    rho0 = np.clip(rho_obs[0].copy(), 0.0, 1.0)
    rho_star = 0.32
    hx = (0.25 - (rho0 - rho_star)**2)
    hy = (rho0 - rho_star)
    hz = (rho_star - rho0)
    hn = np.sqrt(hx*hx + hy*hy + hz*hz) + 1e-12
    sx = hx/hn + 0.02*rng.standard_normal(M)
    sy = hy/hn + 0.02*rng.standard_normal(M)
    sz = hz/hn + 0.02*rng.standard_normal(M)
    r0 = np.clip(0.6 + 0.05*rng.standard_normal(M), r_min, r_max)
    sx, sy, sz = project_ball(sx, sy, sz, r0)

    # 分组向量 S[g,3]
    groups = [slice(g*group_size, min((g+1)*group_size, M)) for g in range(G)]
    S = np.zeros((G,3))
    for g, sl in enumerate(groups):
        S[g,0] = float(np.mean(sx[sl]))
        S[g,1] = float(np.mean(sy[sl]))
        S[g,2] = float(np.mean(sz[sl]))

    def expand_S(S):
        """把组向量展开为逐 cell 的 sx,sy,sz，并统一半径到 r0"""
        sx_e = np.zeros(M); sy_e = np.zeros(M); sz_e = np.zeros(M)
        for g, sl in enumerate(groups):
            sx_e[sl] = S[g,0]; sy_e[sl] = S[g,1]; sz_e[sl] = S[g,2]
        r = np.clip(np.sqrt(sx_e**2+sy_e**2+sz_e**2), r_min, r_max)
        return project_ball(sx_e, sy_e, sz_e, r)
    
    def resample_time(arr, dt_src, dt_tgt, K_tgt):
        """时间重采样：将数组从dt_src网格插值到dt_tgt网格
        arr: (K_src, M)
        返回: (K_tgt, M)
        """
        K_src = arr.shape[0]
        M = arr.shape[1]
        t_src = np.arange(K_src) * dt_src
        t_tgt = np.arange(K_tgt) * dt_tgt
        out = np.empty((K_tgt, M))
        for j in range(M):
            out[:, j] = np.interp(t_tgt, t_src, arr[:, j])
        return out

    def simulate_window(sx_e, sy_e, sz_e):
        """固定自旋，滚动模型步（开边界）；返回重采样到观测网格的 ρ(t,x) 与 q(t,x)
        使用外层传入的 demand_in 和 supply_out
        """
        # 计算模型需要运行多少步才能覆盖观测时间窗口
        T_window = K * dt_obs  # 观测时间窗口长度
        K_model = int(np.ceil(T_window / dt_model))  # 模型步数
        
        rho = rho0.copy()
        rho_seq = []
        q_seq = []
        for k in range(K_model):
            vf, Qcap, wloc, rho_jam, rho_c = spin_to_params(
                sx_e, sy_e, sz_e, np.minimum(1.0, np.sqrt(sx_e*sx_e+sy_e*sy_e+sz_e*sz_e)),
                v_limit, v_min, w_min, w_max, Q0, a0, az_map, b0, bx_map, c0, by_map, d0, dy_map
            )
            rho, y = ctm_step_open(rho, vf, Qcap, wloc, rho_jam, dx, dt_model,
                                   demand_in=demand_in, supply_out=supply_out)
            rho_seq.append(rho.copy())
            q_seq.append(y.copy())
        
        rho_model = np.stack(rho_seq, 0)  # (K_model, M)
        q_model = np.stack(q_seq, 0)      # (K_model, M)
        
        # 重采样到观测时间网格
        rho_resampled = resample_time(rho_model, dt_model, dt_obs, K)
        q_resampled = resample_time(q_model, dt_model, dt_obs, K)
        
        return rho_resampled, q_resampled

    def energy_prior(sx_e, sy_e, sz_e):
        """海森堡能量先验 + 二范数平滑（相邻 cell）"""
        left, right = roll_lr(M)
        H_align = - (sx_e*sx_e[left] + sy_e*sy_e[left] + sz_e*sz_e[left]).mean()
        smooth = ((np.diff(sx_e)**2).mean() + (np.diff(sy_e)**2).mean() + (np.diff(sz_e)**2).mean())
        return lam_H*H_align + lam_smooth*smooth

    def loss_from_fields(rho_pred, q_pred):
        """全场损失（逐点 L2），q 用同尺度（不再取均值）"""
        Lr = np.mean((rho_pred - rho_obs[:K])**2)
        Lq = np.mean((q_pred   - q_obs [:K])**2)
        return w_rho*Lr + w_q*Lq

    # 迭代
    for it in range(iters):
        sx_e, sy_e, sz_e = expand_S(S)
        rho_pred, q_pred = simulate_window(sx_e, sy_e, sz_e)
        L_prior = energy_prior(sx_e, sy_e, sz_e)
        L_data  = loss_from_fields(rho_pred, q_pred)
        L = L_data + L_prior

        if use_spsa:
            # ---- SPSA：两次仿真近似梯度 ----
            Delta = np.random.choice([-1.0, 1.0], size=S.shape)
            S_plus  = S + spsa_c * Delta
            S_minus = S - spsa_c * Delta

            sx_p, sy_p, sz_p = expand_S(S_plus)
            rp, qp = simulate_window(sx_p, sy_p, sz_p)
            Lp = loss_from_fields(rp, qp) + energy_prior(sx_p, sy_p, sz_p)

            sx_m, sy_m, sz_m = expand_S(S_minus)
            rm, qm = simulate_window(sx_m, sy_m, sz_m)
            Lm = loss_from_fields(rm, qm) + energy_prior(sx_m, sy_m, sz_m)

            # g ≈ (Lp-Lm)/(2c) * sign(Δ)  （用 Δ 的符号作约化）
            ghat = (Lp - Lm) / (2.0*spsa_c) * Delta
            grad = ghat
        else:
            # ---- 有限差分（更慢但更准）----
            eps_fd = 1e-3
            grad = np.zeros_like(S)
            for g in range(G):
                for c in range(3):
                    S[g,c] += eps_fd
                    sx_p, sy_p, sz_p = expand_S(S)
                    rp, qp = simulate_window(sx_p, sy_p, sz_p)
                    Lp = loss_from_fields(rp, qp) + energy_prior(sx_p, sy_p, sz_p)
                    S[g,c] -= 2*eps_fd
                    sx_m, sy_m, sz_m = expand_S(S)
                    rm, qm = simulate_window(sx_m, sy_m, sz_m)
                    Lm = loss_from_fields(rm, qm) + energy_prior(sx_m, sy_m, sz_m)
                    grad[g,c] = (Lp - Lm) / (2*eps_fd)
                    S[g,c] += eps_fd

        # 梯度更新与半径投影（逐组）
        S -= step * grad
        for g in range(G):
            r = np.linalg.norm(S[g]); r = np.clip(r, 1e-6, 0.9)
            S[g] = S[g] * (np.clip(r, r_min, r_max) / r)

        print(f"[Iter {it+1:02d}] Loss={L:.6f}  data={L_data:.6f}  prior={L_prior:.6f}")

    sx0, sy0, sz0 = expand_S(S)
    return rho0, sx0, sy0, sz0

# ---------------------- 主流程（CLI） ----------------------

def parse_lanes_arg(s: str) -> Optional[List[int]]:
    """'all' -> None；'6,7,8' -> [6,7,8]"""
    if s is None or s.lower() == "all":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser("YTDJ inverse initialization (strict Edie + SLE + SPSA)")
    ap.add_argument("--csv", type=str, default="data/VTDJ_6-10.csv", help="path to YTDJ CSV")
    ap.add_argument("--direction", type=str, default="wb", choices=["eb","wb"], help="select eastbound/westbound")
    ap.add_argument("--fps", type=float, default=24.0, help="video fps (time step ≈ 1/fps)")
    ap.add_argument("--road-length", type=float, default=362.0, help="road segment length (m)")
    ap.add_argument("--cells", type=int, default=90, help="number of spatial cells M")
    ap.add_argument("--dt-obs", type=float, default=0.5, help="observation time step for gridding (s)")
    ap.add_argument("--t0", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--T-obs", type=float, default=60.0, help="observation horizon for inversion (s)")
    ap.add_argument("--lanes", type=str, default="all", help="e.g., '6,7,8,9,10' or 'all'")
    ap.add_argument("--use-sle", action="store_true", help="merge same-direction lanes via SLE (divide by L(x))")

    ap.add_argument("--group-size", type=int, default=6, help="cells per group for inversion")
    ap.add_argument("--K-inv", type=int, default=60, help="model steps in the short window")
    ap.add_argument("--iters", type=int, default=50, help="iterations of SPSA/FD")
    ap.add_argument("--seed", type=int, default=123, help="random seed")
    ap.add_argument("--save-prefix", type=str, default="inv_ytdj", help="output prefix")

    # 可选：调参
    ap.add_argument("--use-spsa", action="store_true", help="use SPSA instead of finite difference")
    ap.add_argument("--step", type=float, default=0.5, help="gradient step size")
    ap.add_argument("--spsa-c", type=float, default=1e-2, help="SPSA perturbation magnitude")
    ap.add_argument("--w-rho", type=float, default=1.0, help="loss weight for density field")
    ap.add_argument("--w-q", type=float, default=0.5, help="loss weight for flow field")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # 1) 读数据、选方向、可选车道过滤
    df = load_ytdj_csv(args.csv, fps=args.fps)
    df = select_direction(df, args.direction, args.road_length)
    lanes = parse_lanes_arg(args.lanes)
    df = filter_lanes(df, lanes)

    # 2) 严格 Edie + （可选）SLE 折算
    dx = args.road_length / args.cells
    dt_frame = 1.0 / args.fps
    lane_count = None
    if args.use_sle and (lanes is None):
        lane_count = estimate_lane_count_profile(df, 0.0, args.road_length, dx)

    rho_obs, q_obs, u_obs, t_grid = edie_grid_strict(
        df, x_min=0.0, x_max=args.road_length, dx=dx,
        t0=args.t0, T_obs=args.T_obs, dt_obs=args.dt_obs, dt_frame=dt_frame,
        lane_count=lane_count
    )

    # 3) 估计边界流条件
    demand_in, supply_out = estimate_boundary_flows(rho_obs, q_obs, dx, args.dt_obs)
    print(f"Estimated boundary flows: demand_in={demand_in:.4f} veh/s, supply_out={supply_out:.4f} veh/s")

    # 4) 反演（与核心模型的 dt/CFL 对齐：dt_model = 0.40 / max(v_limit, w_max)）
    v_limit, w_max = 1.05, 0.9
    dt_model = 0.40 / max(v_limit, w_max)

    rho0, sx0, sy0, sz0 = inverse_spin_shortwindow(
        rho_obs, q_obs, dx, dt_model, args.dt_obs,
        demand_in=demand_in, supply_out=supply_out,
        group_size=args.group_size, K_inv=args.K_inv, iters=args.iters,
        use_spsa=args.use_spsa, spsa_c=args.spsa_c, step=args.step,
        w_rho=args.w_rho, w_q=args.w_q, seed=args.seed
    )

    # 5) 保存反演结果
    np.savez(f"{args.save_prefix}_inverse_init.npz",
             rho0=rho0, sx0=sx0, sy0=sy0, sz0=sz0,
             rho_obs_t0=rho_obs[0],  # 保存观测的初始密度（用于评估对比）
             meta=dict(dx=dx, dt_model=dt_model, cells=args.cells, t0=args.t0, 
                      road_length=args.road_length, dt_obs=args.dt_obs, T_obs=args.T_obs))

    print("Saved inverse init to:", f"{args.save_prefix}_inverse_init.npz")

if __name__ == "__main__":
    main()
