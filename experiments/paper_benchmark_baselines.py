"""
Heterogeneous baseline implementations for SpinFlow comparison experiments.

Baseline 1: Three-Phase PWA-CTM  (white-box, hard phase switching + sigmoid blend)
Baseline 2: VBGMM + Spatial KDE  (black-box, statistical clustering + prototype projection)
Baseline 3: PI-DeepONet-Traffic   (PIML, operator learning + early stopping)

All baselines share the same preprocessed FD point set and spatial grid.
"""

import os, sys, time
import numpy as np

from paths import SPIN_SRC

sys.path.insert(0, SPIN_SRC)
from spinflow.fd_model import TriangularFD, PHASES, PHASE_INDEX


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _forward_metrics(k_pts, q_pts, v_pts, x_pts, pi_x, prototypes, dx, cells):
    """Compute RMSE/R2 for flow and velocity given spatial phase weights pi_x."""
    n = len(k_pts)
    q_pred = np.zeros(n)
    v_pred = np.zeros(n)
    for i in range(n):
        cell = int(np.clip(x_pts[i] / dx, 0, cells - 1))
        qp = sum(pi_x[cell, g] * prototypes[g].flow(np.array([k_pts[i]]))[0]
                 for g in range(len(prototypes)))
        q_pred[i] = qp
        v_pred[i] = qp / k_pts[i] if k_pts[i] > 0 else 0.0

    q_err = (q_pred - q_pts) * 3600
    v_err = (v_pred - v_pts) * 3.6
    rmse_q = float(np.sqrt(np.mean(q_err ** 2)))
    rmse_v = float(np.sqrt(np.mean(v_err ** 2)))
    r2_q = float(1 - np.sum(q_err ** 2) / (np.sum((q_pts * 3600 - (q_pts * 3600).mean()) ** 2) + 1e-12))
    r2_v = float(1 - np.sum(v_err ** 2) / (np.sum((v_pts * 3.6 - (v_pts * 3.6).mean()) ** 2) + 1e-12))
    return rmse_q, r2_q, rmse_v, r2_v, q_pred, v_pred


def _entropy_profile(pi_x):
    eps = 1e-10
    H = -np.sum(np.clip(pi_x, eps, 1.0) * np.log(np.clip(pi_x, eps, 1.0)), axis=1)
    return H


def _detect_transition_ped(pi_x, prototypes, k_pts, q_pts, x_pts, x_pos, dx, cells,
                           rho_obs=None):
    """PED-based transition detection (argmin PED within high-entropy zone).

    Fallback hierarchy when no high-entropy coexistence zone exists:
      1. Max |drho/dx| interior cell (density gradient peak, physically meaningful).
      2. Max-entropy interior cell (least homogeneous phase composition).
      3. Midpoint.
    """
    eps = 1e-10
    H = _entropy_profile(pi_x)
    tau_H = np.log(2)
    sigma_q = max(float(np.std(q_pts - q_pts.mean())), 1e-6)

    q_g_all = np.array([p.flow(k_pts) * 3600 for p in prototypes])
    bw = 4 * dx
    PED = np.ones(cells)
    for i in range(cells):
        mask = np.abs(x_pts - x_pos[i]) <= bw
        if mask.sum() < 2:
            continue
        q_pred_local = np.dot(pi_x[i], q_g_all[:, mask])
        res = q_pred_local - q_pts[mask] * 3600
        delta_F = float(np.mean(res ** 2)) / (sigma_q ** 2 + 1e-12)
        PED[i] = np.exp(-delta_F)

    omega = H >= tau_H
    margin = max(3, int(0.12 * cells))
    interior = np.zeros(cells, dtype=bool)
    interior[margin:-margin] = True
    cand = np.where(omega & interior)[0]

    if len(cand) == 0:
        # Fallback 1: steepest density gradient (requires rho_obs)
        if rho_obs is not None and len(rho_obs) == cells:
            grad_rho = np.abs(np.gradient(rho_obs.astype(float), dx))
            grad_rho[~interior] = 0.0
            if grad_rho.max() > 1e-9:
                return float(x_pos[int(np.argmax(grad_rho))])
        # Fallback 2: highest entropy interior cell
        H_int = H.copy()
        H_int[~interior] = -1.0
        return float(x_pos[int(np.argmax(H_int))])

    idx = int(cand[np.argmin(PED[cand])])
    return float(x_pos[idx])


def _phys_residual(pi_x, prototypes, rho_obs, rho_start, rho_end, dx, dt_window):
    """Mean squared conservation residual |drho/dt + dq/dx|."""
    if rho_start is None or rho_end is None or dt_window is None:
        return None
    M = len(rho_obs)
    q_pred = np.zeros(M)
    for g in range(len(prototypes)):
        q_pred += pi_x[:, g] * prototypes[g].flow(rho_obs)
    dq_dx = np.zeros(M)
    dq_dx[1:-1] = (q_pred[2:] - q_pred[:-2]) / (2 * dx)
    dq_dx[0] = (q_pred[1] - q_pred[0]) / dx
    dq_dx[-1] = (q_pred[-1] - q_pred[-2]) / dx
    drho_dt = (rho_end - rho_start) / dt_window
    return float(np.mean((drho_dt + dq_dx) ** 2))


# ============================================================================
# Baseline 1: Three-Phase PWA-CTM (white-box, hard + sigmoid blend)
# ============================================================================

def run_pwa_ctm(fd_points, prototypes, dx, cells, x_pos, rho_obs,
                seed=42, rho_start=None, rho_end=None, dt_window=None, **kw):
    """
    Three-Phase Piecewise-Affine Cell Transmission Model.

    Design principle: a CTM phase is a property of the CELL's macroscopic state
    (its bulk density rho_obs[c]), not of individual (k,q) point observations.

    Key insight: SpinFlow's EM-calibrated prototypes are soft mixture components
    and are NOT appropriate for hard assignment (hard-assigning proto[2] with
    Q0=500 veh/h to free-flow cells completely breaks predictions).  PWA-CTM
    therefore calibrates its OWN three prototypes from FD points that belong to
    spatially identified low / medium / high-density zones, matching the CTM
    rule "each cell is in its own phase based on measured bulk density".

    Adaptive blend_width at phase boundaries: width ∝ local rho_obs std so that
    transition zones get wider sigmoid blends and clear single-phase zones stay sharp.
    """
    from fd_model import calibrate_triangular_fd as _cal_fd

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    k_pts = fd_points['k'] / 1000.0   # veh/m
    q_pts = fd_points['q'] / 3600.0   # veh/s
    v_pts = fd_points['v'] / 3.6      # m/s
    x_pts = fd_points['x_center']     # m
    k_raw = fd_points['k']            # veh/km (for calibration)
    q_raw = fd_points['q']            # veh/h  (for calibration)
    n = len(k_pts)

    # Step 1: density zones from rho_obs
    # Divide cells into three equal-sized density zones.  Density percentiles
    # are computed on observed (non-zero) cells to avoid ghost lanes.
    rho_nonzero = rho_obs[rho_obs > 1e-6]
    if len(rho_nonzero) >= 3:
        r33 = np.percentile(rho_nonzero, 33)   # veh/m
        r67 = np.percentile(rho_nonzero, 67)
    else:
        r33 = rho_obs.mean() * 0.67
        r67 = rho_obs.mean() * 1.33

    # Assign each FD observation to a density zone via its cell's rho_obs
    cell_of_obs = np.clip((x_pts / dx).astype(int), 0, cells - 1)
    rho_at_obs = rho_obs[cell_of_obs]
    zone_masks = [
        rho_at_obs < r33,
        (rho_at_obs >= r33) & (rho_at_obs < r67),
        rho_at_obs >= r67,
    ]

    # Step 2: calibrate prototypes per zone
    # Each prototype is trained ONLY on its density zone's FD points, making it
    # appropriate for hard assignment in that zone (no mismatch with EM usage).
    pwa_protos = []
    for gm in zone_masks:
        if gm.sum() >= 10:
            p = _cal_fd(k_raw[gm], q_raw[gm])
        else:
            p = _cal_fd(k_raw, q_raw)          # fallback: single FD on all data
        pwa_protos.append(p)

    # Step 3: hard phase label per cell
    # Assign each cell to its density zone — simple O(M) rule.
    phase_map = np.zeros(cells, dtype=int)
    for c in range(cells):
        if rho_obs[c] < r33:
            phase_map[c] = 0
        elif rho_obs[c] < r67:
            phase_map[c] = 1
        else:
            phase_map[c] = 2

    # Smooth isolated single-cell phase islands (suppress spurious noise)
    for c in range(1, cells - 1):
        if phase_map[c - 1] == phase_map[c + 1] and phase_map[c] != phase_map[c - 1]:
            phase_map[c] = phase_map[c - 1]

    # Metastability: small probability of boundary cell inheriting its neighbour
    for c in range(1, cells):
        if phase_map[c] != phase_map[c - 1]:
            if rng.random() < 0.15:
                phase_map[c] = phase_map[c - 1]

    # Step 4: sigmoid blend at zone boundaries
    # Local rho_obs standard deviation over ±W cells drives the blend width:
    # high variance ↔ uncertain transition ↔ wider blend.
    W = max(3, int(cells * 0.05))
    rho_std_local = np.zeros(cells)
    for c in range(cells):
        lo_c = max(0, c - W)
        hi_c = min(cells, c + W + 1)
        rho_std_local[c] = float(np.std(rho_obs[lo_c:hi_c]))

    rho_std_ref = np.percentile(rho_std_local[rho_std_local > 0], 75) \
                  if (rho_std_local > 0).any() else 1.0

    base_w = max(2, int(cells * 0.04))
    max_w  = max(6, int(cells * 0.12))

    pi_x = np.zeros((cells, 3))
    for c in range(cells):
        pi_x[c, phase_map[c]] = 1.0

    for c in range(1, cells):
        if phase_map[c] != phase_map[c - 1]:
            p_prev, p_curr = phase_map[c - 1], phase_map[c]
            sigma_rel = rho_std_local[c] / (rho_std_ref + 1e-12)
            adapt = float(base_w + (max_w - base_w) * (2.0 / (1.0 + np.exp(-sigma_rel)) - 1.0))
            bw = max(base_w, int(round(adapt)))
            lo = max(0, c - bw)
            hi = min(cells, c + bw)
            for j in range(lo, hi):
                t = 1.0 / (1.0 + np.exp(-4.0 * (j - c) / bw))
                pi_x[j, :] = 0.0
                pi_x[j, p_prev] = 1.0 - t
                pi_x[j, p_curr] = t

    pi_x = np.clip(pi_x, 0, 1)
    pi_x /= pi_x.sum(axis=1, keepdims=True) + 1e-12

    # Use pwa_protos (not SpinFlow EM protos) for forward metrics — fair comparison
    rmse_q, r2_q, rmse_v, r2_v, _, _ = _forward_metrics(
        k_pts, q_pts, v_pts, x_pts, pi_x, pwa_protos, dx, cells)

    x_star = _detect_transition_ped(pi_x, pwa_protos, k_pts, q_pts, x_pts, x_pos, dx, cells,
                                    rho_obs=rho_obs)
    H = _entropy_profile(pi_x)
    phys_res = _phys_residual(pi_x, pwa_protos, rho_obs, rho_start, rho_end, dx, dt_window)
    runtime = time.perf_counter() - t0

    return {
        'rmse_q': rmse_q, 'r2_q': r2_q, 'rmse_v': rmse_v, 'r2_v': r2_v,
        'x_star': x_star, 'pi_x': pi_x,
        'phys_residual': phys_res, 'entropy_std': float(H.std()),
        'runtime_s': runtime, 'convergence_steps': 1,
    }


# ============================================================================
# Baseline 2: VBGMM + Spatial KDE + Prototype Projection (black-box)
# ============================================================================

def run_vbgmm(fd_points, prototypes, dx, cells, x_pos, rho_obs,
              seed=42, rho_start=None, rho_end=None, dt_window=None, **kw):
    from sklearn.mixture import BayesianGaussianMixture
    t0 = time.perf_counter()

    k_pts = fd_points['k'] / 1000.0
    q_pts = fd_points['q'] / 3600.0
    v_pts = fd_points['v'] / 3.6
    x_pts = fd_points['x_center']
    n = len(k_pts)

    x_max = x_pos.max() + dx
    features = np.column_stack([
        k_pts / (k_pts.max() + 1e-12),
        q_pts / (q_pts.max() + 1e-12),
        x_pts / (x_max + 1e-12),
    ])

    bgm = BayesianGaussianMixture(
        n_components=3, covariance_type='full',
        max_iter=500, random_state=seed, n_init=3,
    )
    bgm.fit(features)
    resp = bgm.predict_proba(features)
    n_iter = int(bgm.n_iter_)

    # Align VBGMM clusters to SpinFlow phase order by matching mean density
    cluster_mean_k = np.array([np.average(k_pts, weights=resp[:, c]) for c in range(3)])
    order = np.argsort(cluster_mean_k)
    perm = [None, None, None]
    perm[0] = int(order[0])   # lowest density -> free
    perm[1] = int(order[1])   # mid -> critical
    perm[2] = int(order[2])   # highest density -> congested
    resp_aligned = resp[:, perm]

    # Spatial KDE smoothing -> pi_x
    sigma_spatial = 2.0 * dx
    pi_x = np.zeros((cells, 3))
    w_sum = np.zeros(cells)
    for i in range(n):
        for c in range(cells):
            d = abs(x_pos[c] - x_pts[i])
            w = np.exp(-0.5 * (d / sigma_spatial) ** 2)
            if w > 1e-4:
                pi_x[c] += w * resp_aligned[i]
                w_sum[c] += w
    valid = w_sum > 0
    pi_x[valid] /= w_sum[valid, np.newaxis]
    pi_x[~valid] = 1.0 / 3.0
    pi_x /= pi_x.sum(axis=1, keepdims=True) + 1e-12

    # VBGMM: calibrate prototypes from hard cluster assignments
    # Using SpinFlow's shared prototypes causes large velocity errors when
    # VBGMM's statistical clusters mis-align with traffic phase boundaries.
    # Each cluster gets its own FD calibrated on the points it dominates (>0.4).
    from fd_model import calibrate_triangular_fd as _cal_fd
    hard_assign = np.argmax(resp_aligned, axis=1)
    vbgmm_protos = []
    for g in range(3):
        mask_g = hard_assign == g
        if mask_g.sum() >= 15:
            try:
                p = _cal_fd(fd_points['k'][mask_g], fd_points['q'][mask_g])
                vbgmm_protos.append(p)
            except Exception:
                vbgmm_protos.append(prototypes[g])
        else:
            vbgmm_protos.append(prototypes[g])   # fallback to SpinFlow proto

    # Prototype-projection post-processing using VBGMM's own prototypes
    for c in range(cells):
        q_proto = np.array([vbgmm_protos[g].flow(np.array([rho_obs[c]]))[0] for g in range(3)])
        local_mask = np.abs(x_pts - x_pos[c]) <= dx * 3
        if local_mask.sum() < 1:
            continue
        q_obs_local = q_pts[local_mask].mean()
        best_g = int(np.argmin(np.abs(q_proto - q_obs_local)))
        one_hot = np.zeros(3)
        one_hot[best_g] = 1.0
        pi_x[c] = 0.70 * pi_x[c] + 0.30 * one_hot

    pi_x /= pi_x.sum(axis=1, keepdims=True) + 1e-12

    rmse_q, r2_q, rmse_v, r2_v, _, _ = _forward_metrics(
        k_pts, q_pts, v_pts, x_pts, pi_x, vbgmm_protos, dx, cells)

    x_star = _detect_transition_ped(pi_x, vbgmm_protos, k_pts, q_pts, x_pts, x_pos, dx, cells,
                                    rho_obs=rho_obs)
    H = _entropy_profile(pi_x)
    phys_res = _phys_residual(pi_x, vbgmm_protos, rho_obs, rho_start, rho_end, dx, dt_window)
    runtime = time.perf_counter() - t0

    return {
        'rmse_q': rmse_q, 'r2_q': r2_q, 'rmse_v': rmse_v, 'r2_v': r2_v,
        'x_star': x_star, 'pi_x': pi_x,
        'phys_residual': phys_res, 'entropy_std': float(H.std()),
        'runtime_s': runtime, 'convergence_steps': n_iter,
    }


# ============================================================================
# Baseline 3: PI-DeepONet-Traffic (PIML, operator learning + early stopping)
# ============================================================================

def run_pi_deeponet(fd_points, prototypes, dx, cells, x_pos, rho_obs,
                    seed=42, rho_start=None, rho_end=None, dt_window=None,
                    epochs=2000, lr=1e-3, lam_pde=5.0, **kw):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    t0 = time.perf_counter()

    k_pts = fd_points['k'] / 1000.0
    q_pts = fd_points['q'] / 3600.0
    v_pts = fd_points['v'] / 3.6
    x_pts = fd_points['x_center']
    n = len(k_pts)

    x_max = x_pos.max() + dx
    rho_norm = k_pts / (k_pts.max() + 1e-12)
    x_norm = x_pts / (x_max + 1e-12)
    q_target = q_pts.copy()

    X_branch = torch.tensor(np.column_stack([rho_norm, x_norm]), dtype=torch.float32)
    X_trunk = torch.tensor(x_norm.reshape(-1, 1), dtype=torch.float32)
    Y = torch.tensor(q_target, dtype=torch.float32)

    class BranchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 64),
            )
        def forward(self, x):
            return self.net(x)

    class TrunkNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 64),
            )
        def forward(self, x):
            return self.net(x)

    branch = BranchNet()
    trunk = TrunkNet()
    params = list(branch.parameters()) + list(trunk.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    # PDE collocation
    rho_grid = torch.tensor(rho_obs, dtype=torch.float32)
    rho_grid_norm = rho_grid / (rho_grid.max() + 1e-12)
    x_grid_norm = torch.tensor(x_pos / (x_max + 1e-12), dtype=torch.float32).requires_grad_(True)

    # Convergence: stop when rolling mean of loss over last `window` epochs
    # changes by less than `conv_tol` relative to the window before that.
    # 5e-3 (~0.5% relative plateau) is a practical threshold for FD operator nets.
    conv_window = 100
    conv_tol = 5e-3
    loss_history = []
    actual_epochs = epochs

    for epoch in range(epochs):
        opt.zero_grad()
        b_out = branch(X_branch)
        t_out = trunk(X_trunk)
        q_hat = (b_out * t_out).sum(dim=1)
        loss_data = ((q_hat - Y) ** 2).mean()

        x_in = x_grid_norm.unsqueeze(1)
        rho_in = torch.stack([rho_grid_norm, x_grid_norm], dim=1)
        b_g = branch(rho_in)
        t_g = trunk(x_in)
        q_g = (b_g * t_g).sum(dim=1)
        dq_dx = torch.autograd.grad(q_g.sum(), x_grid_norm, create_graph=True)[0]
        loss_pde = (dq_dx ** 2).mean()

        loss = loss_data + lam_pde * loss_pde
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        # Check convergence every `conv_window` epochs after 2×window epochs
        if len(loss_history) >= 2 * conv_window and epoch % conv_window == 0:
            mean_new = float(np.mean(loss_history[-conv_window:]))
            mean_old = float(np.mean(loss_history[-2*conv_window:-conv_window]))
            rel_change = abs(mean_old - mean_new) / (abs(mean_old) + 1e-12)
            if rel_change < conv_tol:
                actual_epochs = epoch + 1
                break

    # Evaluate on FD points
    with torch.no_grad():
        b_out = branch(X_branch)
        t_out = trunk(X_trunk)
        q_hat_np = (b_out * t_out).sum(dim=1).numpy()

    q_err = (q_hat_np - q_pts) * 3600
    v_pred = np.where(k_pts > 0, q_hat_np / k_pts, 0.0)
    v_err = (v_pred - v_pts) * 3.6
    rmse_q = float(np.sqrt(np.mean(q_err ** 2)))
    rmse_v = float(np.sqrt(np.mean(v_err ** 2)))
    r2_q = float(1 - np.sum(q_err ** 2) / (np.sum((q_pts * 3600 - (q_pts * 3600).mean()) ** 2) + 1e-12))
    r2_v = float(1 - np.sum(v_err ** 2) / (np.sum((v_pts * 3.6 - (v_pts * 3.6).mean()) ** 2) + 1e-12))

    # Construct pseudo-pi_x from network predictions to enable PED / entropy / transition
    # For each spatial cell, assign phase weights proportional to softmax of -|q_pred - q_g(rho)|
    pi_x = np.zeros((cells, 3))
    with torch.no_grad():
        x_grid_eval = torch.tensor(x_pos / (x_max + 1e-12), dtype=torch.float32)
        rho_eval = torch.tensor(rho_obs / (k_pts.max() + 1e-12), dtype=torch.float32)
        br_in = torch.stack([rho_eval, x_grid_eval], dim=1)
        tr_in = x_grid_eval.unsqueeze(1)
        q_grid_pred = (branch(br_in) * trunk(tr_in)).sum(dim=1).numpy()

    for c in range(cells):
        q_proto = np.array([prototypes[g].flow(np.array([rho_obs[c]]))[0] for g in range(3)])
        dists = -np.abs(q_grid_pred[c] - q_proto) * 1e4
        dists -= dists.max()
        w = np.exp(dists)
        pi_x[c] = w / (w.sum() + 1e-12)

    x_star = _detect_transition_ped(pi_x, prototypes, k_pts, q_pts, x_pts, x_pos, dx, cells,
                                    rho_obs=rho_obs)
    H = _entropy_profile(pi_x)
    phys_res = _phys_residual(pi_x, prototypes, rho_obs, rho_start, rho_end, dx, dt_window)
    runtime = time.perf_counter() - t0

    return {
        'rmse_q': rmse_q, 'r2_q': r2_q, 'rmse_v': rmse_v, 'r2_v': r2_v,
        'x_star': x_star, 'pi_x': pi_x,
        'phys_residual': phys_res, 'entropy_std': float(H.std()),
        'runtime_s': runtime, 'convergence_steps': actual_epochs,
    }
