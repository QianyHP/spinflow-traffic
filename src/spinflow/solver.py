"""
SpinFlow inversion engine: EM updates for the latent spin field and triangular FD prototypes.

E-step: responsibilities from Gaussian flow likelihood on FD points; spatial kernel pooling
builds a target mixture pi_target(x); cross-entropy pulls spin-induced pi(x) toward that
target, with optional CTM-style mass-conservation residual and a discrete Laplacian on s.

M-step: re-fit each triangular FD with weights responsibility x point quality.

Spin update: map total_grad on mixture logits back through the fixed antisymmetric spin-to-logits
map (no unit-norm projection on s—“phase unfolding” in the paper narrative).
"""

import numpy as np
from typing import List, Dict, Tuple
from .fd_model import (
    TriangularFD,
    calibrate_triangular_fd,
    calibrate_three_prototypes,
    PHASE_INDEX,
    PHASE_DISPLAY,
)
from .phase_utils import spin_to_mixture_weights_softmax


DEFAULT_SIGMA_SPATIAL_FACTOR = 1.5  # dimensionless, sigma_spatial = factor * dx

def estimate_observation_noise(fd_points: Dict, prototypes: List[TriangularFD]) -> float:
    """
    Data-driven flow scale sigma_q: std of q residuals to nearest prototype in q.

    Used to normalize FD likelihood terms; should track observation noise rather than
    spread due to heterogeneous phases (hence residual to a local prototype fit).
    Returns sigma_q in veh/s (SI).
    """
    k_obs = fd_points['k'] / 1000.0  # veh/m
    q_obs = fd_points['q'] / 3600.0  # veh/s
    n_points = len(k_obs)
    
    residuals_q = []
    
    for i in range(n_points):
        k_p = k_obs[i]
        q_p = q_obs[i]
        
        min_dist = float('inf')
        for proto in prototypes:
            q_proto = proto.flow(np.array([k_p]))[0]
            dist = abs(q_proto - q_p)
            if dist < min_dist:
                min_dist = dist
                best_q = q_proto
        
        residuals_q.append(q_p - best_q)
    
    sigma_q = np.std(residuals_q)
    return sigma_q


def e_step(fd_points: Dict, 
          prototypes: List[TriangularFD],
          sx: np.ndarray, sy: np.ndarray, sz: np.ndarray,
          dx: float,
          lam_fd_q: float = 1.0,
          lam_smo: float = 0.0,
          rho_obs: np.ndarray = None,
          rho_start: np.ndarray = None,
          rho_end: np.ndarray = None,
          dt_window: float = 60.0,
          lam_phys: float = 0.0,
          quality_weights: np.ndarray = None,
          sigma_q: float = None) -> Tuple:
    """
    E-step: responsibilities, pooled pi_target, losses, and gradient w.r.t. mixture logits.

    Total objective combines normalized flow RMSE at FD points, cross-entropy(pi, pi_target),
    optional discrete conservation residual, and spin Laplacian (lam_smo).
    """
    M = len(sx)
    n_prototypes = len(prototypes)
    
    if sigma_q is None:
        raise ValueError("sigma_q must be provided (call estimate_observation_noise)")
    
    pi = spin_to_mixture_weights_softmax(sx, sy, sz, n_prototypes)
    
    k_points = fd_points['k'] / 1000.0  # veh/m
    q_points = fd_points['q'] / 3600.0  # veh/s
    x_points = fd_points['x_center']    # m
    n_points = len(k_points)
    
    if quality_weights is None:
        quality_weights = np.ones(n_points)
    
    # q_g_all[g, p]: flow from prototype g at FD point p
    q_g_all = np.stack([prototypes[g].flow(k_points) for g in range(n_prototypes)], axis=0)
    log_resp = -0.5 * ((q_points[np.newaxis, :] - q_g_all) / sigma_q) ** 2  # [n_proto, n_pts]
    log_resp = log_resp.T  # [n_points, n_prototypes]
    log_resp -= log_resp.max(axis=1, keepdims=True)
    responsibilities = np.exp(log_resp)
    resp_sum = responsibilities.sum(axis=1, keepdims=True) + 1e-12
    responsibilities /= resp_sum
    
    sigma_spatial = DEFAULT_SIGMA_SPATIAL_FACTOR * dx
    x_cells = np.arange(M, dtype=np.float64) * dx  # [M]

    dists = np.abs(x_points[:, np.newaxis] - x_cells[np.newaxis, :])  # [n_pts, M]
    spatial_w = np.exp(-0.5 * (dists / sigma_spatial) ** 2)           # [n_pts, M]
    spatial_w[spatial_w < 1e-4] = 0.0

    w_full = spatial_w * quality_weights[:, np.newaxis]  # [n_pts, M]

    # Gaussian-smooth FD evidence onto the road grid: each cell gets a weighted average of
    # prototype responsibilities from nearby FD points, so π_target(x) varies smoothly in x
    # instead of collapsing to isolated spikes at sample locations.
    pi_weights = w_full.sum(axis=0)           # [M]
    pi_target  = w_full.T @ responsibilities  # [M, n_prototypes]

    mask_valid = pi_weights > 0
    pi_target[mask_valid] /= pi_weights[mask_valid, np.newaxis]
    
    if not np.all(mask_valid):
        valid_indices = np.where(mask_valid)[0]
        all_indices = np.arange(M)
        for g in range(n_prototypes):
            if len(valid_indices) > 0:
                pi_target[:, g] = np.interp(all_indices, valid_indices, pi_target[valid_indices, g])
            else:
                pi_target[:, g] = 1.0 / n_prototypes
    
    pi_target /= pi_target.sum(axis=1, keepdims=True) + 1e-12

    # === π matching loss (cross-entropy) ===
    # Physics meaning: enforce that the spin-induced mixture π(x) explains the
    # posterior evidence aggregated from FD points (pi_target).
    pi_safe = np.clip(pi, 1e-12, 1.0)
    loss_pi = float(-np.mean(np.sum(pi_target * np.log(pi_safe), axis=1)))

    # === Heisenberg exchange (spin smoothness) ===
    # E_H = sum_i ||s_{i+1} - s_i||^2  (encourages spatial coherence)
    dsx = sx[1:] - sx[:-1]
    dsy = sy[1:] - sy[:-1]
    dsz = sz[1:] - sz[:-1]
    loss_smooth = float(np.mean(dsx**2 + dsy**2 + dsz**2))
    
    cell_indices = np.clip((x_points / dx).astype(int), 0, M - 1)  # [n_points]
    pi_at_pts = pi[cell_indices]          # [n_points, n_prototypes]
    q_pred_pts = np.einsum('pg,gp->p', pi_at_pts, q_g_all)  # [n_points]
    loss_fd_q = float(
        np.mean(quality_weights * ((q_pred_pts - q_points) / sigma_q) ** 2)
    )
    
    # Cross-entropy w.r.t. softmax logits: grad = pi - pi_target
    grad_logits = (pi - pi_target)  # [M, 3]

    # Optional discrete continuity residual: drho/dt + dq/dx (see paper / CTM discretization).
    loss_phys = 0.0
    grad_logits_phys = np.zeros((M, n_prototypes)) 

    if lam_phys > 0 and rho_start is not None and rho_end is not None:
        q_pred_cells = np.zeros(M)
        q_g_list = []
        for g in range(n_prototypes):
            q_g = prototypes[g].flow(rho_obs)
            q_g_list.append(q_g)
            q_pred_cells += pi[:, g] * q_g
            
        dq_dx = np.zeros(M)
        dq_dx[1:-1] = (q_pred_cells[2:] - q_pred_cells[:-2]) / (2 * dx)
        dq_dx[0] = (q_pred_cells[1] - q_pred_cells[0]) / dx
        dq_dx[-1] = (q_pred_cells[-1] - q_pred_cells[-2]) / dx
        
        drho_dt = (rho_end - rho_start) / dt_window
        conservation_resid = drho_dt + dq_dx # [M]
        loss_phys = np.mean(conservation_resid**2)

        # ∂L/∂q from squared residual, then chain rule to logits via ∂q/∂π and softmax Jacobian
        # (each cell: dq/dπ_g = q_g - q_pred on the mixture flow).
        grad_q = np.zeros(M)
        grad_q[1:-1] = (conservation_resid[:-2] - conservation_resid[2:]) / (M * dx)
        grad_q[0] = conservation_resid[0] / (M * dx)
        grad_q[-1] = -conservation_resid[-1] / (M * dx)
        
        for g in range(n_prototypes):
            grad_logits_phys[:, g] = grad_q * pi[:, g] * (q_g_list[g] - q_pred_cells)

    total_grad_logits = lam_fd_q * grad_logits + lam_phys * grad_logits_phys

    total_loss = (lam_fd_q * loss_fd_q +
                  lam_phys * loss_phys +
                  lam_fd_q * loss_pi +
                  lam_smo * loss_smooth)
    
    loss_components = {
        'loss_fd_q': loss_fd_q,
        'loss_phys': loss_phys,
        'loss_pi': loss_pi,
        'loss_smooth': loss_smooth,
    }
    
    return pi, total_grad_logits, total_loss, responsibilities, loss_components


def m_step(fd_points: Dict,
           responsibilities: np.ndarray,
           n_prototypes: int = 3,
           prev_prototypes: List[TriangularFD] = None,
           quality_weights: np.ndarray = None) -> List[TriangularFD]:
    """
    M-step: re-calibrate each triangular FD with weights resp_pg * quality_p;
    warm-start from prev_prototypes when available.
    """
    k_points = fd_points['k']
    q_points = fd_points['q']
    
    if quality_weights is None:
        quality_weights = np.ones(len(k_points))
    
    prototypes = []
    
    for g in range(n_prototypes):
        weights_g = responsibilities[:, g] * quality_weights
        
        init_params = None
        if prev_prototypes is not None and g < len(prev_prototypes):
            init_params = prev_prototypes[g].to_dict()
        
        fd_g = calibrate_triangular_fd(k_points, q_points, weights=weights_g, init_params=init_params)
        prototypes.append(fd_g)
    
    return prototypes


def update_spin_from_logits_grad(grad_logits: np.ndarray,
                                sx: np.ndarray,
                                sy: np.ndarray,
                                sz: np.ndarray,
                                lam_smo: float = 0.0,
                                step_size: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One gradient step on (s_x, s_y, s_z) from gradients w.r.t. mixture logits.

    Logits use the same antisymmetric map as spin_to_mixture_weights_softmax; chain rule
    folds dL/dlogits into dL/ds. No projection back to |s|=1 (phase unfolding).
    Adds discrete Laplacian regularization on s when lam_smo > 0.

    Args:
        grad_logits: dL/d(logits), shape [M, n_prototypes].
        sx, sy, sz: Current spin components [M].
        lam_smo: Weight on squared finite-difference penalty on s.
        step_size: Gradient-descent step.

    Returns:
        Updated sx, sy, sz.
    """
    M = len(sx)
    n_prototypes = grad_logits.shape[1]
    

    
    competition_sign = np.sign(sz - sy)  # +1 if sz>sy, -1 if sz<sy, 0 if sz=sy
    
    grad_sz = (grad_logits[:, 0] 
               - grad_logits[:, 1] * competition_sign 
               - grad_logits[:, 2])
    
    grad_sx = grad_logits[:, 1]
    
    grad_sy = (-grad_logits[:, 0] 
               + grad_logits[:, 1] * competition_sign 
               + grad_logits[:, 2])
    
    grad_sx_total = grad_sx
    grad_sy_total = grad_sy
    grad_sz_total = grad_sz

    # Heisenberg exchange gradient: ∂/∂s_i sum ||s_{i+1}-s_i||^2
    # interior: 2*(2 s_i - s_{i-1} - s_{i+1})
    # boundaries: 2*(s_0 - s_1), 2*(s_{M-1} - s_{M-2})
    if lam_smo > 0 and M >= 2:
        lap_sx = np.zeros(M)
        lap_sy = np.zeros(M)
        lap_sz = np.zeros(M)

        lap_sx[1:-1] = 2 * sx[1:-1] - sx[:-2] - sx[2:]
        lap_sy[1:-1] = 2 * sy[1:-1] - sy[:-2] - sy[2:]
        lap_sz[1:-1] = 2 * sz[1:-1] - sz[:-2] - sz[2:]

        lap_sx[0] = sx[0] - sx[1]
        lap_sy[0] = sy[0] - sy[1]
        lap_sz[0] = sz[0] - sz[1]

        lap_sx[-1] = sx[-1] - sx[-2]
        lap_sy[-1] = sy[-1] - sy[-2]
        lap_sz[-1] = sz[-1] - sz[-2]

        grad_sx_total += 2.0 * lam_smo * lap_sx
        grad_sy_total += 2.0 * lam_smo * lap_sy
        grad_sz_total += 2.0 * lam_smo * lap_sz
    
    sx_new = sx - step_size * grad_sx_total
    sy_new = sy - step_size * grad_sy_total
    sz_new = sz - step_size * grad_sz_total
    
    return sx_new, sy_new, sz_new


def em_inverse_fd(fd_points: Dict,
                   dx: float,
                   M: int,
                   rho_obs: np.ndarray,
                   v_obs: np.ndarray = None,
                   rho_start: np.ndarray = None,
                   rho_end: np.ndarray = None,
                   dt_window: float = None,
                   n_prototypes: int = None,
                   em_iters: int = None,
                   inner_iters: int = None,
                   lam_fd_q: float = None,
                   lam_phys: float = None,
                   lam_smo: float = 0.0,
                   convergence_tol: float = None,
                   learning_rate: float = None,
                   seed: int = 123) -> Tuple:
    """
    EM inversion of spin field s(x) with observed density rho_obs(x) fixed.

    Alternates E-step (gradients on logits / s) and M-step (prototype re-fit).
    Initializes s from a Boltzmann prior over distances to K-means attractors in
    normalized (rho, v) space with phase-dependent inverse temperatures.

    Returns optimized s, final prototypes, rho0 == rho_obs, and loss traces.
    """

    rng = np.random.default_rng(seed)
    
    print("\n" + "="*80)
    print("EM-FD Spin Field Inversion (Data-Driven, Minimal Subjective Parameters)")
    print("="*80)
    print(f"Theory: Statistical Physics + Kerner Three-Phase")
    print(f"Philosophy: Let data guide the inversion, not subjective priors")
    print(f"")
    print(f"Data-driven components:")
    print(f"  - Initial phase: inferred from rho_obs quantiles")
    print(f"  - Spin radius: Unconstrained (Phase Unfolding)")
    print(f"  - Loss weights: Fixed prior (Robust balancing)")
    print(f"  - Phase evolution: pure FD matching")
    print(f"")
    print(f"Minimal subjective parameters:")
    print(f"  - Radius regularization: Removed (Free energy landscape)")
    print(f"  - Learning rate: α=0.05 (standard gradient descent)")
    print(f"")
    print(f"Inversion target: Spin field s(x) ONLY")
    print(f"Density field ρ(x): From observation (Edie method)")
    print(f"Spatial cells: {M}")
    print(f"FD observation points: {fd_points['n_points']}")
    print(f"Prototypes: {n_prototypes}")
    print(f"EM iterations: {em_iters}")
    print(f"Loss weights: λ_q={lam_fd_q}, λ_phys={lam_phys}")
    print(f"Convergence tolerance: {convergence_tol}")
    print(f"Observed density: mean={rho_obs.mean():.4f}, std={rho_obs.std():.4f}")
    
    prototypes, attractor_info = calibrate_three_prototypes(fd_points, n_prototypes, verbose=False)
    
    sigma_q_adaptive = estimate_observation_noise(fd_points, prototypes)
    
    print(f"\n[Adaptive Normalization] Flow noise estimation:")
    print(f"  σ_q = {sigma_q_adaptive:.4f} veh/s ({sigma_q_adaptive*3600:.1f} veh/h)")
    print(f"  → Estimated from prototype fitting residuals")
    
    if 'weight' in fd_points and fd_points['weight'] is not None:
        quality_weights_global = fd_points['weight']
        quality_weights_global = quality_weights_global / quality_weights_global.mean()
        print(f"\n[Quality Weighting] Using soft-threshold weights from sampler")
        print(f"  Weight range: [{quality_weights_global.min():.3f}, {quality_weights_global.max():.3f}]")
    else:
        quality_weights_global = None
        print(f"\n[Quality Weighting] No weights provided")
    
    print("\n[Initialization] Mapping Phase Attractor Field...")
    
    centroids = attractor_info['centroids']  # canonical order: Free, Critical, Congested
    sigmas_sq = attractor_info['sigmas']
    rho_min, rho_max, v_min, v_max = attractor_info['scalers']
    
    v_input_scaled = v_obs
    
    rho_norm = (rho_obs - rho_min) / (rho_max - rho_min)
    v_norm = (v_input_scaled - v_min) / (v_max - v_min)
    
    betas = 0.5 / (sigmas_sq + 1e-6)
    betas = np.clip(betas, 1.0, 100.0)
    
    print(
        "  Adaptive Temperature (Beta) per phase: "
        f"{PHASE_DISPLAY['free']}={betas[PHASE_INDEX['free']]:.1f}, "
        f"{PHASE_DISPLAY['critical']}={betas[PHASE_INDEX['critical']]:.1f}, "
        f"{PHASE_DISPLAY['congested']}={betas[PHASE_INDEX['congested']]:.1f}"
    )
    
    sx = np.zeros(M)
    sy = np.zeros(M)
    sz = np.zeros(M)
    
    X_state = np.column_stack([rho_norm, v_norm])
    dists_sq = np.zeros((M, n_prototypes))
    for g in range(n_prototypes):
        dists_sq[:, g] = np.sum((X_state - centroids[g])**2, axis=1)
        
    energies = dists_sq * betas[np.newaxis, :]
    
    min_E = np.min(energies, axis=1, keepdims=True)
    probs = np.exp(-(energies - min_E))
    probs /= (np.sum(probs, axis=1, keepdims=True) + 1e-12)
    
    pi_F = probs[:, PHASE_INDEX["free"]]
    pi_S = probs[:, PHASE_INDEX["critical"]]
    pi_J = probs[:, PHASE_INDEX["congested"]]
    
    scale = 2.0
    eps = 1e-4
    pi_F = np.maximum(pi_F, eps)
    pi_J = np.maximum(pi_J, eps)
    pi_S = np.maximum(pi_S, eps)
    
    b = scale * np.log(pi_F / pi_J)
    noise = rng.standard_normal(M) * 0.05
    
    sz = 0.5 * b + noise
    sy = -0.5 * b + noise
    sx = scale * np.log(pi_S) + np.abs(b) + noise

    print(f"  Initialized Spin Field using Structural Entropy (Adaptive Beta)")
    print(f"  - Consistent with Kerner's phase variance properties")
    
    history = {
        'loss': [],
        'loss_fd_q': [],
        'loss_phys': [],
        'loss_pi': [],
        'loss_smooth': [],
    }
    
    prev_loss = float('inf')
    converged = False
    
    for em_iter in range(em_iters):
        show_iter = (em_iter % 10 == 0) or (em_iter == em_iters - 1)
        if show_iter:
            print(f"\n--- EM Iteration {em_iter+1}/{em_iters} ---")
        
        for inner_it in range(inner_iters):
            pi, grad_logits, loss, resp, loss_comp = e_step(
                fd_points, prototypes, sx, sy, sz, dx,
                lam_fd_q=lam_fd_q,
                lam_smo=lam_smo,
                rho_obs=rho_obs,
                rho_start=rho_start,
                rho_end=rho_end,
                dt_window=dt_window,
                lam_phys=lam_phys,
                quality_weights=quality_weights_global,
                sigma_q=sigma_q_adaptive
            )
            
            step_size = learning_rate
            
            sx, sy, sz = update_spin_from_logits_grad(
                grad_logits, sx, sy, sz,
                lam_smo=lam_smo,
                step_size=step_size
            )
        
        history['loss'].append(loss)
        history['loss_fd_q'].append(loss_comp['loss_fd_q'])
        history['loss_phys'].append(loss_comp['loss_phys'])
        history['loss_pi'].append(loss_comp['loss_pi'])
        history['loss_smooth'].append(loss_comp['loss_smooth'])
        
        if em_iter > 0:
            loss_change = abs(prev_loss - loss) / (abs(prev_loss) + 1e-12)
            if show_iter:
                print(f"  Loss: {loss:.6f}, Change: {loss_change:.6f}")
            
            if loss_change < convergence_tol:
                print(f"\n  - Converged at iter {em_iter+1}! Loss change {loss_change:.6f} < {convergence_tol}")
                converged = True
        elif show_iter:
            print(f"  Loss: {loss:.6f}")
        
        prev_loss = loss
        
        prototypes = m_step(fd_points, resp, n_prototypes, 
                          prev_prototypes=prototypes,
                          quality_weights=quality_weights_global)
        
        if converged:
            break
    
    rho0 = rho_obs
    
    print("\n" + "="*80)
    if converged:
        print(f"Spin Field Inversion CONVERGED at iteration {em_iter+1}/{em_iters}")
    else:
        print(f"Spin Field Inversion Completed (max iterations reached)")
    print("="*80)
    
    return sx, sy, sz, prototypes, rho0, history
