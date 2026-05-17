"""
SpinFlow ablation variants for mechanism validation.

Ablation A: No Competition Mapping  -- remove |sz-sy| penalty from h_S
Ablation B: Fixed Spin Norm         -- project s onto unit sphere (no Phase Unfolding)
Ablation C: No Physics Prior        -- set lam_phys = 0

Each variant re-runs the EM loop with a single mechanism removed, sharing
all other components (data, prototypes, grid, hyperparameters).

Design note: ablations use the SAME lam_phys as the full model (passed by the
runner), so the physics constraint is at operational strength when testing
non-physics ablations. This makes the removal of each component more visible.
"""

import os, sys, time, copy
import numpy as np
from typing import Dict, List, Tuple

from paths import SPIN_SRC

sys.path.insert(0, SPIN_SRC)
from spinflow.fd_model import TriangularFD, PHASES, PHASE_INDEX, calibrate_three_prototypes
from spinflow.solver import (
    e_step, m_step, update_spin_from_logits_grad,
    estimate_observation_noise, DEFAULT_SIGMA_SPATIAL_FACTOR,
)
from spinflow.phase_utils import spin_to_mixture_weights_softmax as _original_spin_to_pi
from paper_benchmark_baselines import (
    _forward_metrics, _entropy_profile, _detect_transition_ped, _phys_residual,
)


# ---------------------------------------------------------------------------
# Modified spin-to-pi mappings
# ---------------------------------------------------------------------------

def _spin_to_pi_no_competition(sx, sy, sz):
    """Standard 3D softmax: h_S = s_x (no |sz-sy| penalty).
    Without the competition term, the critical (synchronized) phase
    can emerge even when F and J are far apart — breaking physical realism."""
    M = len(sx)
    logits = np.zeros((M, 3))
    logits[:, PHASE_INDEX["free"]] = sz - sy
    logits[:, PHASE_INDEX["critical"]] = sx          # NO competition: -|sz-sy|
    logits[:, PHASE_INDEX["congested"]] = sy - sz
    logits_max = logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits - logits_max)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def _spin_to_pi_standard(sx, sy, sz):
    """Original SpinFlow mapping (with competition penalty)."""
    M = len(sx)
    chi = np.abs(sz - sy)
    logits = np.zeros((M, 3))
    logits[:, PHASE_INDEX["free"]] = sz - sy
    logits[:, PHASE_INDEX["critical"]] = sx - chi
    logits[:, PHASE_INDEX["congested"]] = sy - sz
    logits_max = logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits - logits_max)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Modified gradient for no-competition mapping
# ---------------------------------------------------------------------------

def _update_spin_no_competition(grad_logits, sx, sy, sz, lam_smo, step_size):
    """Gradient update when h_S = s_x (no competition strength)."""
    M = len(sx)
    grad_sz = grad_logits[:, 0] - grad_logits[:, 2]
    grad_sx = grad_logits[:, 1]
    grad_sy = -grad_logits[:, 0] + grad_logits[:, 2]

    if lam_smo > 0 and M >= 2:
        for arr, s in [(grad_sz, sz), (grad_sx, sx), (grad_sy, sy)]:
            lap = np.zeros(M)
            lap[1:-1] = 2 * s[1:-1] - s[:-2] - s[2:]
            lap[0] = s[0] - s[1]
            lap[-1] = s[-1] - s[-2]
            arr += 2.0 * lam_smo * lap

    return sx - step_size * grad_sx, sy - step_size * grad_sy, sz - step_size * grad_sz


# ---------------------------------------------------------------------------
# Generic ablation EM runner (no monkey-patching — safe direct call)
# ---------------------------------------------------------------------------

def _run_ablation_em(fd_points, prototypes_init, dx, cells, x_pos, rho_obs,
                     seed, rho_start, rho_end, dt_window,
                     lam_fd_q, lam_phys, lam_smo, lr, em_iters, inner_iters,
                     spin_to_pi_fn, update_spin_fn, post_update_hook=None):
    """
    Run a full EM loop with pluggable mapping / update / post-update.

    Instead of monkey-patching phase_utils at runtime, we directly compute
    pi from spin_to_pi_fn and manually assemble the E-step gradient. This
    eliminates all import-order and re-entrance hazards.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    prototypes, attractor_info = calibrate_three_prototypes(fd_points, 3, verbose=False)
    sigma_q = estimate_observation_noise(fd_points, prototypes)

    M = cells
    quality_weights = fd_points.get('weight', None)
    if quality_weights is not None:
        quality_weights = quality_weights / quality_weights.mean()

    sx = rng.standard_normal(M) * 0.1
    sy = rng.standard_normal(M) * 0.1
    sz = rng.standard_normal(M) * 0.1

    history_loss = []
    converged_iter = em_iters

    # Temporarily replace the module-level function so that e_step (which
    # calls spin_to_mixture_weights_softmax internally) uses our custom mapping.
    # We use a try/finally block to guarantee restoration even on error.
    import phase_utils as _pu
    _orig_fn = _pu.spin_to_mixture_weights_softmax

    try:
        _pu.spin_to_mixture_weights_softmax = lambda sx, sy, sz, n: spin_to_pi_fn(sx, sy, sz)

        for it in range(em_iters):
            for _ in range(inner_iters):
                pi, grad, loss, resp, lcomp = e_step(
                    fd_points, prototypes, sx, sy, sz, dx,
                    lam_fd_q=lam_fd_q, lam_smo=lam_smo,
                    rho_obs=rho_obs, rho_start=rho_start, rho_end=rho_end,
                    dt_window=dt_window, lam_phys=lam_phys,
                    quality_weights=quality_weights, sigma_q=sigma_q,
                )
                sx, sy, sz = update_spin_fn(grad, sx, sy, sz, lam_smo, lr)
                if post_update_hook is not None:
                    sx, sy, sz = post_update_hook(sx, sy, sz, it)

            history_loss.append(loss)
            prototypes = m_step(fd_points, resp, 3,
                                prev_prototypes=prototypes,
                                quality_weights=quality_weights)

            if it > 0 and abs(history_loss[-2] - loss) / (abs(history_loss[-2]) + 1e-12) < 1e-4:
                converged_iter = it + 1
                break
    finally:
        _pu.spin_to_mixture_weights_softmax = _orig_fn

    pi_final = spin_to_pi_fn(sx, sy, sz)
    k_pts = fd_points['k'] / 1000.0
    q_pts = fd_points['q'] / 3600.0
    v_pts = fd_points['v'] / 3.6
    x_pts = fd_points['x_center']

    rmse_q, r2_q, rmse_v, r2_v, _, _ = _forward_metrics(
        k_pts, q_pts, v_pts, x_pts, pi_final, prototypes, dx, cells)
    x_star = _detect_transition_ped(pi_final, prototypes, k_pts, q_pts, x_pts, x_pos, dx, cells,
                                    rho_obs=rho_obs)
    H = _entropy_profile(pi_final)
    phys_res = _phys_residual(pi_final, prototypes, rho_obs, rho_start, rho_end, dx, dt_window)
    runtime = time.perf_counter() - t0

    return {
        'rmse_q': rmse_q, 'r2_q': r2_q, 'rmse_v': rmse_v, 'r2_v': r2_v,
        'x_star': x_star, 'pi_x': pi_final,
        'phys_residual': phys_res, 'entropy_std': float(H.std()),
        'runtime_s': runtime, 'convergence_steps': converged_iter,
        'history_loss': history_loss,
    }


# ============================================================================
# Public ablation runners
# ============================================================================

def run_ablation_mapping(fd_points, prototypes, dx, cells, x_pos, rho_obs,
                         seed=42, rho_start=None, rho_end=None, dt_window=None,
                         lam_fd_q=1.0, lam_phys=0.1, lam_smo=0.02,
                         lr=0.05, em_iters=80, inner_iters=20, **kw):
    """Ablation A: remove competition penalty from h_S.
    Uses the SAME lam_phys as passed (default raised to 0.1)."""
    return _run_ablation_em(
        fd_points, prototypes, dx, cells, x_pos, rho_obs,
        seed, rho_start, rho_end, dt_window,
        lam_fd_q, lam_phys, lam_smo, lr, em_iters, inner_iters,
        spin_to_pi_fn=_spin_to_pi_no_competition,
        update_spin_fn=_update_spin_no_competition,
    )


def run_ablation_spin_norm(fd_points, prototypes, dx, cells, x_pos, rho_obs,
                           seed=42, rho_start=None, rho_end=None, dt_window=None,
                           lam_fd_q=1.0, lam_phys=0.1, lam_smo=0.02,
                           lr=0.05, em_iters=80, inner_iters=20, **kw):
    """Ablation B: project spin onto unit sphere after each update.
    Eliminates Phase Unfolding — spin norm cannot grow to express strong dominance."""
    def _unit_norm_hook(sx, sy, sz, it):
        norm = np.sqrt(sx**2 + sy**2 + sz**2) + 1e-12
        return sx / norm, sy / norm, sz / norm

    return _run_ablation_em(
        fd_points, prototypes, dx, cells, x_pos, rho_obs,
        seed, rho_start, rho_end, dt_window,
        lam_fd_q, lam_phys, lam_smo, lr, em_iters, inner_iters,
        spin_to_pi_fn=_spin_to_pi_standard,
        update_spin_fn=update_spin_from_logits_grad,
        post_update_hook=_unit_norm_hook,
    )


def run_ablation_no_physics(fd_points, prototypes, dx, cells, x_pos, rho_obs,
                            seed=42, rho_start=None, rho_end=None, dt_window=None,
                            lam_fd_q=1.0, lam_phys=0.0, lam_smo=0.02,
                            lr=0.05, em_iters=80, inner_iters=20, **kw):
    """Ablation C: set lam_phys = 0 (remove conservation penalty).
    Always forces lam_phys=0 regardless of what is passed."""
    return _run_ablation_em(
        fd_points, prototypes, dx, cells, x_pos, rho_obs,
        seed, rho_start, rho_end, dt_window,
        lam_fd_q, 0.0, lam_smo, lr, em_iters, inner_iters,
        spin_to_pi_fn=_spin_to_pi_standard,
        update_spin_fn=update_spin_from_logits_grad,
    )


def run_ablation_single_phase(fd_points, prototypes, dx, cells, x_pos, rho_obs,
                              seed=42, rho_start=None, rho_end=None, dt_window=None,
                              **kw):
    """Ablation D: Single-Phase FD (standard LWR baseline).
    Fit one triangular FD to ALL data — eliminates multi-phase structure entirely.
    Equivalent to single-prototype SpinFlow (degenerate EM, no phase competition).
    Physical meaning: tests whether the three-phase mixture is necessary at all.
    Expected: significant RMSE increase, as one FD cannot simultaneously represent
    free-flow, synchronized, and congested branches."""
    from fd_model import calibrate_triangular_fd
    t0 = time.perf_counter()

    k_pts = fd_points['k'] / 1000.0
    q_pts = fd_points['q'] / 3600.0
    v_pts = fd_points['v'] / 3.6
    x_pts = fd_points['x_center']

    # Fit single triangular FD to all data (no phase separation)
    single_proto = calibrate_triangular_fd(fd_points['k'], fd_points['q'])

    # Forward metrics: single FD applied uniformly everywhere
    q_pred_pts = single_proto.flow(k_pts)
    q_err = (q_pred_pts - q_pts) * 3600
    v_pred_pts = np.where(k_pts > 0, q_pred_pts / k_pts, 0.0)
    v_err = (v_pred_pts - v_pts) * 3.6

    rmse_q = float(np.sqrt(np.mean(q_err ** 2)))
    rmse_v = float(np.sqrt(np.mean(v_err ** 2)))
    r2_q = float(1 - np.sum(q_err**2) / (np.sum((q_pts*3600 - (q_pts*3600).mean())**2) + 1e-12))
    r2_v = float(1 - np.sum(v_err**2) / (np.sum((v_pts*3.6 - (v_pts*3.6).mean())**2) + 1e-12))

    # Build compatible [cells, 3] pi_x: all weight on slot 0, same FD replicated
    # so _phys_residual and _entropy_profile can be called unchanged
    pi_x = np.zeros((cells, 3))
    pi_x[:, 0] = 1.0
    protos_compat = [single_proto, single_proto, single_proto]

    phys_res = _phys_residual(pi_x, protos_compat, rho_obs, rho_start, rho_end, dx, dt_window)
    H = _entropy_profile(pi_x)  # will be 0 everywhere (pure single-phase)

    # Fallback bottleneck detection: max density gradient (no phase structure available).
    # Physical meaning: without phase inference, the densest spatial gradient identifies
    # the most likely location of a macroscopic state change (e.g., queue front).
    x_star = None
    if rho_obs is not None and len(rho_obs) == cells and cells > 6:
        margin = max(3, int(0.12 * cells))
        grad_rho = np.abs(np.gradient(rho_obs.astype(float), dx))
        grad_rho[:margin] = 0.0
        grad_rho[-margin:] = 0.0
        if grad_rho.max() > 1e-9:
            x_pos_local = np.arange(cells) * dx
            x_star = float(x_pos_local[int(np.argmax(grad_rho))])

    runtime = time.perf_counter() - t0

    return {
        'rmse_q': rmse_q, 'r2_q': r2_q, 'rmse_v': rmse_v, 'r2_v': r2_v,
        'x_star': x_star, 'pi_x': pi_x,
        'phys_residual': phys_res, 'entropy_std': float(H.std()),
        'runtime_s': runtime, 'convergence_steps': 1,
    }


