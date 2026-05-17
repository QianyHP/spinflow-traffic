"""
Triangular fundamental diagrams (FD) and SpinFlow prototype calibration.

``TriangularFD`` implements the piecewise-linear q(ρ) used for each phase branch.
``calibrate_triangular_fd`` fits one triangle to (k,q) scatter; ``identify_phase_attractors``
runs K-means in normalized (ρ, v) space; ``calibrate_three_prototypes`` builds the
three phase-specific triangles used by the EM loop.
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Dict, List, Tuple, Sequence

# =============================================================================
# Phase definitions (single source of truth)
# Keep these here to avoid circular imports (phase_utils imports TriangularFD).
# =============================================================================

# Canonical phase order used across the pipeline
PHASES: Tuple[str, str, str] = ("free", "critical", "congested")

# Phase order produced by speed-sorted clustering (low -> high speed)
PHASES_BY_SPEED: Tuple[str, str, str] = ("congested", "critical", "free")

PHASE_INDEX = {phase: i for i, phase in enumerate(PHASES)}
PHASE_DISPLAY = {
    "free": "F",
    "critical": "S",
    "congested": "J",
}
PHASE_COLORS = {
    "free": "blue",
    "critical": "green",
    "congested": "red",
}


def reorder_from_speed_order(values: Sequence) -> Sequence:
    """
    Reorder a sequence from PHASES_BY_SPEED to PHASES (canonical).
    Expects values in order: (congested, critical, free).
    """
    indices = [PHASES_BY_SPEED.index(phase) for phase in PHASES]
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[i] for i in indices]


class TriangularFD:
    """Piecewise-linear (triangular) fundamental diagram q(ρ)."""
    
    def __init__(self, vf: float, w: float, rho_jam: float, Q0: float):
        """
        Args:
            vf: Free-flow speed (m/s).
            w: Shock-wave speed (m/s), negative.
            rho_jam: Jam density (veh/m).
            Q0: Capacity (veh/s).
        """
        self.vf = vf
        self.w = w
        self.rho_jam = rho_jam
        self.Q0 = Q0
        
        self.rho_c = (abs(w) / (vf + abs(w))) * rho_jam
    
    def flow(self, rho: np.ndarray) -> np.ndarray:
        """q = min(vf*ρ, Q0, |w|(ρ_jam - ρ)) with non-negativity."""
        q_free = self.vf * rho
        q_cong = abs(self.w) * (self.rho_jam - rho)
        q = np.minimum(np.minimum(q_free, self.Q0), q_cong)
        return np.maximum(q, 0.0)
    
    def to_dict(self) -> Dict:
        return {
            'vf': self.vf,
            'w': self.w,
            'rho_jam': self.rho_jam,
            'Q0': self.Q0,
            'rho_c': self.rho_c
        }


def calibrate_triangular_fd(k_data: np.ndarray, 
                           q_data: np.ndarray,
                           weights: np.ndarray = None,
                           init_params: Dict = None) -> TriangularFD:
    """
    Fit one triangular FD to scatter (k in veh/km, q in veh/h).

    Returns a ``TriangularFD`` in SI units (veh/m, veh/s).
    """
    if weights is None:
        weights = np.ones(len(k_data))
    
    rho_data = k_data / 1000.0  # veh/m
    q_data_si = q_data / 3600.0  # veh/s
    
    if init_params is None:
        v_data = q_data / k_data  # km/h
        v_median = np.median(v_data[~np.isnan(v_data) & (v_data > 0)])
        
        vf_init = max(v_median / 3.6, 20.0 / 3.6)
        w_init = -15.0 / 3.6
        rho_jam_init = min(np.percentile(rho_data, 95), 0.25)
        Q0_init = min(np.percentile(q_data_si, 95), 0.60)
    else:
        vf_init = init_params.get('vf', 25.0/3.6)
        w_init = init_params.get('w', -15.0/3.6)
        rho_jam_init = init_params.get('rho_jam', 0.15)
        Q0_init = init_params.get('Q0', 0.35)
    
    x0 = np.array([vf_init, abs(w_init), rho_jam_init, Q0_init])
    bounds = Bounds(
        lb=[5.0/3.6, 5.0/3.6, 0.05, 0.1],
        ub=[120.0/3.6, 50.0/3.6, 0.35, 0.70]
    )
    
    def loss(x):
        vf, w_abs, rho_jam, Q0 = x
        fd = TriangularFD(vf, -w_abs, rho_jam, Q0)
        q_pred = fd.flow(rho_data)
        residuals = q_pred - q_data_si
        return np.sum(weights * residuals**2)
    
    result = minimize(loss, x0, method='L-BFGS-B', bounds=bounds)
    
    vf_opt, w_abs_opt, rho_jam_opt, Q0_opt = result.x
    
    return TriangularFD(vf_opt, -w_abs_opt, rho_jam_opt, Q0_opt)



from scipy.cluster.vq import kmeans2

def identify_phase_attractors(k_data: np.ndarray, 
                            q_data: np.ndarray, 
                            v_data: np.ndarray, 
                            n_prototypes: int = 3) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    K-means attractors in normalized (rho, v) for SpinFlow initialization.

    Clusters are sorted by ascending speed (J, S, F order); per-cluster variance
    captures how "broad" each attractor basin is (used as inverse temperature).

    Returns:
        sorted_centroids: shape (n_prototypes, 2) in [0,1]^2.
        sigmas: mean squared distance to centroid per cluster.
        scalers: (rho_min, rho_max, v_min, v_max) in SI.
        fixed_labels: point labels in speed-sorted cluster indexing.
    """
    rho_data = k_data / 1000.0 # veh/m
    v_data_ms = v_data / 3.6   # m/s
    
    rho_min, rho_max = rho_data.min(), rho_data.max()
    v_min, v_max = v_data_ms.min(), v_data_ms.max()
    
    if rho_max == rho_min: rho_max += 1.0
    if v_max == v_min: v_max += 1.0
    
    rho_norm = (rho_data - rho_min) / (rho_max - rho_min)
    v_norm = (v_data_ms - v_min) / (v_max - v_min)
    
    X = np.column_stack([rho_norm, v_norm])
    
    centroids, labels = kmeans2(X, k=n_prototypes, minit='points', seed=123)
    
    idx_sorted = np.argsort(centroids[:, 1])
    sorted_centroids = centroids[idx_sorted]
    
    sigmas = np.zeros(n_prototypes)
    map_old_to_new = {old: new for new, old in enumerate(idx_sorted)}
    fixed_labels = np.array([map_old_to_new[l] for l in labels])
    
    for g in range(n_prototypes):
        points_g = X[fixed_labels == g]
        if len(points_g) > 1:
            d2 = np.sum((points_g - sorted_centroids[g])**2, axis=1)
            sigmas[g] = np.mean(d2)
        else:
            sigmas[g] = 0.05
            
    sigmas = np.maximum(sigmas, 1e-4)

    return sorted_centroids, sigmas, (rho_min, rho_max, v_min, v_max), fixed_labels


def calibrate_three_prototypes(fd_points: Dict,
                               n_prototypes: int = 3,
                               verbose: bool = True) -> Tuple[List[TriangularFD], Dict]:
    """
    Fit three triangular FDs after K-means phase attractors on (rho, v).

    Returns prototypes in canonical order (Free, Critical, Congested) plus
    ``attractor_info`` for entropy-aware spin initialization.
    """
    k = fd_points['k']
    q = fd_points['q']
    v = fd_points['v']
    
    centroids, sigmas, scalers, labels = identify_phase_attractors(k, q, v, n_prototypes)
    
    prototypes = []
    phase_names_speed = [PHASE_DISPLAY[p] for p in PHASES_BY_SPEED]

    if verbose:
        print("\n" + "="*80)
        print("FD Prototype Calibration (Phase Attractor Based)")
        print("="*80)
        print("Attractors Identified (Normalized rho, v) with Structural Entropy:")
        for i, c in enumerate(centroids):
            sigma = sigmas[i]
            print(f"  {phase_names_speed[i]}: center=({c[0]:.2f}, {c[1]:.2f}), sigma^2={sigma:.4f}")

    for g in range(n_prototypes):
        idx_group = (labels == g)
        
        if np.sum(idx_group) < 5:
            if verbose:
                print(f"  Warning: Attractor {phase_names_speed[g]} has few points, using heuristic fallback.")
            k_group = k
            q_group = q
            v_group = v
        else:
            k_group = k[idx_group]
            q_group = q[idx_group]
            v_group = v[idx_group]
        
        v_mean_g = np.mean(v_group) / 3.6
        rho_mean_g = np.mean(k_group) / 1000.0
        q_mean_g = np.mean(q_group) / 3600.0
        
        phase_speed = PHASES_BY_SPEED[g]
        if phase_speed == "congested":
            init = {
                'vf': 15.0/3.6,
                'w': max(v_mean_g, -25/3.6),
                'rho_jam': max(rho_mean_g * 1.2, 0.15),
                'Q0': q_mean_g
            }
        elif phase_speed == "critical":
            init = {'vf': 50.0/3.6, 'w': -15.0/3.6, 'rho_jam': 0.20, 'Q0': max(q_mean_g, 0.5)}
        else:  # free
            init = {'vf': max(v_mean_g, 80/3.6), 'w': -10.0/3.6, 'rho_jam': 0.15, 'Q0': max(q_mean_g, 0.6)}
        
        fd = calibrate_triangular_fd(k_group, q_group, init_params=init)
        prototypes.append(fd)
        
        if verbose:
            print(f"\n{phase_names_speed[g]} Prototype:")
            print(f"  Points: {np.sum(idx_group)} | v_avg: {np.mean(v_group):.1f} km/h")
            print(f"  vf={fd.vf*3.6:.1f}, w={fd.w*3.6:.1f}, Q0={fd.Q0*3600:.0f}, rho_jam={fd.rho_jam*1000:.0f}")
            
    if verbose:
        print("="*80)
    
    # Reorder to canonical phase order (Free, Critical, Congested)
    prototypes = reorder_from_speed_order(prototypes)
    centroids = reorder_from_speed_order(centroids)
    sigmas = reorder_from_speed_order(sigmas)

    if verbose:
        print("\nCanonical Order (Free, Critical, Congested):")
        for phase, proto in zip(PHASES, prototypes):
            name = PHASE_DISPLAY[phase]
            print(f"  {name:9s}: vf={proto.vf*3.6:.1f}, w={proto.w*3.6:.1f}, "
                  f"Q0={proto.Q0*3600:.0f}, rho_jam={proto.rho_jam*1000:.0f}")

    attractor_info = {
        'centroids': centroids,
        'scalers': scalers,
        'sigmas': sigmas,
        'phase_order': PHASES,
    }
    
    return prototypes, attractor_info




