"""
Phase utilities: map latent spin components s(x) to mixture weights pi(x) and
effective macroscopic FD parameters.

The spin-to-mixture map is fixed (no learned parameters): an antisymmetric,
competition-aware construction ties Free vs Congested opposition to the
magnitude |s_z - s_y|, so the synchronized (critical) phase can dominate when
those two components are balanced—consistent with Kerner three-phase picture.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from .fd_model import TriangularFD, PHASES, PHASE_INDEX, PHASE_DISPLAY, PHASE_COLORS


def spin_to_mixture_weights_softmax(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray,
                                    n_prototypes: int = 3) -> np.ndarray:
    """
    Map spin field (s_x, s_y, s_z) to phase mixture weights pi via softmax.

    Logits (order: Free, Critical, Congested) follow the antisymmetric form used
    in the paper: free = s_z - s_y, congested = s_y - s_z, and critical =
    s_x - |s_z - s_y| so that the middle phase is promoted when |s_z - s_y| is
    small (balanced competition between free and congested tendencies).

    Args:
        sx, sy, sz: Spin components along the spatial grid, shape [M].
        n_prototypes: Must match len(PHASES) (three-phase SpinFlow).

    Returns:
        pi: Normalized mixture weights, shape [M, n_prototypes].
    """
    M = len(sx)

    if n_prototypes != len(PHASES):
        raise ValueError("Antisymmetric mapping supports only 3 prototypes")

    # |s_z - s_y| encodes how strongly free and congested logits disagree.
    competition_strength = np.abs(sz - sy)

    logits = np.zeros((M, len(PHASES)))
    logits[:, PHASE_INDEX["free"]] = sz - sy
    logits[:, PHASE_INDEX["critical"]] = sx - competition_strength
    logits[:, PHASE_INDEX["congested"]] = sy - sz

    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    # Boltzmann / softmax: interpret logits as negative energies at inverse temperature = 1.
    pi = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return pi


def mixture_fd_params(pi: np.ndarray,
                     prototypes: List[TriangularFD]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convex combine triangular-FD parameters with weights pi(x).

    For each cell i: vf_i = sum_g pi_{ig} vf^g, and similarly for Q0, w, rho_jam.
    Useful for diagnostics and CTM-style updates that need a single effective FD.

    Returns:
        vf, Qcap, w, rho_jam: Effective fields along x, each shape [M].
    """
    M = pi.shape[0]
    n_prototypes = len(prototypes)

    vf = np.zeros(M)
    Qcap = np.zeros(M)
    w = np.zeros(M)
    rho_jam = np.zeros(M)

    for g, fd_proto in enumerate(prototypes):
        vf += pi[:, g] * fd_proto.vf
        Qcap += pi[:, g] * fd_proto.Q0
        w += pi[:, g] * fd_proto.w
        rho_jam += pi[:, g] * fd_proto.rho_jam

    return vf, Qcap, w, rho_jam


def visualize_phase_map(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray,
                       x_pos: np.ndarray,
                       prototypes: List[TriangularFD],
                       save_path: str = None):
    """Multi-panel figure: pi_g(x), dominant phase, effective vf and Qcap."""
    pi = spin_to_mixture_weights_softmax(sx, sy, sz, len(prototypes))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    for i, phase in enumerate(PHASES):
        ax.plot(
            x_pos,
            pi[:, i],
            label=f"π_{PHASE_DISPLAY[phase].lower()}",
            linewidth=2,
            alpha=0.8,
            color=PHASE_COLORS[phase],
        )
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Phase Weight π_g(x)')
    ax.set_title('Phase Mixture Weights', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    dominant_phase = np.argmax(pi, axis=1)
    colors_map = np.array([PHASE_COLORS[p] for p in PHASES])
    for phase_id in range(len(PHASES)):
        mask = (dominant_phase == phase_id)
        if mask.sum() > 0:
            ax.scatter(x_pos[mask], np.ones(mask.sum()) * phase_id,
                      c=colors_map[phase_id], s=100, alpha=0.7,
                      label=PHASE_DISPLAY[PHASES[phase_id]])
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Dominant Phase')
    ax.set_title('Dominant Phase Map', fontweight='bold')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([PHASE_DISPLAY[p] for p in PHASES])
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    vf, Qcap, w, rho_jam = mixture_fd_params(pi, prototypes)
    ax.plot(x_pos, vf*3.6, linewidth=2, color='blue', label='vf(x) from mixture')
    for i, proto in enumerate(prototypes):
        ax.axhline(proto.vf*3.6, linestyle='--', linewidth=1.5, alpha=0.6,
                   label=f'Prototype {i}: {proto.vf*3.6:.1f} km/h')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Free-flow speed (km/h)')
    ax.set_title('Effective vf(x) from Mixture', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_pos, Qcap*3600, linewidth=2, color='green', label='Qcap(x) from mixture')
    for i, proto in enumerate(prototypes):
        ax.axhline(proto.Q0*3600, linestyle='--', linewidth=1.5, alpha=0.6,
                   label=f'Prototype {i}: {proto.Q0*3600:.0f} veh/h')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Capacity (veh/h)')
    ax.set_title('Effective Qcap(x) from Mixture', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()
