"""
SpinFlow CLI pipeline: trajectories → quasi-stationary FD samples → Edie density → EM inversion.

Steps: load CSV, direction/lane filters, parallelogram sampling, FD cloud extraction,
spatial Edie averaging for rho(x) and boundary densities, then ``em_inverse_fd``.
Writes ``*_inverse_init.npz`` (with meta), parallelogram and phase-map PNGs.

Examples:
    python -m spinflow --dataset YTDJ
    spinflow --dataset RML --direction eb
    SPINFLOW_CONFIG=/path/custom.yaml spinflow --dataset HighD
"""

import argparse
import numpy as np
import yaml
import os
from pathlib import Path

from .repo_paths import repo_root, pipeline_config_path

from .sampler import sample_parallelograms, extract_fd_points, visualize_parallelograms
from .phase_utils import visualize_phase_map, spin_to_mixture_weights_softmax
from .solver import (
    em_inverse_fd,
    DEFAULT_SIGMA_SPATIAL_FACTOR,
)

from .preprocessing import load_trajectory_csv, select_direction, filter_lanes, parse_lanes_arg
from .observation import compute_edie_sle, DEFAULT_LANE_COUNT_THRESHOLD


def main():
    ap = argparse.ArgumentParser("YTDJ FD-based Inversion")
    
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pipeline YAML (defaults + per-dataset keys). "
             "Default: $SPINFLOW_CONFIG if set, else configs/pipeline.yaml, "
             "else legacy data/config.yaml if present.",
    )
    # Dataset / I/O
    ap.add_argument("--dataset", type=str, default="YTDJ", help="Dataset name (YTDJ, RML, XAM-N6)")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV file. If None, derived from dataset.")
    ap.add_argument("--direction", type=str, default=None, choices=["eb","wb"])
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--road-length", type=float, default=None)
    ap.add_argument("--x-offset", type=float, default=None, help="(m) ignore initial segment [0, x_offset] and shift s so domain starts at 0")
    ap.add_argument("--lanes", type=str, default=None)
    
    # Parallelogram sampling
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--T-sample", type=float, default=None, help="sampling time window (default: to data end)")
    ap.add_argument("--wave-speed", type=float, default=None, help="wave speed (km/h, negative, optimal: -8.5)")
    ap.add_argument("--Lw", type=float, default=None, help="parallelogram long edge (m)")
    ap.add_argument("--H", type=float, default=None, help="parallelogram height (m)")
    ap.add_argument("--min-points", type=int, default=None, help="minimum points per parallelogram")
    ap.add_argument("--v-max", type=float, default=None, help="Expected free flow speed (km/h)")
    
    # Inversion (EM)
    ap.add_argument("--cells", type=int, default=None)
    ap.add_argument("--n-prototypes", type=int, default=None)
    ap.add_argument("--em-iters", type=int, default=None, help="EM iterations (will stop early if converged)")
    ap.add_argument("--inner-iters", type=int, default=None, help="E-step inner iterations")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-prefix", type=str, default=None)
    
    # Loss weights and optimizer
    ap.add_argument("--lam-fd-q", type=float, default=None)
    ap.add_argument("--lam-phys", type=float, default=None)
    ap.add_argument("--lam-smo", type=float, default=None, help="Heisenberg exchange strength")
    ap.add_argument("--convergence-tol", type=float, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)

    
    args = ap.parse_args()

    _REPO = repo_root()

    def _as_repo_path(p):
        """Interpret non-absolute paths as relative to repository root."""
        p = Path(p)
        return str(p.resolve()) if p.is_absolute() else str((_REPO / p).resolve())

    # Configuration: unset CLI fields are filled from YAML so runs are reproducible from a single file.
    env_cfg = os.environ.get("SPINFLOW_CONFIG")
    if args.config is not None:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (_REPO / config_path).resolve()
    elif env_cfg:
        config_path = Path(env_cfg).expanduser()
        if not config_path.is_absolute():
            config_path = (_REPO / config_path).resolve()
    else:
        cand = pipeline_config_path()
        legacy = _REPO / "data" / "config.yaml"
        if cand.is_file():
            config_path = cand
        elif legacy.is_file():
            config_path = legacy
        else:
            raise FileNotFoundError(
                f"No pipeline config found. Add {cand} or set SPINFLOW_CONFIG / --config."
            )

    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    print(f"\nLoading configuration from {config_path}...")
    with open(str(config_path), 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    merged_config = full_config.get('defaults', {}).copy()

    if args.dataset in full_config:
        ds_config = full_config[args.dataset]
        print(f"  Applying settings for [{args.dataset}]:")
        merged_config.update(ds_config)

    print("  Merging YAML into CLI defaults (explicit CLI flags win)...")
    for key, value in merged_config.items():
        if hasattr(args, key):
            if getattr(args, key) is None:
                setattr(args, key, value)
            else:
                print(f"    ! CLI Override: {key}={getattr(args, key)} (YAML: {value})")

    if args.csv is None:
        raise ValueError(
            "CSV path not set. Add ``csv`` under the dataset block in the pipeline YAML "
            "or pass --csv path/to/file.csv."
        )

    args.csv = _as_repo_path(args.csv)

    if args.save_prefix is None:
        args.save_prefix = str(_REPO / "results" / args.dataset / args.dataset)
    else:
        args.save_prefix = _as_repo_path(args.save_prefix)

    # Ensure output directory exists for this dataset.
    # Rationale: make the pipeline reproducible on a fresh workspace.
    os.makedirs(os.path.dirname(args.save_prefix), exist_ok=True)

    np.random.seed(args.seed)
    
    print("="*80)
    print(f"{args.dataset} FD-Based Inversion Pipeline")
    print("="*80)
    
    # 1) Load trajectories
    print("\n[Step 1/5] Loading and preprocessing trajectory data...")
    df = load_trajectory_csv(args.csv, fps=args.fps)
    # select_direction needs the raw road length for coordinate flipping (wb)
    road_length_raw = float(args.road_length)
    df = select_direction(df, args.direction, road_length_raw)
    lanes = parse_lanes_arg(args.lanes)
    df = filter_lanes(df, lanes)

    # Optional spatial offset: drop [0, x_offset] and shift coordinates to start from 0.
    # Rationale: boundary ROI / detector edges can introduce systematic artifacts (e.g., blue borders).
    x_offset = float(getattr(args, "x_offset", 0.0) or 0.0)
    if x_offset < 0:
        raise ValueError(f"x_offset must be >= 0, got {x_offset}")
    if x_offset > 0:
        if x_offset >= road_length_raw:
            raise ValueError(f"x_offset={x_offset} must be smaller than road_length={road_length_raw}")
        df = df[(df["s"] >= x_offset) & (df["s"] <= road_length_raw)].copy()
        df["s"] = df["s"] - x_offset

    # After shifting, treat effective road length as the model domain length.
    args.road_length = float(road_length_raw - x_offset)
    
    if 'lane' not in df.columns:
        print("  Warning: No lane info, assuming single lane")
        df['lane'] = 1
    
    t_max = args.t0 + args.T_sample
    df_window = df[(df['t'] >= args.t0) & (df['t'] < t_max)].copy()
    
    print(f"  Total samples: {len(df)}")
    print(f"  Window samples: {len(df_window)} (t={args.t0}-{t_max}s)")
    print(f"  Direction: {args.direction}")
    
    # 2) Quasi-stationary parallelograms
    print("\n[Step 2/5] Quasi-stationary sampling with parallelograms...")
    sampling_speeds = list(range(2, int(args.v_max) + 2, 2))
    
    parallelograms = sample_parallelograms(
        df_window,
        wave_speed=args.wave_speed,
        given_speeds=sampling_speeds,
        Lw=args.Lw,
        H=args.H,
        t_min=args.t0,
        t_max=t_max,
        x_min=0.0,
        x_max=args.road_length,
        min_points=args.min_points,
        seed=args.seed
    )
    
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
    
    visualize_parallelograms(df_window, parallelograms, 
                            save_path=f"{args.save_prefix}_parallelograms.png",
                            xlim=(args.t0, t_max),
                            ylim=(0, args.road_length))
    
    # 3) Edie-consistent rho(x) and window boundaries for lam_phys
    print("\n[Step 3/5] Computing observed density field (Edie method)...")
    dx = args.road_length / args.cells
    x_pos = np.arange(args.cells) * dx
    dt_grid = 0.25
    
    t_obs_start = args.t0
    t_obs_end = args.t0 + args.T_sample
    df_obs = df_window[(df_window['t'] >= t_obs_start) & (df_window['t'] < t_obs_end)].copy()
    
    # Ensure vx exists (load_trajectory_csv may already compute it)
    if 'vx' not in df_obs.columns:
        raise ValueError("Expected 'vx' in trajectory data. Please include it or let loader estimate it.")

    edie = compute_edie_sle(
        df_obs,
        dx=dx,
        dt=dt_grid,
        t_start=t_obs_start,
        t_end=t_obs_end,
        road_length=args.road_length,
        fps=args.fps,
        lane_count_threshold=DEFAULT_LANE_COUNT_THRESHOLD,
    )
    rho_sle = edie.rho_sle
    q_sle = edie.q_sle
    
    rho_mean_raw = np.mean(rho_sle, axis=0)
    q_mean_raw = np.mean(q_sle, axis=0)
    v_mean_raw = np.zeros_like(q_mean_raw)
    mask = rho_mean_raw > 1e-6
    v_mean_raw[mask] = q_mean_raw[mask] / rho_mean_raw[mask]
    
    rho_start_raw = np.mean(rho_sle[:4], axis=0)
    rho_end_raw = np.mean(rho_sle[-4:], axis=0)
    rho_jam_est = np.quantile(rho_sle[rho_sle > 0], 0.99) if np.any(rho_sle > 0) else 1.0
    rho_mean_norm = np.clip(rho_mean_raw / max(rho_jam_est, 1e-6), 0.0, 1.0)
    rho_delta_norm = (rho_end_raw - rho_start_raw) / max(rho_jam_est, 1e-6)
    
    print(f"  Spatial grid: {args.cells} cells, dx={dx:.2f}m")
    print(f"  Observed density (60s window, Edie method):")
    print(f"    mean={rho_mean_norm.mean():.4f}, Δρ={rho_delta_norm.mean():+.4f}")
    print(f"  Jam density estimate: {rho_jam_est:.4f} veh/m")
    
    # 4) EM inversion (rho fixed from Edie)
    print("\n[Step 4/5] Inverting spin field s(x) via FD matching...")
    
    sx0, sy0, sz0, prototypes_final, rho0_final, history = em_inverse_fd(
        fd_points,
        dx,
        M=args.cells,
        rho_obs=rho_mean_raw,
        v_obs=v_mean_raw,
        rho_start=rho_start_raw,
        rho_end=rho_end_raw,
        dt_window=args.T_sample,
        n_prototypes=args.n_prototypes,
        em_iters=args.em_iters,
        inner_iters=args.inner_iters,
        lam_fd_q=args.lam_fd_q,
        lam_phys=args.lam_phys,
        lam_smo=args.lam_smo,
        convergence_tol=args.convergence_tol,
        learning_rate=args.learning_rate,
        seed=args.seed
    )

    
    pi_final = spin_to_mixture_weights_softmax(sx0, sy0, sz0, 3)
    
    # 5) Save NPZ + phase map
    print("\n[Step 5/5] Saving results...")
    
    np.savez(
        f"{args.save_prefix}_inverse_init.npz",
        rho0=rho0_final,
        rho_start=rho_start_raw,
        rho_end=rho_end_raw,
        rho_jam_est=rho_jam_est,
        sx0=sx0,
        sy0=sy0,
        sz0=sz0,
        pi_final=pi_final,
        prototypes=[p.to_dict() for p in prototypes_final],
        fd_points=fd_points,
        history=history,
        meta=dict(
            dx=dx,
            dt=dt_grid,
            cells=args.cells,
            t0=args.t0,
            T_sample=args.T_sample,  # Added T_sample
            fps=args.fps,            # Added fps
            road_length=args.road_length,
            road_length_raw=road_length_raw,
            x_offset=x_offset,
            road_begin=x_offset,
            road_end=road_length_raw,
            dataset=args.dataset,
            csv_path=args.csv,
            direction=args.direction,
            lanes=args.lanes,
            method='FD-EM-SpinInversion',
            theory='MacroscopicPhaseTransition',
            inversion_target='spin_only_rho_observed',
            density_source='Edie_observation',
            n_prototypes=3,
            lam_fd_q=args.lam_fd_q,
            sigma_spatial_factor=DEFAULT_SIGMA_SPATIAL_FACTOR,
            lane_count_threshold=DEFAULT_LANE_COUNT_THRESHOLD,
            lam_smo=args.lam_smo,
            normalization='adaptive',
            final_loss=history['loss'][-1] if history['loss'] else None,
            final_loss_q=history['loss_fd_q'][-1] if history['loss_fd_q'] else None,
            parallelogram_params=dict(Lw=args.Lw, H=args.H, wave_speed=args.wave_speed)
        )
    )
    
    print(f"  Saved: {args.save_prefix}_inverse_init.npz")
    
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

