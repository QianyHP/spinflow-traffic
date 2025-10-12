
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from .ctm import step_ring
from .spin import spin_step, project_ball, SpinParams
from .mapping import spin_to_params, MapParams

@dataclass
class InitRhoConfig:
    background: float = 0.18
    noise_std: float = 0.02
    jam_blocks: Dict[str, Any] = None

@dataclass
class InitSpinConfig:
    r0: float = 0.85
    sx_bias: float = 0.10
    sy_bias: float = 0.25
    sz_bias: float = 0.20
    noise_std: float = 0.06

def init_rho(M: int, cfg: InitRhoConfig):
    rho = cfg.background * np.ones(M)
    if cfg.noise_std > 0:
        rho += np.random.normal(0.0, cfg.noise_std, size=M)
    rho = np.clip(rho, 0.05, 0.45)
    if cfg.jam_blocks:
        for _ in range(cfg.jam_blocks.get('count', 1)):
            length = np.random.randint(M//12, M//8 + 1)
            start = np.random.randint(0, M); end = start + length
            hi = np.random.uniform(cfg.jam_blocks.get('hi_min', 0.62), cfg.jam_blocks.get('hi_max', 0.78))
            if end <= M: rho[start:end] = hi
            else: k=end-M; rho[start:] = hi; rho[:k] = hi
    return rho

def init_spin(M: int, cfg: InitSpinConfig):
    sx = cfg.sx_bias + cfg.noise_std*np.random.randn(M)
    sy = cfg.sy_bias + cfg.noise_std*np.random.randn(M)
    sz = cfg.sz_bias + cfg.noise_std*np.random.randn(M)
    r  = cfg.r0 * np.ones(M)
    return project_ball(sx, sy, sz, r)

def run_ctm_only(steps=150, M=60, mp: MapParams=None, rho_cfg: InitRhoConfig=None, dt=0.4):
    if mp is None: mp = MapParams()
    if rho_cfg is None:
        rho_cfg = InitRhoConfig(background=0.18, noise_std=0.02, jam_blocks={'count':1,'hi_min':0.62,'hi_max':0.78})
    rho = init_rho(M, rho_cfg)
    vf, Qcap, w, rho_jam = spin_to_params(np.zeros(M), np.zeros(M), np.zeros(M), 0.0*np.ones(M), mp)
    rho_hist = np.zeros((steps, M)); flow_mean = np.zeros(steps)
    for t in range(steps):
        rho, y = step_ring(rho, vf, Qcap, w, rho_jam, dt=dt)
        rho_hist[t] = rho; flow_mean[t] = np.mean(y)
    return {'rho_hist': rho_hist, 'flow_mean': flow_mean}

def run_spin_ctm(steps=150, M=60, mp: MapParams=None, sp: SpinParams=None,
                 rho_cfg: InitRhoConfig=None, spin_cfg: InitSpinConfig=None, dt=0.4):
    if mp is None: mp = MapParams()
    if sp is None: sp = SpinParams()
    if rho_cfg is None:
        rho_cfg = InitRhoConfig(background=0.18, noise_std=0.02, jam_blocks={'count':1,'hi_min':0.62,'hi_max':0.78})
    if spin_cfg is None:
        spin_cfg = InitSpinConfig()
    rho = init_rho(M, rho_cfg)
    sx, sy, sz = init_spin(M, spin_cfg)
    rho_hist = np.zeros((steps, M)); flow_mean = np.zeros(steps)
    for t in range(steps):
        vf, Qcap, w, rho_jam = spin_to_params(sx, sy, sz, np.minimum(np.sqrt(sx*sx+sy*sy+sz*sz),1.0), mp)
        rho, y = step_ring(rho, vf, Qcap, w, rho_jam, dt=dt)
        rho_hist[t] = rho; flow_mean[t] = np.mean(y)
        sx, sy, sz, _ = spin_step(sx, sy, sz, rho, sp)
    return {'rho_hist': rho_hist, 'flow_mean': flow_mean}
