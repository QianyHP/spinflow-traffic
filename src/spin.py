
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class SpinParams:
    Jx: float = 0.8
    Jy: float = 0.8
    Jz: float = 0.5
    eta: float = 0.14
    temp: float = 0.06
    r_bar: float = 0.6
    radius_reg: float = 0.02
    rho_star: float = 0.32

def project_ball(sx, sy, sz, r):
    norm = np.sqrt(sx*sx + sy*sy + sz*sz) + 1e-12
    return sx * (r / norm), sy * (r / norm), sz * (r / norm)

def spin_step(sx, sy, sz, rho, pars: SpinParams):
    M = rho.size
    idx = np.arange(M); L = np.roll(idx, 1); R = np.roll(idx, -1)
    hz = (pars.rho_star - rho) * 2.0
    hx = (0.25 - (rho - pars.rho_star)**2) * 2.2
    hy = (rho - pars.rho_star) * 2.5
    grad_sx = -pars.Jx * (sx[L] + sx[R]) - hx
    grad_sy = -pars.Jy * (sy[L] + sy[R]) - hy
    grad_sz = -pars.Jz * (sz[L] + sz[R]) - hz
    r_curr = np.sqrt(sx*sx + sy*sy + sz*sz) + 1e-8
    grad_sx += pars.radius_reg * ((r_curr - pars.r_bar) * (sx / r_curr))
    grad_sy += pars.radius_reg * ((r_curr - pars.r_bar) * (sy / r_curr))
    grad_sz += pars.radius_reg * ((r_curr - pars.r_bar) * (sz / r_curr))
    noise = lambda: np.sqrt(2*pars.temp*pars.eta) * np.random.randn(M)
    sx = sx - pars.eta * grad_sx + noise()
    sy = sy - pars.eta * grad_sy + noise()
    sz = sz - pars.eta * grad_sz + noise()
    r = np.minimum(np.sqrt(sx*sx + sy*sy + sz*sz), 1.0)
    sx, sy, sz = project_ball(sx, sy, sz, r)
    return sx, sy, sz, r
