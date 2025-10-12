
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class CTMParams:
    v_min: float = 0.4
    v_top: float = 1.0
    w_min: float = 0.4
    w_max: float = 0.9
    Q0: float = 0.30

def demand_supply(rho, vf, Qcap, w, rho_jam):
    demand = np.minimum(vf * rho, Qcap)
    supply = np.minimum(Qcap, w * (rho_jam - rho))
    return demand, supply

def step_ring(rho, vf, Qcap, w, rho_jam, dt=0.4, dx=1.0):
    idx = np.arange(rho.size); right = np.roll(idx, -1); left = np.roll(idx, 1)
    demand, supply = demand_supply(rho, vf, Qcap, w, rho_jam)
    y = np.minimum(demand, supply[right])
    rho_new = rho + (dt/dx) * (y[left] - y)
    rho_new = np.clip(rho_new, 0.0, rho_jam)
    return rho_new, y
