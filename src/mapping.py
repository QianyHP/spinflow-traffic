
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class MapParams:
    v_min: float = 0.4
    v_top: float = 1.0
    w_min: float = 0.4
    w_max: float = 0.9
    Q0: float = 0.30
    az_map: float = 2.4
    bx_map: float = 2.0
    by_map: float = 2.0
    dy_map: float = 2.0

def sigmoid(x): return 1/(1+np.exp(-x))

def spin_to_params(sx, sy, sz, r, mp: MapParams):
    zx = r * sx; zy = r * sy; zz = r * sz
    vf = mp.v_min + (mp.v_top - mp.v_min) * sigmoid(mp.az_map * zz)
    cap_mult = 0.5 + 0.5 * sigmoid(mp.bx_map * zx)
    Qcap = mp.Q0 * cap_mult
    wloc = mp.w_min + (mp.w_max - mp.w_min) * sigmoid(mp.by_map * zy)
    rho_jam = 0.6 + 0.4 * (1 - sigmoid(mp.dy_map * zy))
    return vf, Qcap, wloc, rho_jam
