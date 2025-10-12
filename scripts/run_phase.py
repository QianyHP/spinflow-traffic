
# -*- coding: utf-8 -*-
import numpy as np, yaml
from src.mapping import MapParams
from src.spin import SpinParams
from src.simulate import InitRhoConfig, InitSpinConfig, run_spin_ctm
from src.viz import phase_diagram

cfg_base = yaml.safe_load(open('configs/base.yaml'))
cfg_ctm  = yaml.safe_load(open('configs/ctm.yaml'))
cfg_spin = yaml.safe_load(open('configs/spin.yaml'))
cfg_ps   = yaml.safe_load(open('configs/phase_scan.yaml'))
steps, M, dt = cfg_base['steps'], cfg_base['cells'], cfg_base['dt']
mp = MapParams(**cfg_ctm['ctm'], **cfg_spin['mapping'])
sp = SpinParams(**cfg_spin['spin'])
rho_cfg = InitRhoConfig(**cfg_ctm['rho_init'])
G = cfg_ps['grid']
sx_lo, sx_hi = cfg_ps['sx_range']; sy_lo, sy_hi = cfg_ps['sy_range']
sx_vals = np.linspace(sx_lo, sx_hi, G); sy_vals = np.linspace(sy_lo, sy_hi, G)
std_grid = np.zeros((G, G)); reg_grid = np.zeros((G, G), dtype=int)
for ix, sxb in enumerate(sx_vals):
    for iy, syb in enumerate(sy_vals):
        spin_cfg = InitSpinConfig(r0=cfg_spin['spin_init']['r0'], sx_bias=sxb, sy_bias=syb,
                                  sz_bias=cfg_spin['spin_init']['sz_bias'], noise_std=cfg_spin['spin_init']['noise_std'])
        res = run_spin_ctm(steps=steps, M=M, mp=mp, sp=sp, rho_cfg=rho_cfg, spin_cfg=spin_cfg, dt=dt)
        std = float(np.std(res['flow_mean'][int(0.25*steps):]))
        std_grid[iy, ix] = std
thr = np.quantile(std_grid, cfg_ps['threshold_quantile'])
reg_grid[std_grid > thr] = 1
phase_diagram(reg_grid, sx_vals, sy_vals, title='Phase diagram: spin-bias plane (0=FF,1=Osc)')
