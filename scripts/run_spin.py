
# -*- coding: utf-8 -*-
import numpy as np, yaml
from pathlib import Path
from src.mapping import MapParams
from src.spin import SpinParams
from src.simulate import run_spin_ctm, InitRhoConfig, InitSpinConfig
from src.viz import timespace, make_gif

cfg_base = yaml.safe_load(open('configs/base.yaml'))
cfg_ctm  = yaml.safe_load(open('configs/ctm.yaml'))
cfg_spin = yaml.safe_load(open('configs/spin.yaml'))
np.random.seed(cfg_base.get('seed', 2025))
steps, M, dt = cfg_base['steps'], cfg_base['cells'], cfg_base['dt']
rho_cfg  = InitRhoConfig(**cfg_ctm['rho_init'])
spin_cfg = InitSpinConfig(**cfg_spin['spin_init'])
mp = MapParams(**cfg_ctm['ctm'], **cfg_spin['mapping'])
sp = SpinParams(**cfg_spin['spin'])
res = run_spin_ctm(steps=steps, M=M, mp=mp, sp=sp, rho_cfg=rho_cfg, spin_cfg=spin_cfg, dt=dt)
Path('outputs').mkdir(exist_ok=True)
timespace(res['rho_hist'], title='Spin–CTM: time–space density', outfile='outputs/spin_timespace.png')
make_gif(res['rho_hist'], title='Spin–CTM evolution', outfile='outputs/spin_evolution.gif', stride=3, fps=12)
print('Saved to outputs/*.')
