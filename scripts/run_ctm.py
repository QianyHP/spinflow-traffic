
# -*- coding: utf-8 -*-
import numpy as np, yaml
from pathlib import Path
from src.mapping import MapParams
from src.simulate import run_ctm_only, InitRhoConfig
from src.viz import timespace, make_gif

cfg_base = yaml.safe_load(open('configs/base.yaml'))
cfg_ctm = yaml.safe_load(open('configs/ctm.yaml'))
np.random.seed(cfg_base.get('seed', 2025))
steps, M, dt = cfg_base['steps'], cfg_base['cells'], cfg_base['dt']
rho_cfg = InitRhoConfig(**cfg_ctm['rho_init']); mp = MapParams(**cfg_ctm['ctm'])
res = run_ctm_only(steps=steps, M=M, mp=mp, rho_cfg=rho_cfg, dt=dt)
Path('outputs').mkdir(exist_ok=True)
timespace(res['rho_hist'], title='CTM baseline: time–space density', outfile='outputs/ctm_timespace.png')
make_gif(res['rho_hist'], title='CTM baseline evolution', outfile='outputs/ctm_evolution.gif', stride=3, fps=12)
print('Saved to outputs/*.')
