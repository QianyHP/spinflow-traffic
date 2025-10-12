
# -*- coding: utf-8 -*-
import numpy as np, yaml
from pathlib import Path
from src.mapping import MapParams
from src.spin import SpinParams
from src.simulate import run_spin_ctm, InitRhoConfig, InitSpinConfig
from src.viz import make_gif

cfg_base = yaml.safe_load(open('configs/base.yaml'))
cfg_ctm  = yaml.safe_load(open('configs/ctm.yaml'))
cfg_spin = yaml.safe_load(open('configs/spin.yaml'))
cfg_lr   = yaml.safe_load(open('configs/long_run.yaml'))
np.random.seed(cfg_base.get('seed', 2025))
steps = cfg_lr['steps']; M = cfg_base['cells']; dt = cfg_base['dt']
rho_cfg  = InitRhoConfig(**cfg_ctm['rho_init'])
spin_cfg = InitSpinConfig(**cfg_spin['spin_init'])
mp = MapParams(**cfg_ctm['ctm'], **cfg_spin['mapping']); sp = SpinParams(**cfg_spin['spin'])
res = run_spin_ctm(steps=steps, M=M, mp=mp, sp=sp, rho_cfg=rho_cfg, spin_cfg=spin_cfg, dt=dt)
Path('outputs').mkdir(exist_ok=True)
gif_path = 'outputs/long_evolution.gif'
make_gif(res['rho_hist'], title='Long-run Spin–CTM evolution', outfile=gif_path, stride=cfg_lr['gif_stride'], fps=12)
jam_thr = cfg_lr['jam_threshold']; tail = slice(int(steps*2/3), None)
jam_fraction = float(np.mean(res['rho_hist'][tail] > jam_thr))
dissipated = jam_fraction < cfg_lr['dissipated_cut']
print(f'Late-time jam fraction: {jam_fraction:.3f} | Dissipated? {dissipated}')
print('Saved GIF:', gif_path)
