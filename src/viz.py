
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt, imageio
from pathlib import Path

def timespace(rho_hist, title='Time–space density', outfile=None):
    fig = plt.figure(figsize=(8, 3.6))
    plt.imshow(rho_hist, aspect='auto', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Density (ρ/ρ_jam)')
    plt.xlabel('Cell index'); plt.ylabel('Time step'); plt.title(title)
    plt.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=180)
    plt.show()

def make_gif(rho_hist, title='Evolution', outfile='anim.gif', stride=3, fps=12):
    frames = []
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    for t in range(1, rho_hist.shape[0], stride):
        fig = plt.figure(figsize=(7.2, 3.4))
        plt.imshow(rho_hist[:t, :], aspect='auto', origin='lower', vmin=0, vmax=1)
        plt.colorbar(label='Density (ρ/ρ_jam)')
        plt.xlabel('Cell index'); plt.ylabel('Time (frames)')
        plt.title(f'{title} (t={t})'); plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image); plt.close(fig)
    imageio.mimsave(outfile, frames, fps=fps)
    return outfile

def phase_diagram(regime_grid, sx_vals, sy_vals, title='Phase diagram', outfile=None):
    fig = plt.figure(figsize=(6.5,4.6))
    extent=[sx_vals[0], sx_vals[-1], sy_vals[0], sy_vals[-1]]
    plt.imshow(regime_grid, origin='lower', extent=extent, aspect='auto')
    cbar = plt.colorbar(); cbar.set_label('Regime index')
    plt.xlabel('Initial mean $s_x$'); plt.ylabel('Initial mean $s_y$')
    plt.title(title); plt.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=180)
    plt.show()
