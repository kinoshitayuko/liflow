# -*- coding: utf-8 -*-
import re
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


# -------- user parameters --------
input_path     = "output.xyz"
species_target = "He"
lag_frames     = 50       # how many frames ahead to compute displacement
bins           = 80

# time per frame [ps]
time_step_ps   = 0.01
dump_stride    = 1
dt_frame_ps    = time_step_ps * dump_stride
# ---------------------------------


def iter_xyz_frames(path):
    """XYZ frame generator."""
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            if not re.fullmatch(r"\d+", line):
                continue
            natoms  = int(line)
            comment = f.readline()
            if not comment:
                return
            comment = comment.strip()
            species = []
            pos = np.empty((natoms, 3), dtype=float)
            for i in range(natoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Bad atom line: {parts}")
                species.append(parts[0])
                pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
            yield {"species": species, "pos": pos, "comment": comment}


def compute_displacements(path, species_target, lag_frames):
    """Collect (dx,dy,dz) for all time windows t -> t+lag using unwrapped coords."""
    it  = iter_xyz_frames(path)
    fr0 = next(it)
    idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
    if not idx:
        raise ValueError(f"species '{species_target}' not found")
    buf       = deque(maxlen=lag_frames + 1)
    buf.append(fr0["pos"][idx])
    n_frames  = 1
    dxyz_list = []
    for fr in it:
        for k in idx:
            if fr["species"][k] != species_target:
                raise ValueError("Species order inconsistent across frames.")
        buf.append(fr["pos"][idx])
        n_frames += 1
        if len(buf) == lag_frames + 1:
            dr = buf[-1] - buf[0]
            dxyz_list.append(dr)
    dxyz = np.vstack(dxyz_list) if dxyz_list else np.empty((0, 3), dtype=float)
    return dxyz, n_frames, len(idx)


def add_stats_box(ax, data, n_frames, n_target, n_expected, lag_frames, species_target):
    mu  = float(np.mean(data))           if data.size      else float('nan')
    sd  = float(np.std(data, ddof=1))    if data.size >= 2 else float('nan')
    txt = (
        f"frames={n_frames}, lag={lag_frames}\n"
        f"{species_target} count={n_target}\n"
        f"n(samples)={data.size} (expected {n_expected})\n"
        f"mean={mu:.4f} A\nstd={sd:.4f} A"
    )
    at = AnchoredText(txt, prop=dict(size=9), frameon=True, loc='upper right')
    at.patch.set_alpha(0.85)
    ax.add_artist(at)


def plot_hist(data, xlabel, title, out_path, bins, hist_range, **stats_kw):
    color = stats_kw.pop('color', '#1f77b4')
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hist(data, bins=bins, range=hist_range, color=color, alpha=0.90, edgecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Counts')
    ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.3)
    add_stats_box(ax, data, **stats_kw)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    dxyz, n_frames, n_target = compute_displacements(input_path, species_target, lag_frames)

    lag_ps     = lag_frames * dt_frame_ps
    n_expected = n_target * max(n_frames - lag_frames, 0)
    base       = Path(input_path).with_suffix('')
    prefix     = f'{base}.{species_target}.lag{lag_frames}f'

    stats_kw = dict(n_frames=n_frames, n_target=n_target,
                    n_expected=n_expected, lag_frames=lag_frames,
                    species_target=species_target)

    # --- figure 1: |dr| magnitude ---
    disp_mag   = np.linalg.norm(dxyz, axis=1)
    mag_range  = (float(disp_mag.min()), float(disp_mag.max())) if disp_mag.size else None
    plot_hist(
        disp_mag,
        xlabel     = f'|dr| over {lag_frames} frames (= {lag_ps:g} ps)  (A)',
        title      = f'{species_target}: displacement magnitude histogram (unwrap)',
        out_path   = f'{prefix}.disp_hist.png',
        bins       = bins,
        hist_range = mag_range,
        color      = '#1f77b4',
        **stats_kw,
    )

    # --- figures 2-4: dx, dy, dz components (shared x range) ---
    comp_range = (float(dxyz.min()), float(dxyz.max())) if dxyz.size else None
    for comp, label, color, suffix in zip(
        range(3),
        ['dx', 'dy', 'dz'],
        ['#d62728', '#2ca02c', '#9467bd'],
        ['dx_hist', 'dy_hist', 'dz_hist'],
    ):
        plot_hist(
            dxyz[:, comp],
            xlabel     = f'Δ{label[1]} over {lag_frames} frames (= {lag_ps:g} ps)  (A)',
            title      = f'{species_target}: Δ{label[1]} histogram (unwrap)',
            out_path   = f'{prefix}.{suffix}.png',
            bins       = bins,
            hist_range = comp_range,
            color      = color,
            **stats_kw,
        )


if __name__ == "__main__":
    main()
