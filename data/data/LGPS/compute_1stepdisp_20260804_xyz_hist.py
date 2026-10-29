
# -*- coding: utf-8 -*-
import re
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


# -------- user parameters --------
input_path = "../../../scripts/traj_output_LGPS/traj_650K_42_20260514_sample.xyz"
species_target = "Li"

lag_frames = 1      # ←「??step後」をここで指定（フレーム単位）
bins = 80

# 表示用（1フレームが何psか）
time_step_ps = 1.005
dump_stride = 1
dt_frame_ps = time_step_ps * dump_stride
# --------------------------------


def iter_xyz_frames(path):
    """Plain XYZ reader: natoms / comment / natoms lines (sp x y z ...)."""
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

            natoms = int(line)
            comment = f.readline()
            if not comment:
                return
            comment = comment.strip()

            species = []
            pos = np.empty((natoms, 3), dtype=float)
            for i in range(natoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError("Bad atom line: need 'sp x y z ...'")
                species.append(parts[0])
                pos[i, 0] = float(parts[1])
                pos[i, 1] = float(parts[2])
                pos[i, 2] = float(parts[3])

            yield {"species": species, "pos": pos, "comment": comment}


def compute_species_displacement_components_unwrap(path, species_target, lag_frames):
    """
    unwrap座標前提:
      1) 1フレーム目で species_target のインデックスを決める
      2) 全時間窓 t→t+lag で (Δx,Δy,Δz) を全原子分集める
    返り値: dxyz (n_samples,3), n_frames, n_target
    """
    it = iter_xyz_frames(path)
    fr0 = next(it)  # 1フレーム目

    idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
    if len(idx) == 0:
        raise ValueError(f"No atoms found for species_target={species_target}")

    buf = deque(maxlen=lag_frames + 1)
    buf.append(fr0["pos"][idx])
    n_frames = 1

    dxyz_list = []

    for fr in it:
        for k in idx:
            if fr["species"][k] != species_target:
                raise ValueError("Atom ordering/species inconsistent across frames.")

        buf.append(fr["pos"][idx])
        n_frames += 1

        if len(buf) == lag_frames + 1:
            dr = buf[-1] - buf[0]   # (n_target,3) = (Δx,Δy,Δz)
            dxyz_list.append(dr)

    if not dxyz_list:
        dxyz = np.empty((0, 3), dtype=float)
    else:
        dxyz = np.vstack(dxyz_list)  # (n_windows*n_target,3)

    return dxyz, n_frames, len(idx)


def add_stats_box(ax, data, n_frames, lag_frames, species_target, n_target, n_expected):
    mu = float(np.mean(data)) if data.size else float("nan")
    sd = float(np.std(data, ddof=1)) if data.size >= 2 else float("nan")
    at = AnchoredText(
        f"frames={n_frames}, lag={lag_frames}\n"
        f"{species_target} count={n_target}\n"
        f"n(samples)={data.size} (expected {n_expected})\n"
        f"mean={mu:.4f} Å\nstd={sd:.4f} Å",
        prop=dict(size=9), frameon=True, loc="upper right"
    )
    at.patch.set_alpha(0.85)
    ax.add_artist(at)


def save_component_hist(data, label, out_png, bins, hist_range,
                        species_target, lag_frames, lag_ps,
                        n_frames, n_target, n_expected):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hist(data, bins=bins, range=hist_range, color="#1f77b4", alpha=0.90, edgecolor="white")
    ax.set_xlabel(f"{label} over {lag_frames} frames (= {lag_ps:g} ps)  (Å)")
    ax.set_ylabel("Counts")
    ax.set_title(f"{species_target}: {label} histogram (unwrap)")
    ax.grid(True, ls="--", alpha=0.3)
    add_stats_box(ax, data, n_frames, lag_frames, species_target, n_target, n_expected)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)  # 図オブジェクトのsavefigでもOK
    plt.close(fig)


def main():
    dxyz, n_frames, n_target = compute_species_displacement_components_unwrap(
        input_path, species_target, lag_frames
    )

    lag_ps = lag_frames * dt_frame_ps
    n_expected = n_target * max(n_frames - lag_frames, 0)

    dx = dxyz[:, 0]
    dy = dxyz[:, 1]
    dz = dxyz[:, 2]

    # 3軸でビン範囲を揃えたい場合（見比べやすい）
    if dxyz.size:
        vmin = float(np.min(dxyz))
        vmax = float(np.max(dxyz))
        hist_range = (vmin, vmax)
    else:
        hist_range = None

    base = Path(input_path).with_suffix("")
    out_dx = Path(f"{base}.{species_target}.dx_hist_lag{lag_frames}f.png")
    out_dy = Path(f"{base}.{species_target}.dy_hist_lag{lag_frames}f.png")
    out_dz = Path(f"{base}.{species_target}.dz_hist_lag{lag_frames}f.png")

    save_component_hist(dx, "Δx", out_dx, bins, hist_range, species_target, lag_frames, lag_ps,
                        n_frames, n_target, n_expected)
    save_component_hist(dy, "Δy", out_dy, bins, hist_range, species_target, lag_frames, lag_ps,
                        n_frames, n_target, n_expected)
    save_component_hist(dz, "Δz", out_dz, bins, hist_range, species_target, lag_frames, lag_ps,
                        n_frames, n_target, n_expected)

    print(f"Saved: {out_dx}")
    print(f"Saved: {out_dy}")
    print(f"Saved: {out_dz}")


if __name__ == "__main__":
    main()
