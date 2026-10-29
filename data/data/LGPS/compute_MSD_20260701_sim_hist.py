# -*- coding: utf-8 -*-
import re
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


# -------- user parameters --------
input_path = "../../../scripts/traj_output_LGPS/traj_650K_42_20260514_2.xyz"
species_target = "Li"

lag_frames = 1      # ←「??step後」をここで指定（フレーム単位）
bins = 80

# 表示用（1フレームが何psか）
time_step_ps = 0.5
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


def compute_species_displacement_magnitudes_unwrap(path, species_target, lag_frames):
    """
    unwrap座標前提:
      1) 1フレーム目で species_target のインデックスを決める
      2) 全時間窓 t→t+lag で |r(t+lag)-r(t)| を全原子分集める
    返り値: disp(|Δr|) 1D array, n_frames, n_target
    """
    it = iter_xyz_frames(path)
    fr0 = next(it)  # 1フレーム目
    idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
    if len(idx) == 0:
        raise ValueError(f"No atoms found for species_target={species_target}")

    buf = deque(maxlen=lag_frames + 1)

    # frame0を入れる
    buf.append(fr0["pos"][idx])
    n_frames = 1

    disp = []

    # 残りフレームをストリーミング処理
    for fr in it:
        # 念のため、同じインデックスが同じ元素かチェック（順序が固定前提）
        for k in idx:
            if fr["species"][k] != species_target:
                raise ValueError("Atom ordering/species inconsistent across frames.")

        buf.append(fr["pos"][idx])
        n_frames += 1

        if len(buf) == lag_frames + 1:
            dr = buf[-1] - buf[0]                  # unwrap差分
            mag = np.linalg.norm(dr, axis=1)       # (n_target,)
            disp.extend(mag.tolist())

    return np.array(disp, dtype=float), n_frames, len(idx)


def main():
    disp_A, n_frames, n_target = compute_species_displacement_magnitudes_unwrap(
        input_path, species_target, lag_frames
    )

    lag_ps = lag_frames * dt_frame_ps
    n_samples_expected = n_target * max(n_frames - lag_frames, 0)

    base = Path(input_path).with_suffix("")
    out_png = Path(f"{base}.{species_target}.disp_hist_lag{lag_frames}f.png")
    #out_csv = Path(f"{base}.{species_target}.disp_lag{lag_frames}f.csv")

    # # CSV保存（1列：|Δr|）
    # np.savetxt(out_csv, disp_A, delimiter=",", header="disp_A", comments="")
    # print(f"Saved: {out_csv}")

    # ヒストグラム
    plt.figure(figsize=(6.8, 4.6))
    plt.hist(disp_A, bins=bins, color="#1f77b4", alpha=0.90, edgecolor="white")
    plt.xlabel(f"|Δr| over {lag_frames} frames (= {lag_ps:g} ps)  (Å)")
    plt.ylabel("Counts")
    plt.title(f"{species_target}: displacement magnitude histogram (unwrap)")

    mu = float(np.mean(disp_A)) if disp_A.size else float("nan")
    sd = float(np.std(disp_A, ddof=1)) if disp_A.size >= 2 else float("nan")
    at = AnchoredText(
        f"frames={n_frames}, lag={lag_frames}\n"
        f"{species_target} count={n_target}\n"
        f"n(samples)={disp_A.size} (expected {n_samples_expected})\n"
        f"mean={mu:.4f} Å\nstd={sd:.4f} Å",
        prop=dict(size=9), frameon=True, loc="upper right"
    )
    at.patch.set_alpha(0.85)
    plt.gca().add_artist(at)

    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()