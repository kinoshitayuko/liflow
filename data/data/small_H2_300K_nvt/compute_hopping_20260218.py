# -*- coding: utf-8 -*-
import re
from collections import deque
from pathlib import Path
import numpy as np


# -------- user parameters (constants) --------
INPUT_PATH = "output.xyz"
SPECIES_TARGET = "He"

LAG_FRAMES = 50          # 「何フレーム後」をここで指定
HOP_THRESHOLD_A = 5.0    # ここが閾値（Å）

# 1フレームが何psか（必要なら rate(ps^-1) の計算に使える）
TIME_STEP_PS = 0.01
DUMP_STRIDE = 1
DT_FRAME_PS = TIME_STEP_PS * DUMP_STRIDE
# --------------------------------------------


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

            yield {"species": species, "pos": pos}


def compute_species_displacement_magnitudes_unwrap(path, species_target, lag_frames):
    it = iter_xyz_frames(path)
    fr0 = next(it)

    idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
    if len(idx) == 0:
        raise ValueError(f"No atoms found for species_target={species_target}")

    buf = deque(maxlen=lag_frames + 1)
    buf.append(fr0["pos"][idx])
    n_frames = 1
    disp = []

    for fr in it:
        for k in idx:
            if fr["species"][k] != species_target:
                raise ValueError("Atom ordering/species inconsistent across frames.")

        buf.append(fr["pos"][idx])
        n_frames += 1

        if len(buf) == lag_frames + 1:
            dr = buf[-1] - buf[0]
            mag = np.linalg.norm(dr, axis=1)
            disp.extend(mag.tolist())

    return np.array(disp, dtype=float), n_frames, len(idx)


def main():
    disp_A, n_frames, n_target = compute_species_displacement_magnitudes_unwrap(
        INPUT_PATH, SPECIES_TARGET, LAG_FRAMES
    )

    hop_mask = disp_A >= HOP_THRESHOLD_A
    hop_count = int(np.count_nonzero(hop_mask))  # [web:721]
    total = int(disp_A.size)
    hop_rate = (hop_count / total) if total > 0 else float("nan")

    lag_ps = LAG_FRAMES * DT_FRAME_PS
    n_samples_expected = n_target * max(n_frames - LAG_FRAMES, 0)

    base = Path(INPUT_PATH).with_suffix("")
    out_all = Path(f"{base}.{SPECIES_TARGET}.disp_lag{LAG_FRAMES}f.csv")
    out_hop = Path(f"{base}.{SPECIES_TARGET}.disp_lag{LAG_FRAMES}f_thr{HOP_THRESHOLD_A:g}A_only.csv")

    # 必要ならCSV保存（全サンプル / ホップのみ）
    # np.savetxt(out_all, disp_A, delimiter=",", header="disp_A", comments="")
    # np.savetxt(out_hop, disp_A[hop_mask], delimiter=",",
    #            header=f"disp_A_ge_{HOP_THRESHOLD_A:g}A", comments="")

    print("=== Hop frequency ===")
    print(f"input         : {INPUT_PATH}")
    print(f"species       : {SPECIES_TARGET}")
    print(f"lag_frames    : {LAG_FRAMES}  (={lag_ps:g} ps)")
    print(f"threshold_A   : {HOP_THRESHOLD_A:g} Å  (|Δr| >= threshold)")
    print(f"frames        : {n_frames}")
    print(f"n_target      : {n_target}")
    print(f"n_samples     : {total}  (expected {n_samples_expected})")
    print(f"hop_count     : {hop_count}")
    print(f"hop_rate      : {hop_rate:.8f}")
    print(f"Saved: {out_all}")
    print(f"Saved: {out_hop}")


if __name__ == "__main__":
    main()
