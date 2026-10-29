# -*- coding: utf-8 -*-
import re
from pathlib import Path
import numpy as np

# -------- user parameters (constants) --------
INPUT_PATH = "output.xyz"

# XYZの元素ラベルが H なら "H"、Heなら "He" など
ATOM_LABEL = "He"

# 偶数個あるATOM_LABELを (0,1), (2,3), ... とペアにして「H2分子」とみなす
LAG_FRAMES = 50          # 「何フレーム後」をここで指定

# 「前回からのMSD増分」しきい値（Å^2）
DELTA_MSD_THRESHOLD_A2 = 1500.0

# 1フレームが何psか（あなたの出力に合わせて）
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


def make_pairs_from_first_frame(species, atom_label):
    idx = [i for i, sp in enumerate(species) if sp == atom_label]
    if len(idx) == 0:
        raise ValueError(f"No atoms found for ATOM_LABEL={atom_label}")
    if len(idx) % 2 != 0:
        raise ValueError(f"Number of {atom_label} atoms is odd: {len(idx)}")
    pairs = [(idx[i], idx[i + 1]) for i in range(0, len(idx), 2)]
    return pairs


def compute_com_positions(pos, pairs):
    # 等質量2原子のCOMなので単純平均
    com = np.empty((len(pairs), 3), dtype=float)
    for m, (a, b) in enumerate(pairs):
        com[m] = 0.5 * (pos[a] + pos[b])
    return com


def main():
    it = iter_xyz_frames(INPUT_PATH)
    fr0 = next(it)

    pairs = make_pairs_from_first_frame(fr0["species"], ATOM_LABEL)
    n_mol = len(pairs)

    com0 = compute_com_positions(fr0["pos"], pairs)  # (n_mol, 3)

    # t=0
    times_ps = [0.0]
    msd_A2 = [0.0]
    delta_msd_A2 = [0.0]  # 先頭は定義できないので0

    first_cross = None
    prev_msd = 0.0

    frame_i = 0
    for fr in it:
        frame_i += 1

        # lag_framesごとに評価
        if frame_i % LAG_FRAMES != 0:
            continue

        comt = compute_com_positions(fr["pos"], pairs)
        dr = comt - com0
        msd = float(np.mean(np.sum(dr * dr, axis=1)))  # 分子平均の<|dr|^2>

        dmsd = msd - prev_msd
        prev_msd = msd

        t_ps = frame_i * DT_FRAME_PS
        times_ps.append(t_ps)
        msd_A2.append(msd)
        delta_msd_A2.append(dmsd)

        if first_cross is None and dmsd >= DELTA_MSD_THRESHOLD_A2:
            first_cross = (t_ps, msd, dmsd, len(times_ps) - 1)

    print("=== H2 COM ΔMSD threshold crossing ===")
    print(f"input               : {INPUT_PATH}")
    print(f"atom_label          : {ATOM_LABEL}")
    print(f"n_molecules(pairs)  : {n_mol}")
    print(f"lag_frames          : {LAG_FRAMES}  (={LAG_FRAMES * DT_FRAME_PS:g} ps)")
    print(f"delta_msd_threshold : {DELTA_MSD_THRESHOLD_A2:g} Å^2")

    if first_cross is None:
        print("first_cross         : None (never exceeded)")
    else:
        t_ps, msd, dmsd, k = first_cross
        print(f"first_cross (k={k}) : t={t_ps:g} ps, MSD={msd:.6g} Å^2, ΔMSD={dmsd:.6g} Å^2")

    # CSV保存
    # base = Path(INPUT_PATH).with_suffix("")
    # out_csv = Path(f"{base}.{ATOM_LABEL}2_com.msd_from_t0_lag{LAG_FRAMES}f_dmsd.csv")
    # arr = np.column_stack([np.array(times_ps), np.array(msd_A2), np.array(delta_msd_A2)])
    # np.savetxt(out_csv, arr, delimiter=",", header="time_ps,msd_A2,delta_msd_A2", comments="")
    # print(f"Saved: {out_csv}")
    
    # 既存: 全部保存
    base = Path(INPUT_PATH).with_suffix("")
    out_csv_all = Path(f"{base}.{ATOM_LABEL}2_com.msd_from_t0_lag{LAG_FRAMES}f_dmsd.csv")
    arr = np.column_stack([np.array(times_ps), np.array(msd_A2), np.array(delta_msd_A2)])  # [web:871]
    np.savetxt(out_csv_all, arr, delimiter=",", header="time_ps,msd_A2,delta_msd_A2", comments="")  # [web:813]
    print(f"Saved: {out_csv_all}")

    # 追加: 閾値超えだけ保存
    mask = arr[:, 2] >= DELTA_MSD_THRESHOLD_A2  # delta_msd_A2列で判定（ブールマスク）[web:724]
    arr_exceed = arr[mask]  # 条件を満たす行だけ抽出 [web:724]

    out_csv_exceed = Path(
        f"{base}.{ATOM_LABEL}2_com.msd_from_t0_lag{LAG_FRAMES}f_dmsd_exceed{DELTA_MSD_THRESHOLD_A2:g}.csv"
    )
    np.savetxt(out_csv_exceed, arr_exceed, delimiter=",",
            header="time_ps,msd_A2,delta_msd_A2", comments="")  # [web:813]
    print(f"Saved: {out_csv_exceed}  (rows={arr_exceed.shape[0]})")



if __name__ == "__main__":
    main()
