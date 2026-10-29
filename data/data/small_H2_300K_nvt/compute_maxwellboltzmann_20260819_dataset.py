# # -*- coding: utf-8 -*-
# import re
# from collections import deque
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.offsetbox import AnchoredText


# # -------- user parameters --------
# # input_paths = [
# #     "positions_650K.xyz",
# #     "positions_900K.xyz",
# #     "positions_1150K.xyz",
# #     "positions_1400K.xyz",
# # ]
# input_paths = [
#     "positions_650K.xyz"
# ]
# species_target = "Li"

# lag_frames = 67
# bins = 80

# time_step_ps = 0.015
# dump_stride = 1
# dt_frame_ps = time_step_ps * dump_stride

# out_stem = "positions_ALLTEMP"  # まとめ出力用のファイル名の元
# # --------------------------------


# def iter_xyz_frames(path):
#     with open(path, "r") as f:
#         while True:
#             line = f.readline()
#             if not line:
#                 return
#             line = line.strip()
#             if not line:
#                 continue
#             if not re.fullmatch(r"\d+", line):
#                 continue

#             natoms = int(line)
#             comment = f.readline()
#             if not comment:
#                 return
#             comment = comment.strip()

#             species = []
#             pos = np.empty((natoms, 3), dtype=float)
#             for i in range(natoms):
#                 parts = f.readline().split()
#                 if len(parts) < 4:
#                     raise ValueError("Bad atom line: need 'sp x y z ...'")
#                 species.append(parts[0])
#                 pos[i, 0] = float(parts[1])
#                 pos[i, 1] = float(parts[2])
#                 pos[i, 2] = float(parts[3])

#             yield {"species": species, "pos": pos, "comment": comment}


# def compute_species_displacement_magnitudes_unwrap(path, species_target, lag_frames):
#     it = iter_xyz_frames(path)
#     fr0 = next(it)
#     idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
#     if len(idx) == 0:
#         raise ValueError(f"No atoms found for species_target={species_target} in {path}")

#     buf = deque(maxlen=lag_frames + 1)
#     buf.append(fr0["pos"][idx])
#     n_frames = 1
#     disp = []

#     for fr in it:
#         for k in idx:
#             if fr["species"][k] != species_target:
#                 raise ValueError(f"Atom ordering/species inconsistent across frames in {path}.")

#         buf.append(fr["pos"][idx])
#         n_frames += 1

#         if len(buf) == lag_frames + 1:
#             dr = buf[-1] - buf[0]
#             mag = np.linalg.norm(dr, axis=1)
#             disp.extend(mag.tolist())

#     return np.array(disp, dtype=float), n_frames, len(idx)


# def main():
#     all_disp = []
#     total_n_target = 0
#     total_expected = 0
#     per_file_info = []

#     for path in input_paths:
#         disp_A, n_frames, n_target = compute_species_displacement_magnitudes_unwrap(
#             path, species_target, lag_frames
#         )
#         n_expected = n_target * max(n_frames - lag_frames, 0)
#         all_disp.append(disp_A)
#         total_n_target += n_target
#         total_expected += n_expected
#         per_file_info.append((path, n_frames, n_target, disp_A.size))
#         print(f"[{path}] frames={n_frames}, {species_target} count={n_target}, samples={disp_A.size}")

#     disp_all = np.concatenate(all_disp) if all_disp else np.array([])

#     lag_ps = lag_frames * dt_frame_ps
#     out_png = Path(f"{out_stem}.{species_target}.disp_hist_lag{lag_frames}f.png")

#     mu = float(np.mean(disp_all)) if disp_all.size else float("nan")
#     sd = float(np.std(disp_all, ddof=1)) if disp_all.size >= 2 else float("nan")
#     max_disp = float(np.max(disp_all)) if disp_all.size else float("nan")
#     print(f"[ALL COMBINED] Maximum displacement (|Δr|): {max_disp:.4f} Å")

#     plt.figure(figsize=(6.8, 4.6))
#     plt.hist(disp_all, bins=bins, color="#1f77b4", alpha=0.90, edgecolor="white")
#     plt.xlabel(f"|Δr| over {lag_frames} frames (= {lag_ps:g} ps)  (Å)")
#     plt.ylabel("Counts")
#     plt.title(f"{species_target}: displacement magnitude histogram (unwrap, combined {len(input_paths)} files)")
    
    
#     # ---------------- ここから追加 ----------------
#     if disp_all.size > 0:
#         top_pcts = [1, 2, 5, 10]
#         colors = ['#663399', '#4169E1', '#E75480', '#D2691E']
        
#         for pct, color in zip(top_pcts, colors):
#             # numpy.percentileを使って上位の閾値を計算 (100 - pct)
#             val = np.percentile(disp_all, 100.0 - pct)
#             plt.axvline(val, color=color, linestyle='--', linewidth=1.5, 
#                         label=f'top {pct}% = {val:.3f} Å')
        
#         # 凡例を追加（AnchoredTextと被らないように左上に配置するか、bbox_to_anchorで調整）
#         plt.legend(loc='center right', bbox_to_anchor=(0.995, 0.65), fontsize=9)
#     # ---------------- ここまで追加 ----------------
    

#     file_list_str = "\n".join(Path(p).stem for p in input_paths)
#     at = AnchoredText(
#         #f"files=\n{file_list_str}\n"
#         f"lag={lag_frames}\n"
#         f"n(samples)={disp_all.size} (expected {total_expected})\n"
#         f"mean={mu:.4f} Å\nstd={sd:.4f} Å\nmax={max_disp:.4f} Å",
#         prop=dict(size=8), frameon=True, loc="upper right"
#     )
#     at.patch.set_alpha(0.85)
#     plt.gca().add_artist(at)

#     plt.grid(True, ls="--", alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=180)
#     plt.close()
#     print(f"Saved: {out_png}")


# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
import re
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import skew, kurtosis


# -------- user parameters --------
# input_paths = [
#     "positions_650K.xyz",
#     "positions_900K.xyz",
#     "positions_1150K.xyz",
#     "positions_1400K.xyz",
# ]
input_paths = [
    "output.xyz"
]
species_target = "He"


lag_frames = 1000
bins = 80


time_step_ps = 0.01
dump_stride = 1
dt_frame_ps = time_step_ps * dump_stride


out_stem = "positions_ALLTEMP"  # まとめ出力用のファイル名の元
# --------------------------------


def iter_xyz_frames(path):
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
    it = iter_xyz_frames(path)
    fr0 = next(it)
    idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
    if len(idx) == 0:
        raise ValueError(f"No atoms found for species_target={species_target} in {path}")

    buf = deque(maxlen=lag_frames + 1)
    buf.append(fr0["pos"][idx])
    n_frames = 1
    disp = []

    for fr in it:
        for k in idx:
            if fr["species"][k] != species_target:
                raise ValueError(f"Atom ordering/species inconsistent across frames in {path}.")

        buf.append(fr["pos"][idx])
        n_frames += 1

        if len(buf) == lag_frames + 1:
            dr = buf[-1] - buf[0]
            mag = np.linalg.norm(dr, axis=1)
            disp.extend(mag.tolist())

    return np.array(disp, dtype=float), n_frames, len(idx)


# ---------------- 3次元マクスウェル・ボルツマン分布 関連関数 ----------------
def mb_pdf(r, sigma):
    """
    3次元マクスウェル・ボルツマン分布(速さ/変位量)の確率密度関数
    f(r) = 4*pi*r^2 * (1/(2*pi*sigma^2))^(3/2) * exp(-r^2/(2*sigma^2))
    """
    coeff = 4.0 * np.pi * (1.0 / (2.0 * np.pi * sigma**2))**1.5
    return coeff * r**2 * np.exp(-r**2 / (2.0 * sigma**2))


def mb_sigma_mle(disp):
    """
    3次元MB分布のscaleパラメータsigmaを最尤推定(=モーメント法と一致)
    <r^2> = 3*sigma^2  より sigma = sqrt(mean(r^2)/3)
    """
    return np.sqrt(np.mean(disp**2) / 3.0)


def mb_theoretical_moments(sigma):
    """
    3次元MB分布の理論的な平均・標準偏差・歪度・超過尖度(解析式)
    歪度・超過尖度はsigmaに依存しない無次元の定数
    """
    mean_theory = sigma * np.sqrt(8.0 / np.pi)
    std_theory = sigma * np.sqrt(3.0 - 8.0 / np.pi)

    pi = np.pi
    skew_theory = 2.0 * np.sqrt(2.0) * (16.0 - 5.0 * pi) / (3.0 * pi - 8.0)**1.5
    kurt_theory = 4.0 * (-96.0 + 40.0 * pi - 3.0 * pi**2) / (3.0 * pi - 8.0)**2  # excess kurtosis

    return mean_theory, std_theory, skew_theory, kurt_theory
# --------------------------------------------------------------------------


def main():
    all_disp = []
    total_n_target = 0
    total_expected = 0
    per_file_info = []

    for path in input_paths:
        disp_A, n_frames, n_target = compute_species_displacement_magnitudes_unwrap(
            path, species_target, lag_frames
        )
        n_expected = n_target * max(n_frames - lag_frames, 0)
        all_disp.append(disp_A)
        total_n_target += n_target
        total_expected += n_expected
        per_file_info.append((path, n_frames, n_target, disp_A.size))
        print(f"[{path}] frames={n_frames}, {species_target} count={n_target}, samples={disp_A.size}")

    disp_all = np.concatenate(all_disp) if all_disp else np.array([])

    lag_ps = lag_frames * dt_frame_ps
    out_png = Path(f"maxwellboltzmann_{out_stem}.{species_target}.disp_hist_lag{lag_frames}f.png")

    mu = float(np.mean(disp_all)) if disp_all.size else float("nan")
    sd = float(np.std(disp_all, ddof=1)) if disp_all.size >= 2 else float("nan")
    max_disp = float(np.max(disp_all)) if disp_all.size else float("nan")
    print(f"[ALL COMBINED] Maximum displacement (|Δr|): {max_disp:.4f} Å")

    # ---------------- 実測データの歪度・尖度 ----------------
    skew_data = float(skew(disp_all)) if disp_all.size else float("nan")
    kurt_data = float(kurtosis(disp_all, fisher=True)) if disp_all.size else float("nan")  # 超過尖度

    # ---------------- 3次元MB分布のフィッティング ----------------
    sigma_fit = mb_sigma_mle(disp_all) if disp_all.size else float("nan")
    mean_theory, std_theory, skew_theory, kurt_theory = mb_theoretical_moments(sigma_fit)

    print(f"[Fit] sigma (per-component) = {sigma_fit:.4f} Å")
    print(f"[Data]  mean={mu:.4f}, std={sd:.4f}, skew={skew_data:.4f}, excess_kurt={kurt_data:.4f}")
    print(f"[Theory MB] mean={mean_theory:.4f}, std={std_theory:.4f}, skew={skew_theory:.4f}, excess_kurt={kurt_theory:.4f}")

    # ---------------- プロット ----------------
    plt.figure(figsize=(7.2, 4.8))

    # ヒストグラムは密度(density=True)にして、MB曲線と同じ縦軸(確率密度)で重ねる
    counts, edges, _ = plt.hist(
        disp_all, bins=bins, density=True,
        color="#1f77b4", alpha=0.75, edgecolor="white",
        label="MD data (histogram)"
    )

    # 3次元マクスウェル・ボルツマン分布の理論曲線(通常の線)
    r_line = np.linspace(1e-6, edges[-1], 400)
    plt.plot(r_line, mb_pdf(r_line, sigma_fit), color="#d62728", lw=2.2,
              label=f"3D Maxwell-Boltzmann fit\n(sigma={sigma_fit:.3f} Å)")

    plt.xlabel(f"|Δr| over {lag_frames} frames (= {lag_ps:g} ps)  (Å)")
    plt.ylabel("Probability density")
    plt.title(f"{species_target}: displacement magnitude vs 3D Maxwell-Boltzmann fit")

    plt.legend(loc="upper right", fontsize=9)

    file_list_str = "\n".join(Path(p).stem for p in input_paths)
    at = AnchoredText(
        f"lag={lag_frames}\n"
        f"n(samples)={disp_all.size} (expected {total_expected})\n"
        f"--- Data ---\n"
        f"mean={mu:.4f} Å, std={sd:.4f} Å\n"
        f"skew={skew_data:.4f}, excess_kurt={kurt_data:.4f}\n"
        f"--- MB fit (sigma={sigma_fit:.4f} Å) ---\n"
        f"mean={mean_theory:.4f} Å, std={std_theory:.4f} Å\n"
        f"skew={skew_theory:.4f}, excess_kurt={kurt_theory:.4f}",
        prop=dict(size=9.5), frameon=True, loc="right",
        bbox_to_anchor=(0.999, 0.63),      # ← ここで位置を調整
        bbox_transform=plt.gca().transAxes
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
