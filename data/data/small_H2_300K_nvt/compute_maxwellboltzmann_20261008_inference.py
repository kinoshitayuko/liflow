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
    #"output.xyz"
    "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20260924_1.xyz"
    #"positions_650K.xyz",
]
species_target = "He"


lag_frames = 1
bins = 80


time_step_ps = 0.5
dump_stride = 1
dt_frame_ps = time_step_ps * dump_stride


out_stem = "positions_ALLTEMP"  # まとめ出力用のファイル名の元


axis_labels = ["x", "y", "z"]

# -------- LiFlow prior parameters --------
temperature_K = 300.0                  # 入力xyzの温度(単一温度の場合のみ有効)
mass_amu = {"He": 4.00}                # species_targetの質量 [u]
prior_scales = [10.0, 36.0]             # LiFlowのscale設定(small, large)。1つだけでも可
kB_eV_per_K = 8.617333262e-5
prior_colors = ["#2ca02c", "#9467bd", "#ff7f0e"]
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



def compute_species_displacement_vectors_unwrap(path, species_target, lag_frames):
    """
    指定species_targetの原子について、lag_frames分の変位ベクトル(dx, dy, dz)を収集する。
    (以前の |Δr| だけを返すバージョンから、xyz成分をそのまま返すように変更)
    注意: 座標の差をそのまま取るため、入力xyzがunwrapped座標であることが前提。
    戻り値:
        disp_vec: shape (n_samples, 3) の変位ベクトル配列(Å)
        n_frames: 読み込んだ全フレーム数
        n_target: species_targetに一致する原子数
    """
    it = iter_xyz_frames(path)
    fr0 = next(it)
    idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
    if len(idx) == 0:
        raise ValueError(f"No atoms found for species_target={species_target} in {path}")


    buf = deque(maxlen=lag_frames + 1)
    buf.append(fr0["pos"][idx])
    n_frames = 1
    disp_vec_list = []


    for fr in it:
        for k in idx:
            if fr["species"][k] != species_target:
                raise ValueError(f"Atom ordering/species inconsistent across frames in {path}.")


        buf.append(fr["pos"][idx])
        n_frames += 1


        if len(buf) == lag_frames + 1:
            dr = buf[-1] - buf[0]  # shape (n_target, 3)
            disp_vec_list.append(dr)


    if disp_vec_list:
        disp_vec = np.concatenate(disp_vec_list, axis=0)  # shape (n_samples, 3)
    else:
        disp_vec = np.empty((0, 3), dtype=float)


    return disp_vec, n_frames, len(idx)



# ---------------- 1次元ガウシアン関連関数 ----------------
def gaussian1d_pdf(x, sigma):
    """
    平均0の1次元ガウス分布の確率密度関数
    f(x) = 1/sqrt(2*pi*sigma^2) * exp(-x^2/(2*sigma^2))
    """
    return (1.0 / np.sqrt(2.0 * np.pi * sigma**2)) * np.exp(-x**2 / (2.0 * sigma**2))



def gaussian1d_sigma_mle(disp_1d):
    """
    平均0を仮定した1次元ガウス分布のsigmaを最尤推定
    <x^2> = sigma^2  より sigma = sqrt(mean(x^2))
    """
    return np.sqrt(np.mean(disp_1d**2))



def gaussian1d_theoretical_moments(sigma):
    """
    1次元ガウス分布(平均0)の理論的な平均・標準偏差・歪度・超過尖度
    歪度・超過尖度はsigmaに依存しない定数(いずれも0)
    """
    mean_theory = 0.0
    std_theory = sigma
    skew_theory = 0.0
    kurt_theory = 0.0
    return mean_theory, std_theory, skew_theory, kurt_theory



def prior_sigma_from_scale(scale, temperature_K, mass_u):
    """
    LiFlowの事前分布の標準偏差: sigma = scale * sqrt(kB*T/m)
    (kB: eV/K, m: u)。数値は論文の単位規約に従いÅとして扱う。
    実装側の計算式と一致しているかは要確認。
    """
    return scale * np.sqrt(kB_eV_per_K * temperature_K / mass_u)
# --------------------------------------------------------------------------



def plot_axis_histogram(disp_axis, axis_label, lag_frames, lag_ps, bins,
                         species_target, out_stem, total_expected,
                         prior_sigmas=None):
    """
    1軸方向の変位ヒストグラムを作成し、
    (1) 平均0・RMSで推定した1次元ガウシアン
    (2) LiFlowの事前分布σ(Maxwell-Boltzmann由来)のガウシアン
    を重ねてpng保存する。
    """
    out_png = Path(
        f"20261008_inference_maxwellboltzmann_{out_stem}.{species_target}.disp_hist_lag{lag_frames}f.{axis_label}.png"
    )


    mu = float(np.mean(disp_axis)) if disp_axis.size else float("nan")
    sd = float(np.std(disp_axis, ddof=1)) if disp_axis.size >= 2 else float("nan")
    max_abs_disp = float(np.max(np.abs(disp_axis))) if disp_axis.size else float("nan")
    print(f"[{axis_label}-axis] Maximum |displacement|: {max_abs_disp:.4f} Å")


    # ---------------- 実測データの歪度・尖度 ----------------
    skew_data = float(skew(disp_axis)) if disp_axis.size else float("nan")
    kurt_data = float(kurtosis(disp_axis, fisher=True)) if disp_axis.size else float("nan")  # 超過尖度


    # ---------------- 1次元ガウシアンのフィット(軸ごとに独立推定) ----------------
    sigma_fit = gaussian1d_sigma_mle(disp_axis) if disp_axis.size else float("nan")
    mean_theory, std_theory, skew_theory, kurt_theory = gaussian1d_theoretical_moments(sigma_fit)


    print(f"[{axis_label}-axis][Fit] sigma = {sigma_fit:.4f} Å")
    print(f"[{axis_label}-axis][Data]  mean={mu:.4f}, std={sd:.4f}, skew={skew_data:.4f}, excess_kurt={kurt_data:.4f}")
    print(f"[{axis_label}-axis][Theory Gaussian] mean={mean_theory:.4f}, std={std_theory:.4f}, "
          f"skew={skew_theory:.4f}, excess_kurt={kurt_theory:.4f}")


    # ---------------- プロット ----------------
    plt.figure(figsize=(7.2, 4.8))


    counts, edges, _ = plt.hist(
        disp_axis, bins=bins, density=True,
        color="#1f77b4", alpha=0.75, edgecolor="white",
        label="MD data (histogram)"
    )


    x_min, x_max = edges[0], edges[-1]
    x_line = np.linspace(x_min, x_max, 400)

    # データから推定したσの1次元ガウシアン
    plt.plot(x_line, gaussian1d_pdf(x_line, sigma_fit), color="#d62728", lw=2.2,
              label=f"1D Gaussian fit (zero-mean, RMS)\n(sigma={sigma_fit:.3f} Å)")

    # LiFlow事前分布のσを標準偏差とする平均0ガウシアン
    prior_lines_for_text = []
    if prior_sigmas:
        for i, (scale, s_prior) in enumerate(prior_sigmas):
            c = prior_colors[i % len(prior_colors)]
            plt.plot(
                x_line, gaussian1d_pdf(x_line, s_prior),
                color=c, lw=2.0, ls="--",
                label=f"LiFlow prior (scale={scale:g})\n(sigma={s_prior:.3f} Å)"
            )
            ratio = s_prior / sigma_fit if sigma_fit > 0 else float("nan")
            print(f"[{axis_label}-axis][Prior] scale={scale:g}, sigma_prior={s_prior:.4f} Å, "
                  f"sigma_prior/sigma_fit={ratio:.3f}")
            prior_lines_for_text.append(
                f"scale={scale:g}: sigma={s_prior:.4f} Å (x{ratio:.2f} of fit)"
            )


    plt.xlabel(f"Δ{axis_label} over {lag_frames} frames (= {lag_ps:g} ps)  (Å)")
    plt.ylabel("Probability density")
    plt.title(f"{species_target}: {axis_label}-axis displacement vs Gaussian fit and LiFlow prior")


    plt.legend(loc="upper right", fontsize=8)


    text = (
        f"lag={lag_frames}\n"
        f"n(samples)={disp_axis.size} (expected {total_expected})\n"
        f"--- Data ---\n"
        f"mean={mu:.4f} Å, std={sd:.4f} Å\n"
        f"skew={skew_data:.4f}, excess_kurt={kurt_data:.4f}\n"
        f"--- Gaussian fit (sigma={sigma_fit:.4f} Å) ---\n"
        f"mean={mean_theory:.4f} Å, std={std_theory:.4f} Å\n"
        f"skew={skew_theory:.4f}, excess_kurt={kurt_theory:.4f}"
    )
    if prior_lines_for_text:
        text += "\n--- LiFlow prior ---\n" + "\n".join(prior_lines_for_text)

    at = AnchoredText(
        text,
        prop=dict(size=8.5), frameon=True, loc="right",
        bbox_to_anchor=(0.999, 0.47),      # ← ここで位置を調整
        bbox_transform=plt.gca().transAxes
    )
    at.patch.set_alpha(0.85)
    plt.gca().add_artist(at)


    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Saved: {out_png}")


    return sigma_fit



def main():
    all_disp_vec = []
    total_n_target = 0
    total_expected = 0


    for path in input_paths:
        disp_vec, n_frames, n_target = compute_species_displacement_vectors_unwrap(
            path, species_target, lag_frames
        )
        n_expected = n_target * max(n_frames - lag_frames, 0)
        all_disp_vec.append(disp_vec)
        total_n_target += n_target
        total_expected += n_expected
        print(f"[{path}] frames={n_frames}, {species_target} count={n_target}, samples={disp_vec.shape[0]}")


    disp_vec_all = (
        np.concatenate(all_disp_vec, axis=0) if all_disp_vec else np.empty((0, 3))
    )


    lag_ps = lag_frames * dt_frame_ps


    # ---------------- LiFlow事前分布のσを計算 ----------------
    prior_sigmas = [
        (s, prior_sigma_from_scale(s, temperature_K, mass_amu[species_target]))
        for s in prior_scales
    ]
    print(f"\n[Prior] T={temperature_K:g} K, m({species_target})={mass_amu[species_target]} u")
    for s, sp in prior_sigmas:
        print(f"  scale={s:g} -> sigma_prior={sp:.4f} Å")


    # ---------------- 軸ごとに独立してヒストグラム作成 + 1次元ガウシアンフィット ----------------
    sigma_per_axis = {}
    for axis_idx, axis_label in enumerate(axis_labels):
        disp_axis = disp_vec_all[:, axis_idx]
        sigma_fit = plot_axis_histogram(
            disp_axis, axis_label, lag_frames, lag_ps, bins,
            species_target, out_stem, total_expected,
            prior_sigmas=prior_sigmas
        )
        sigma_per_axis[axis_label] = sigma_fit


    # ---------------- 異方性チェック用のσ比較 ----------------
    print("\n[Anisotropy check] sigma per axis:")
    for axis_label in axis_labels:
        print(f"  sigma_{axis_label} = {sigma_per_axis[axis_label]:.4f} Å")


    sigmas = np.array([sigma_per_axis[a] for a in axis_labels])
    if np.all(np.isfinite(sigmas)):
        sigma_mean = float(np.mean(sigmas))
        sigma_relative_spread = float((np.max(sigmas) - np.min(sigmas)) / sigma_mean)
        print(f"  mean(sigma) = {sigma_mean:.4f} Å, "
              f"relative spread (max-min)/mean = {sigma_relative_spread:.2%}")
        print("  -> 上記のrelative spreadが小さければ、3軸のσ(この時間幅での二次モーメント)がほぼ一致しているといえます。")

        print("\n[Prior vs Fit] sigma_prior / mean(sigma_fit):")
        for s, sp in prior_sigmas:
            print(f"  scale={s:g}: {sp:.4f} / {sigma_mean:.4f} = {sp / sigma_mean:.3f}")



if __name__ == "__main__":
    main()