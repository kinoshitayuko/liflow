# # -*- coding: utf-8 -*-
# import re
# from collections import deque
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.offsetbox import AnchoredText
# from scipy.stats import skew, kurtosis


# # -------- user parameters --------
# input_paths = [
#     "output.xyz",
# ]

# species_target = "He"

# lag_frames = 50
# bins = 80

# # 保存されたフレーム間の時間幅
# time_step_ps = 0.01
# dump_stride = 1
# dt_frame_ps = time_step_ps * dump_stride

# out_stem = "positions_300K"

# # 図のヒストグラム名
# data_label = "MD data: pooled x/y/z components"

# # -------- current LiFlow prior --------
# temperature_K = 300.0

# # モデルのprior計算で使う質量に合わせてください
# mass_amu = {"He": 4.00}

# prior_scales = [36.0]
# kB_eV_per_K = 8.617333262e-5
# prior_colors = ["#9467bd", "#2ca02c", "#ff7f0e"]
# # ------------------------------------


# def iter_xyz_frames(path):
#     """通常のxyz形式から、フレームを順に読み込む。"""
#     with open(path, "r", encoding="utf-8") as f:
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
#                 raise ValueError(
#                     f"Incomplete XYZ frame: comment line missing in {path}"
#                 )

#             species = []
#             pos = np.empty((natoms, 3), dtype=float)

#             for i in range(natoms):
#                 parts = f.readline().split()

#                 if len(parts) < 4:
#                     raise ValueError(
#                         f"Bad atom line in {path}: "
#                         "need 'species x y z ...'"
#                     )

#                 species.append(parts[0])
#                 pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

#             yield {
#                 "species": species,
#                 "pos": pos,
#                 "comment": comment.strip(),
#             }


# def compute_species_displacement_vectors_unwrap(
#     path, species_target, lag_frames
# ):
#     """
#     指定原子種のlag_frames間の変位を集める。

#     入力はunwrapped座標が前提。
#     この関数自体はunwrap処理をしない。

#     Returns
#     -------
#     disp_vec : ndarray, shape (N, 3)
#         変位ベクトル。座標単位がÅなら変位もÅ。
#     n_frames : int
#         フレーム数。
#     n_target : int
#         対象原子数。
#     """
#     if lag_frames < 1:
#         raise ValueError("lag_frames must be >= 1")

#     it = iter_xyz_frames(path)
#     fr0 = next(it, None)

#     if fr0 is None:
#         raise ValueError(f"No XYZ frames found in {path}")

#     reference_species = fr0["species"]
#     idx = [
#         i
#         for i, sp in enumerate(reference_species)
#         if sp == species_target
#     ]

#     if not idx:
#         raise ValueError(
#             f"No atoms found for species_target={species_target} in {path}"
#         )

#     buf = deque(maxlen=lag_frames + 1)
#     buf.append(fr0["pos"][idx])

#     n_frames = 1
#     disp_vec_list = []

#     for fr in it:
#         if fr["species"] != reference_species:
#             raise ValueError(
#                 f"Atom count/order/species inconsistent in {path}"
#             )

#         buf.append(fr["pos"][idx])
#         n_frames += 1

#         if len(buf) == lag_frames + 1:
#             dr = buf[-1] - buf[0]
#             disp_vec_list.append(dr)

#     if disp_vec_list:
#         disp_vec = np.concatenate(disp_vec_list, axis=0)
#     else:
#         disp_vec = np.empty((0, 3), dtype=float)

#     return disp_vec, n_frames, len(idx)


# def gaussian1d_pdf(x, sigma):
#     """平均0、標準偏差sigmaの1次元ガウスPDF。"""
#     if not np.isfinite(sigma) or sigma <= 0:
#         raise ValueError("sigma must be finite and positive")

#     return (
#         np.exp(-0.5 * (x / sigma) ** 2)
#         / (np.sqrt(2.0 * np.pi) * sigma)
#     )


# def gaussian_common_sigma_mle(disp_vec):
#     """
#     平均0・等方ガウスの3軸共通sigma。

#     sigma^2 = mean(dx^2 + dy^2 + dz^2) / 3
#             = mean(全xyz成分の二乗)
#     """
#     if disp_vec.ndim != 2 or disp_vec.shape[1] != 3:
#         raise ValueError("disp_vec must have shape (N, 3)")

#     if disp_vec.shape[0] == 0:
#         raise ValueError("No displacement samples available")

#     return float(np.sqrt(np.mean(disp_vec**2)))


# def prior_sigma_from_scale(scale, temperature_K, mass_u):
#     """
#     現在のprior計算式：
#         sigma = scale * sqrt(kB * T / m)

#     kBはeV/K、mはu。
#     scaleに変位単位への換算を含める規約を使用。
#     モデル側の実際のprior計算と一致するか確認すること。
#     """
#     if scale <= 0 or temperature_K <= 0 or mass_u <= 0:
#         raise ValueError("scale, temperature and mass must be positive")

#     return float(
#         scale * np.sqrt(kB_eV_per_K * temperature_K / mass_u)
#     )


# def plot_pooled_histogram(
#     disp_vec,
#     lag_frames,
#     lag_ps,
#     bins,
#     species_target,
#     out_stem,
#     expected_vectors,
#     prior_sigmas,
# ):
#     """全xyz成分のヒストグラムと、2種類のガウス曲線を描く。"""
#     n_vectors = disp_vec.shape[0]

#     # 符号付きのxyz成分をまとめる。|dr|ではない。
#     disp_components = disp_vec.reshape(-1)
#     n_components = disp_components.size
#     expected_components = 3 * expected_vectors

#     mu = float(np.mean(disp_components))
#     sd = (
#         float(np.std(disp_components, ddof=1))
#         if n_components >= 2 else float("nan")
#     )
#     max_abs = float(np.max(np.abs(disp_components)))

#     if np.ptp(disp_components) > 0:
#         skew_data = float(skew(disp_components))
#         kurt_data = float(kurtosis(disp_components, fisher=True))
#     else:
#         skew_data = float("nan")
#         kurt_data = float("nan")

#     sigma_common = gaussian_common_sigma_mle(disp_vec)
#     mean_square_component = float(np.mean(disp_components**2))
#     rms_vector = float(
#         np.sqrt(np.mean(np.sum(disp_vec**2, axis=1)))
#     )

#     print("\n[Pooled x/y/z components]")
#     print(f"  lag = {lag_frames} frames = {lag_ps:g} ps")
#     print(f"  n_vectors = {n_vectors:,} (expected {expected_vectors:,})")
#     print(
#         f"  n_components = {n_components:,} "
#         f"(expected {expected_components:,})"
#     )
#     print(f"  mean = {mu:.6f} Å")
#     print(f"  std (ddof=1) = {sd:.6f} Å")
#     print(f"  skew = {skew_data:.6f}")
#     print(f"  excess_kurtosis = {kurt_data:.6f}")
#     print(f"  max |component| = {max_abs:.6f} Å")
#     print(f"  mean(component^2) = {mean_square_component:.6f} Å^2")
#     print(f"  sigma_common (zero-mean RMS) = {sigma_common:.6f} Å")
#     print(f"  RMS of 3D displacement magnitude = {rms_vector:.6f} Å")

#     # 軸別の情報は、共通化で見えなくなる差の確認用。
#     axis_means = np.mean(disp_vec, axis=0)
#     axis_sigmas = np.sqrt(np.mean(disp_vec**2, axis=0))

#     print("\n[Axis diagnostics: zero-mean RMS]")
#     for i, axis in enumerate(["x", "y", "z"]):
#         print(
#             f"  {axis}: mean={axis_means[i]:.6f} Å, "
#             f"sigma={axis_sigmas[i]:.6f} Å"
#         )

#     if sigma_common == 0:
#         raise ValueError(
#             "All displacement components are zero. "
#             "A Gaussian PDF with sigma=0 cannot be plotted."
#         )

#     prior_text_lines = []

#     print("\n[Current prior vs common sigma]")
#     for scale, sigma_prior in prior_sigmas:
#         ratio = sigma_prior / sigma_common
#         print(
#             f"  scale={scale:g}: sigma_prior={sigma_prior:.6f} Å, "
#             f"sigma_prior/sigma_common={ratio:.6f}"
#         )
#         prior_text_lines.append(
#             f"scale={scale:g}: sigma={sigma_prior:.4f} Å\n"
#             f"prior/common={ratio:.3f}"
#         )

#     out_png = Path(
#         f"{out_stem}.{species_target}."
#         f"pooled_xyz_hist_lag{lag_frames}f.png"
#     )
#     out_png.parent.mkdir(parents=True, exist_ok=True)

#     # 右側に凡例と統計情報を置き、データを隠さない。
#     fig, ax = plt.subplots(figsize=(10.8, 5.2))
#     fig.subplots_adjust(left=0.09, right=0.67, bottom=0.16, top=0.87)

#     _, edges, _ = ax.hist(
#         disp_components,
#         bins=bins,
#         density=True,
#         color="#1f77b4",
#         alpha=0.75,
#         edgecolor="white",
#         label=data_label,
#     )

#     # データ範囲とガウスの幅の両方を表示範囲に含める。
#     max_sigma = max(
#         [sigma_common] + [sigma for _, sigma in prior_sigmas]
#     )
#     x_limit = max(abs(edges[0]), abs(edges[-1]), 4.0 * max_sigma)
#     x_line = np.linspace(-x_limit, x_limit, 4001)

#     ax.plot(
#         x_line,
#         gaussian1d_pdf(x_line, sigma_common),
#         color="#d62728",
#         lw=2.2,
#         label=(
#             "Common zero-mean Gaussian\n"
#             f"(sigma={sigma_common:.3f} Å)"
#         ),
#     )

#     for i, (scale, sigma_prior) in enumerate(prior_sigmas):
#         ax.plot(
#             x_line,
#             gaussian1d_pdf(x_line, sigma_prior),
#             color=prior_colors[i % len(prior_colors)],
#             lw=2.0,
#             ls="--",
#             label=(
#                 f"Current prior (scale={scale:g})\n"
#                 f"(sigma={sigma_prior:.3f} Å)"
#             ),
#         )

#     ax.set_xlim(-x_limit, x_limit)
#     ax.set_xlabel(
#         f"Displacement component over {lag_frames} frames "
#         f"(= {lag_ps:g} ps) (Å)"
#     )
#     ax.set_ylabel("Probability density")
#     ax.set_title(
#         f"{species_target}: pooled x/y/z displacement components"
#     )
#     ax.grid(True, ls="--", alpha=0.3)

#     ax.legend(
#         loc="upper left",
#         bbox_to_anchor=(1.02, 1.0),
#         fontsize=9,
#     )

#     text = (
#         f"lag={lag_frames} frames ({lag_ps:g} ps)\n"
#         f"n(vectors)={n_vectors:,}\n"
#         f"n(components)={n_components:,}\n"
#         f"--- Pooled data ---\n"
#         f"mean={mu:.4f} Å\n"
#         f"std={sd:.4f} Å\n"
#         f"skew={skew_data:.4f}\n"
#         f"excess_kurt={kurt_data:.4f}\n"
#         f"--- Common Gaussian ---\n"
#         f"mean=0 Å\n"
#         f"sigma={sigma_common:.4f} Å"
#     )

#     if prior_text_lines:
#         text += (
#             "\n--- Current prior ---\n"
#             + "\n".join(prior_text_lines)
#         )

#     at = AnchoredText(
#         text,
#         prop=dict(size=9),
#         frameon=True,
#         loc="upper left",
#         bbox_to_anchor=(1.02, 0.61),
#         bbox_transform=ax.transAxes,
#         borderpad=0.0,
#     )
#     at.patch.set_alpha(0.9)
#     ax.add_artist(at)

#     fig.savefig(out_png, dpi=180, bbox_inches="tight")
#     plt.close(fig)

#     print(f"\nSaved: {out_png}")

#     return sigma_common


# def main():
#     if not input_paths:
#         raise ValueError("input_paths is empty")

#     if dt_frame_ps <= 0:
#         raise ValueError("dt_frame_ps must be positive")

#     if bins < 1:
#         raise ValueError("bins must be >= 1")

#     if species_target not in mass_amu:
#         raise ValueError(f"Mass is not specified for {species_target}")

#     all_disp_vec = []
#     total_expected_vectors = 0

#     for path in input_paths:
#         disp_vec, n_frames, n_target = (
#             compute_species_displacement_vectors_unwrap(
#                 path, species_target, lag_frames
#             )
#         )

#         n_expected = n_target * max(n_frames - lag_frames, 0)

#         all_disp_vec.append(disp_vec)
#         total_expected_vectors += n_expected

#         print(
#             f"[{path}] frames={n_frames}, "
#             f"{species_target} count={n_target}, "
#             f"displacement vectors={disp_vec.shape[0]:,}"
#         )

#     disp_vec_all = np.concatenate(all_disp_vec, axis=0)

#     if disp_vec_all.shape[0] == 0:
#         raise ValueError(
#             "No displacement samples. "
#             "The trajectory needs more than lag_frames frames."
#         )

#     if not np.isfinite(disp_vec_all).all():
#         raise ValueError("Displacement data contain NaN or infinity")

#     if disp_vec_all.shape[0] != total_expected_vectors:
#         raise ValueError("Actual and expected vector counts do not match")

#     lag_ps = lag_frames * dt_frame_ps

#     prior_sigmas = [
#         (
#             scale,
#             prior_sigma_from_scale(
#                 scale,
#                 temperature_K,
#                 mass_amu[species_target],
#             ),
#         )
#         for scale in prior_scales
#     ]

#     print(
#         f"\n[Prior settings] T={temperature_K:g} K, "
#         f"mass({species_target})={mass_amu[species_target]:g} u"
#     )
#     for scale, sigma_prior in prior_sigmas:
#         print(
#             f"  scale={scale:g} -> sigma_prior={sigma_prior:.6f} Å"
#         )

#     sigma_common = plot_pooled_histogram(
#         disp_vec=disp_vec_all,
#         lag_frames=lag_frames,
#         lag_ps=lag_ps,
#         bins=bins,
#         species_target=species_target,
#         out_stem=out_stem,
#         expected_vectors=total_expected_vectors,
#         prior_sigmas=prior_sigmas,
#     )

#     print(
#         "\n[Result] Common per-axis sigma for a zero-mean "
#         f"isotropic Gaussian: {sigma_common:.6f} Å"
#     )


# if __name__ == "__main__":
#     main()



# -*- coding: utf-8 -*-
import re
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.special import logsumexp
from scipy.stats import skew, kurtosis


# -------- user parameters --------
input_paths = ["output.xyz"]
species_target = "He"
lag_frames = 50
bins = 80
time_step_ps = 0.01
dump_stride = 1
dt_frame_ps = time_step_ps * dump_stride
out_stem = "positions_300K"
data_label = "MD data: pooled x/y/z components"

temperature_K = 300.0
mass_amu = {"He": 4.00}
prior_scales = [36.0]
kB_eV_per_K = 8.617333262e-5
prior_colors = ["#9467bd", "#ff7f0e", "#8c564b"]

# Both mixture means are fixed at zero. Fit pooled scalar components.
mixture_max_iter = 500
mixture_tol = 1e-8         # Change in mean log likelihood
mixture_min_sigma = 1e-5  # Angstrom; avoids a zero-width component
# ---------------------------------


def iter_xyz_frames(path):
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return
            line = line.strip()
            if not line or not re.fullmatch(r"\d+", line):
                continue
            natoms = int(line)
            comment = f.readline()
            if not comment:
                raise ValueError(f"Missing XYZ comment line in {path}")
            species = []
            pos = np.empty((natoms, 3), dtype=float)
            for i in range(natoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Bad atom line in {path}: need species x y z")
                species.append(parts[0])
                pos[i] = [float(v) for v in parts[1:4]]
            yield {"species": species, "pos": pos, "comment": comment.strip()}


def compute_species_displacement_vectors_unwrap(path, species_target, lag_frames):
    """Input must already be unwrapped; this function does not unwrap."""
    it = iter_xyz_frames(path)
    first = next(it, None)
    if first is None:
        raise ValueError(f"No XYZ frames in {path}")
    species_ref = first["species"]
    idx = [i for i, sp in enumerate(species_ref) if sp == species_target]
    if not idx:
        raise ValueError(f"No {species_target} atoms in {path}")
    buf = deque(maxlen=lag_frames + 1)
    buf.append(first["pos"][idx])
    vectors = []
    n_frames = 1
    for frame in it:
        if frame["species"] != species_ref:
            raise ValueError(f"Atom count/order/species changed in {path}")
        buf.append(frame["pos"][idx])
        n_frames += 1
        if len(buf) == lag_frames + 1:
            vectors.append(buf[-1] - buf[0])
    disp = np.concatenate(vectors, axis=0) if vectors else np.empty((0, 3))
    return disp, n_frames, len(idx)


def gaussian_pdf(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def mixture_pdf(x, weights, sigmas):
    return sum(w * gaussian_pdf(x, s) for w, s in zip(weights, sigmas))


def prior_sigma_from_scale(scale, temperature_K, mass_u):
    return float(scale * np.sqrt(kB_eV_per_K * temperature_K / mass_u))


def fit_zero_mean_gaussian_mixture(values, max_iter=500, tol=1e-8, min_sigma=1e-5):
    """Two-component zero-mean Gaussian mixture MLE via EM on pooled 1D data."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite components are required")
    x2 = x * x
    rms = float(np.sqrt(np.mean(x2)))
    if rms <= min_sigma:
        raise ValueError("Displacements are too small for the selected min_sigma")
    variance_floor = min_sigma ** 2
    starts = [
        (0.70, 0.45, 1.7),
        (0.85, 0.60, 2.5),
        (0.50, 0.50, 1.4),
        (0.90, 0.70, 3.0),
    ]
    best = None
    for narrow_weight, narrow_scale, broad_scale in starts:
        weights = np.array([narrow_weight, 1.0 - narrow_weight])
        variances = np.maximum(
            (rms * np.array([narrow_scale, broad_scale])) ** 2,
            variance_floor,
        )
        old_ll = -np.inf
        converged = False
        for iteration in range(1, max_iter + 1):
            log_terms = (
                np.log(weights)[None, :]
                - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
                - 0.5 * x2[:, None] / variances[None, :]
            )
            log_density = logsumexp(log_terms, axis=1)
            ll = float(np.mean(log_density))
            if np.isfinite(old_ll) and abs(ll - old_ll) < tol:
                converged = True
                break
            responsibilities = np.exp(log_terms - log_density[:, None])
            nk = responsibilities.sum(axis=0)
            if np.any(nk <= 1e-12):
                break
            weights = nk / x.size
            variances = np.maximum(
                (responsibilities * x2[:, None]).sum(axis=0) / nk,
                variance_floor,
            )
            old_ll = ll
        log_terms = (
            np.log(weights)[None, :]
            - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
            - 0.5 * x2[:, None] / variances[None, :]
        )
        final_ll = float(np.mean(logsumexp(log_terms, axis=1)))
        order = np.argsort(variances)
        result = {
            "weights": weights[order],
            "sigmas": np.sqrt(variances[order]),
            "mean_log_likelihood": final_ll,
            "iterations": iteration,
            "converged": converged,
        }
        if best is None or final_ll > best["mean_log_likelihood"]:
            best = result
    return best


def plot_pooled_histogram(disp_vec, lag_ps, expected_vectors, prior_sigmas, mixture):
    components = disp_vec.reshape(-1)
    n_vectors = len(disp_vec)
    mu = float(np.mean(components))
    sd = float(np.std(components, ddof=1))
    skew_data = float(skew(components))
    kurt_data = float(kurtosis(components, fisher=True))
    sigma_common = float(np.sqrt(np.mean(components ** 2)))
    weights, sigmas = mixture["weights"], mixture["sigmas"]
    mix_variance = float(np.sum(weights * sigmas ** 2))
    mix_sigma = np.sqrt(mix_variance)
    mix_kurt = float(3.0 * np.sum(weights * sigmas ** 4) / mix_variance ** 2 - 3.0)

    print(f"\n[Pooled data] vectors={n_vectors:,}, components={components.size:,}")
    print(f"mean={mu:.6f} Å, std={sd:.6f} Å")
    print(f"skew={skew_data:.6f}, excess_kurt={kurt_data:.6f}")
    print(f"Common zero-mean sigma={sigma_common:.6f} Å")
    print("\n[Zero-mean two-component mixture]")
    print(f"narrow: weight={weights[0]:.6f}, sigma={sigmas[0]:.6f} Å")
    print(f"broad:  weight={weights[1]:.6f}, sigma={sigmas[1]:.6f} Å")
    print(f"mixture std={mix_sigma:.6f} Å, skew=0, excess_kurt={mix_kurt:.6f}")
    print(f"iterations={mixture['iterations']}, converged={mixture['converged']}")
    if not mixture["converged"]:
        print("Warning: best fit did not satisfy the convergence tolerance; consider increasing max_iter.")

    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    fig.subplots_adjust(left=0.085, right=0.67, bottom=0.14, top=0.90)
    _, edges, _ = ax.hist(
        components, bins=bins, density=True, color="#1f77b4",
        alpha=0.75, edgecolor="white", label=data_label,
    )
    largest_sigma = max([sigma_common, *sigmas, *[s for _, s in prior_sigmas]])
    limit = max(abs(edges[0]), abs(edges[-1]), 4.0 * largest_sigma)
    line = np.linspace(-limit, limit, 4001)
    # ax.plot(
    #     line, gaussian_pdf(line, sigma_common), color="#d62728", lw=2.2,
    #     label=f"Common zero-mean Gaussian\n(sigma={sigma_common:.3f} Å)",
    # )
    ax.plot(
        line, mixture_pdf(line, weights, sigmas), color="#2ca02c", lw=2.3,
        label="Zero-mean Gaussian mixture (2 components)",
    )
    for i, (scale, sigma) in enumerate(prior_sigmas):
        ax.plot(
            line, gaussian_pdf(line, sigma), ls="--", lw=2.0,
            color=prior_colors[i % len(prior_colors)],
            label=f"Current prior (scale={scale:g})\n(sigma={sigma:.3f} Å)",
        )
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(f"Displacement component over {lag_frames} frames (= {lag_ps:g} ps) (Å)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{species_target}: pooled x/y/z displacement components")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

    text = (
        f"lag={lag_frames} frames ({lag_ps:g} ps)\n"
        f"n(vectors)={n_vectors:,}\n"
        f"n(components)={components.size:,}\n"
        f"--- Pooled data ---\n"
        f"mean={mu:.4f} Å, std={sd:.4f} Å\n"
        f"skew={skew_data:.4f}\n"
        f"excess_kurt={kurt_data:.4f}\n"
        # f"--- Common Gaussian ---\n"
        # f"sigma={sigma_common:.4f} Å\n"
        f"--- Zero-mean mixture ---\n"
        f"narrow: w={weights[0]:.4f}, sigma={sigmas[0]:.4f} Å\n"
        f"broad:  w={weights[1]:.4f}, sigma={sigmas[1]:.4f} Å\n"
        f"std={mix_sigma:.4f} Å\n"
        f"skew=0, excess_kurt={mix_kurt:.4f}"
    )
    for scale, sigma in prior_sigmas:
        text += f"\n--- Current prior ---\nscale={scale:g}: sigma={sigma:.4f} Å"
    box = AnchoredText(
        text, prop=dict(size=9), frameon=True, loc="upper left",
        bbox_to_anchor=(1.02, 0.59), bbox_transform=ax.transAxes, borderpad=0.0,
    )
    box.patch.set_alpha(0.9)
    ax.add_artist(box)
    output = Path(f"{out_stem}.{species_target}.pooled_xyz_mixture_lag{lag_frames}f_20261019.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output}")


def main():
    if not input_paths:
        raise ValueError("input_paths is empty")
    if lag_frames < 1 or dt_frame_ps <= 0 or bins < 1:
        raise ValueError("Invalid lag_frames, dt_frame_ps or bins")
    if mixture_max_iter < 1 or mixture_tol <= 0 or mixture_min_sigma <= 0:
        raise ValueError("Invalid mixture fit settings")
    if species_target not in mass_amu:
        raise ValueError(f"Mass not specified for {species_target}")
    if temperature_K <= 0 or mass_amu[species_target] <= 0 or any(s <= 0 for s in prior_scales):
        raise ValueError("Temperature, mass and prior scales must be positive")
    all_vectors = []
    expected_vectors = 0
    for path in input_paths:
        disp, frames, n_target = compute_species_displacement_vectors_unwrap(
            path, species_target, lag_frames)
        all_vectors.append(disp)
        expected_vectors += n_target * max(frames - lag_frames, 0)
        print(f"[{path}] frames={frames}, {species_target} count={n_target}, vectors={len(disp):,}")
    disp_vec = np.concatenate(all_vectors, axis=0)
    if len(disp_vec) == 0:
        raise ValueError("No displacement samples: more than lag_frames frames are needed")
    if not np.isfinite(disp_vec).all():
        raise ValueError("Displacements contain NaN or infinity")
    if len(disp_vec) != expected_vectors:
        raise ValueError("Unexpected displacement sample count")
    mixture = fit_zero_mean_gaussian_mixture(
        disp_vec.reshape(-1), max_iter=mixture_max_iter,
        tol=mixture_tol, min_sigma=mixture_min_sigma)
    prior_sigmas = [
        (s, prior_sigma_from_scale(s, temperature_K, mass_amu[species_target]))
        for s in prior_scales
    ]
    plot_pooled_histogram(
        disp_vec, lag_frames * dt_frame_ps, expected_vectors, prior_sigmas, mixture)


if __name__ == "__main__":
    main()
