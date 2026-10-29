# # -*- coding: utf-8 -*-
# import csv
# import re
# from collections import deque
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import skew, kurtosis


# # -------- user parameters --------
# input_paths = ["output.xyz"]
# species_target = "He"

# lag_frames = 50
# bins = 80

# time_step_ps = 0.01
# dump_stride = 1
# dt_frame_ps = time_step_ps * dump_stride

# out_stem = "positions_300K"

# # Figure settings for the right side of the slide
# figure_size = (5.0, 6.4)
# figure_dpi = 300
# font_size = 11
# legend_font_size = 9

# before_color = "#1f77b4"
# after_color = "#ff7f0e"

# # Match the original code's standard-deviation convention.
# std_ddof = 1

# # Pearson kurtosis: normal distribution -> 3
# # The same definition is used for both the transform and the legends.
# stats_bias = True

# save_pdf = True
# save_statistics_csv = True
# # ---------------------------------


# def iter_xyz_frames(path):
#     with open(path, "r", encoding="utf-8") as f:
#         while True:
#             line = f.readline()
#             if not line:
#                 return

#             line = line.strip()
#             if not line or not re.fullmatch(r"\d+", line):
#                 continue

#             natoms = int(line)
#             comment = f.readline()
#             if not comment:
#                 raise ValueError(f"Missing XYZ comment line in {path}")

#             species = []
#             pos = np.empty((natoms, 3), dtype=float)

#             for i in range(natoms):
#                 parts = f.readline().split()
#                 if len(parts) < 4:
#                     raise ValueError(
#                         f"Bad atom line in {path}: need species x y z"
#                     )

#                 species.append(parts[0])
#                 pos[i] = [float(v) for v in parts[1:4]]

#             yield {
#                 "species": species,
#                 "pos": pos,
#                 "comment": comment.strip(),
#             }


# def compute_species_displacement_vectors_unwrap(
#     path, species_target, lag_frames
# ):
#     """Input coordinates must already be unwrapped."""
#     it = iter_xyz_frames(path)
#     first = next(it, None)

#     if first is None:
#         raise ValueError(f"No XYZ frames in {path}")

#     species_ref = first["species"]
#     idx = [
#         i for i, sp in enumerate(species_ref)
#         if sp == species_target
#     ]

#     if not idx:
#         raise ValueError(f"No {species_target} atoms in {path}")

#     buf = deque(maxlen=lag_frames + 1)
#     buf.append(first["pos"][idx])

#     vectors = []
#     n_frames = 1

#     for frame in it:
#         if frame["species"] != species_ref:
#             raise ValueError(
#                 f"Atom count/order/species changed in {path}"
#             )

#         buf.append(frame["pos"][idx])
#         n_frames += 1

#         if len(buf) == lag_frames + 1:
#             vectors.append(buf[-1] - buf[0])

#     disp = (
#         np.concatenate(vectors, axis=0)
#         if vectors else np.empty((0, 3), dtype=float)
#     )

#     return disp, n_frames, len(idx)


# def calculate_statistics(values):
#     values = np.asarray(values, dtype=float).reshape(-1)

#     if values.size < 2:
#         raise ValueError("At least two samples are required")
#     if not np.isfinite(values).all():
#         raise ValueError("Values contain NaN or infinity")

#     sd = float(np.std(values, ddof=std_ddof))
#     if not np.isfinite(sd) or sd <= 0:
#         raise ValueError(
#             "Standard deviation must be finite and positive"
#         )

#     result = {
#         "n_samples": int(values.size),
#         "mean": float(np.mean(values)),
#         "std": sd,
#         "skew": float(skew(values, bias=stats_bias)),
#         "kurt": float(
#             kurtosis(
#                 values,
#                 fisher=False,
#                 bias=stats_bias,
#             )
#         ),
#     }

#     if not all(
#         np.isfinite(result[key])
#         for key in ("mean", "std", "skew", "kurt")
#     ):
#         raise ValueError("Non-finite statistics were obtained")

#     return result


# def signed_log_transform(values, sigma, k):
#     if not np.isfinite(sigma) or sigma <= 0:
#         raise ValueError("sigma must be finite and positive")
#     if not np.isfinite(k) or k <= 0:
#         raise ValueError("k must be finite and positive")

#     values = np.asarray(values, dtype=float)

#     transformed = (
#         np.sign(values)
#         * np.log1p(k * np.abs(values) / sigma)
#     )

#     if not np.isfinite(transformed).all():
#         raise ValueError("Transformed values contain NaN or infinity")

#     return transformed


# def inverse_signed_log_transform(values, sigma, k):
#     values = np.asarray(values, dtype=float)

#     return (
#         np.sign(values)
#         * (sigma / k)
#         * np.expm1(np.abs(values))
#     )


# def make_legend_label(stats, unit=None):
#     suffix = f" {unit}" if unit else ""

#     return (
#         f"mean = {stats['mean']:.4f}{suffix}\n"
#         f"std = {stats['std']:.4f}{suffix}\n"
#         f"skew = {stats['skew']:.4f}\n"
#         f"kurt = {stats['kurt']:.4f}"
#     )


# def draw_histogram(
#     ax,
#     values,
#     stats,
#     title,
#     xlabel,
#     color,
#     unit=None,
# ):
#     limit = float(np.max(np.abs(values)))
#     if not np.isfinite(limit) or limit <= 0:
#         raise ValueError("Invalid histogram range")

#     # Include every sample and use bins symmetric about zero.
#     edges = np.linspace(-limit, limit, bins + 1)

#     ax.hist(
#         values,
#         bins=edges,
#         density=True,
#         color=color,
#         alpha=0.85,
#         edgecolor="white",
#         linewidth=0.35,
#         label=make_legend_label(stats, unit=unit),
#     )

#     ax.set_xlim(-1.03 * limit, 1.03 * limit)
#     ax.set_ylim(bottom=0)

#     ax.set_title(title, fontsize=font_size + 1, pad=8)
#     ax.set_xlabel(xlabel, fontsize=font_size)
#     ax.set_ylabel("Probability density", fontsize=font_size)

#     ax.tick_params(axis="both", labelsize=font_size - 1)
#     ax.grid(axis="y", linestyle="--", alpha=0.25)
#     ax.set_axisbelow(True)

#     ax.legend(
#         loc="upper right",
#         fontsize=legend_font_size,
#         frameon=True,
#         framealpha=0.95,
#         facecolor="white",
#         edgecolor="#cccccc",
#         handlelength=1.0,
#         handletextpad=0.6,
#         borderpad=0.7,
#     )


# def plot_before_after(
#     x_before,
#     x_after,
#     stats_before,
#     stats_after,
#     lag_ps,
#     output_base,
# ):
#     plt.rcParams.update({
#         "font.family": "DejaVu Sans",
#         "font.size": font_size,
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#     })

#     fig, axes = plt.subplots(
#         2,
#         1,
#         figsize=figure_size,
#         sharex=False,
#         sharey=False,
#     )

#     draw_histogram(
#         ax=axes[0],
#         values=x_before,
#         stats=stats_before,
#         title="Before transformation",
#         xlabel=r"$\Delta x$ ($\mathrm{\AA}$)",
#         color=before_color,
#         unit="Å",
#     )

#     draw_histogram(
#         ax=axes[1],
#         values=x_after,
#         stats=stats_after,
#         title="After signed-log transformation",
#         xlabel=r"Transformed $\Delta x$ (dimensionless)",
#         color=after_color,
#     )

#     fig.suptitle(
#         f"{species_target}: x-displacement over {lag_ps:g} ps",
#         fontsize=font_size + 1,
#         y=0.985,
#     )

#     fig.tight_layout(
#         rect=(0.0, 0.0, 1.0, 0.95),
#         h_pad=1.6,
#     )

#     png_path = output_base.with_suffix(".png")
#     fig.savefig(
#         png_path,
#         dpi=figure_dpi,
#         bbox_inches="tight",
#         facecolor="white",
#     )
#     print(f"Saved: {png_path}")

#     if save_pdf:
#         pdf_path = output_base.with_suffix(".pdf")
#         fig.savefig(
#             pdf_path,
#             bbox_inches="tight",
#             facecolor="white",
#         )
#         print(f"Saved: {pdf_path}")

#     plt.close(fig)


# def save_statistics(
#     output_base,
#     stats_before,
#     stats_after,
#     sigma,
#     k,
#     lag_ps,
#     inverse_max_error,
# ):
#     csv_path = output_base.with_suffix(".csv")

#     fieldnames = [
#         "stage",
#         "species",
#         "component",
#         "lag_frames",
#         "lag_ps",
#         "n_samples",
#         "mean",
#         "std",
#         "skew",
#         "kurt",
#         "kurtosis_definition",
#         "std_ddof",
#         "stats_bias",
#         "transform_sigma",
#         "transform_k",
#         "inverse_max_abs_error",
#     ]

#     with open(
#         csv_path, "w", newline="", encoding="utf-8-sig"
#     ) as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         for stage, stats in (
#             ("before", stats_before),
#             ("after", stats_after),
#         ):
#             writer.writerow({
#                 "stage": stage,
#                 "species": species_target,
#                 "component": "x",
#                 "lag_frames": lag_frames,
#                 "lag_ps": lag_ps,
#                 **stats,
#                 "kurtosis_definition": "Pearson (normal=3)",
#                 "std_ddof": std_ddof,
#                 "stats_bias": stats_bias,
#                 "transform_sigma": sigma,
#                 "transform_k": k,
#                 "inverse_max_abs_error": inverse_max_error,
#             })

#     print(f"Saved: {csv_path}")


# def main():
#     if not input_paths:
#         raise ValueError("input_paths is empty")

#     if not isinstance(lag_frames, int) or lag_frames < 1:
#         raise ValueError("lag_frames must be a positive integer")

#     if not isinstance(bins, int) or bins < 1:
#         raise ValueError("bins must be a positive integer")

#     if not np.isfinite(dt_frame_ps) or dt_frame_ps <= 0:
#         raise ValueError("dt_frame_ps must be finite and positive")

#     if std_ddof not in (0, 1):
#         raise ValueError("std_ddof must be 0 or 1")

#     all_vectors = []
#     expected_vectors = 0

#     for path in input_paths:
#         disp, frames, n_target = (
#             compute_species_displacement_vectors_unwrap(
#                 path,
#                 species_target,
#                 lag_frames,
#             )
#         )

#         all_vectors.append(disp)
#         expected_vectors += (
#             n_target * max(frames - lag_frames, 0)
#         )

#         print(
#             f"[{path}] frames={frames}, "
#             f"{species_target} count={n_target}, "
#             f"vectors={len(disp):,}"
#         )

#     disp_vec = np.concatenate(all_vectors, axis=0)

#     if len(disp_vec) < 2:
#         raise ValueError(
#             "At least two displacement samples are required"
#         )

#     if not np.isfinite(disp_vec).all():
#         raise ValueError("Displacements contain NaN or infinity")

#     if len(disp_vec) != expected_vectors:
#         raise ValueError("Unexpected displacement sample count")

#     # Use x only; do not pool x/y/z.
#     x_before = disp_vec[:, 0]
#     stats_before = calculate_statistics(x_before)

#     sigma_x = stats_before["std"]
#     k_x = stats_before["kurt"]

#     x_after = signed_log_transform(
#         x_before,
#         sigma=sigma_x,
#         k=k_x,
#     )
#     stats_after = calculate_statistics(x_after)

#     restored = inverse_signed_log_transform(
#         x_after,
#         sigma=sigma_x,
#         k=k_x,
#     )

#     inverse_max_error = float(
#         np.max(np.abs(restored - x_before))
#     )

#     if not np.allclose(
#         restored,
#         x_before,
#         rtol=1e-10,
#         atol=1e-12,
#     ):
#         raise ValueError("Inverse-transform consistency check failed")

#     lag_ps = lag_frames * dt_frame_ps

#     output_base = Path(
#         f"{out_stem}_{species_target}_x_signedlog_"
#         f"lag{lag_frames}f"
#     )
#     output_base.parent.mkdir(parents=True, exist_ok=True)

#     print(f"\nSamples: {len(x_before):,}")
#     print(f"Lag: {lag_frames} frames = {lag_ps:g} ps")
#     print(f"Transform sigma_x: {sigma_x:.8g} Å")
#     print(f"Transform K_x: {k_x:.8g} (Pearson kurtosis)")

#     for title, stats in (
#         ("Before transformation", stats_before),
#         ("After transformation", stats_after),
#     ):
#         print(f"\n[{title}]")
#         for key in ("mean", "std", "skew", "kurt"):
#             print(f"{key} = {stats[key]:.8g}")

#     print(
#         "\nInverse-transform max absolute error: "
#         f"{inverse_max_error:.6e} Å"
#     )

#     plot_before_after(
#         x_before=x_before,
#         x_after=x_after,
#         stats_before=stats_before,
#         stats_after=stats_after,
#         lag_ps=lag_ps,
#         output_base=output_base,
#     )

#     if save_statistics_csv:
#         save_statistics(
#             output_base=output_base,
#             stats_before=stats_before,
#             stats_after=stats_after,
#             sigma=sigma_x,
#             k=k_x,
#             lag_ps=lag_ps,
#             inverse_max_error=inverse_max_error,
#         )


# if __name__ == "__main__":
#     main()






# -*- coding: utf-8 -*-
import csv
import re
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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

# Signed-log transformation:
# z = sign(x) * log(1 + log_strength * K_x * abs(x) / sigma_x)
log_strength = 0.1

# Optional: generate a separate before/after figure for each strength.
compare_strengths = True
strength_candidates = [0.05, 0.1, 0.2, 0.5, 1.0]

# Figure settings
figure_size = (5.0, 5.6)
figure_dpi = 300
font_size = 11
legend_font_size = 9

before_color = "#1f77b4"
after_color = "#ff7f0e"

legend_location = "upper right"

# Smaller values reduce whitespace.
layout_pad = 0.45
subplot_h_pad = 0.75
save_pad_inches = 0.03

# Statistics
std_ddof = 1
stats_bias = True

# Pearson kurtosis is used throughout: normal distribution -> 3.
save_pdf = True
save_statistics_csv = True
# ---------------------------------


def iter_xyz_frames(path):
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()

            if not line:
                return

            line = line.strip()
            if not line:
                continue

            if not re.fullmatch(r"\d+", line):
                raise ValueError(
                    f"Expected atom count in {path}, got: {line!r}"
                )

            natoms = int(line)
            if natoms < 1:
                raise ValueError(f"Invalid atom count in {path}")

            comment = f.readline()
            if not comment:
                raise ValueError(
                    f"Missing XYZ comment line in {path}"
                )

            species = []
            pos = np.empty((natoms, 3), dtype=float)

            for i in range(natoms):
                parts = f.readline().split()

                if len(parts) < 4:
                    raise ValueError(
                        f"Bad atom line {i + 1} in {path}: "
                        "need species x y z"
                    )

                species.append(parts[0])
                pos[i] = [float(v) for v in parts[1:4]]

            if not np.isfinite(pos).all():
                raise ValueError(
                    f"Coordinates contain NaN or infinity in {path}"
                )

            yield {
                "species": species,
                "pos": pos,
                "comment": comment.strip(),
            }


def compute_species_displacement_vectors_unwrap(
    path, species_target, lag_frames
):
    """Input coordinates must already be unwrapped."""
    frames = iter_xyz_frames(path)
    first = next(frames, None)

    if first is None:
        raise ValueError(f"No XYZ frames in {path}")

    species_ref = first["species"]
    idx = [
        i for i, sp in enumerate(species_ref)
        if sp == species_target
    ]

    if not idx:
        raise ValueError(
            f"No {species_target} atoms in {path}"
        )

    buf = deque(maxlen=lag_frames + 1)
    buf.append(first["pos"][idx])

    vectors = []
    n_frames = 1

    for frame in frames:
        if frame["species"] != species_ref:
            raise ValueError(
                f"Atom count/order/species changed in {path}"
            )

        buf.append(frame["pos"][idx])
        n_frames += 1

        if len(buf) == lag_frames + 1:
            vectors.append(buf[-1] - buf[0])

    disp = (
        np.concatenate(vectors, axis=0)
        if vectors
        else np.empty((0, 3), dtype=float)
    )

    return disp, n_frames, len(idx)


def calculate_statistics(values):
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size < 2:
        raise ValueError("At least two samples are required")

    if not np.isfinite(values).all():
        raise ValueError("Values contain NaN or infinity")

    sd = float(np.std(values, ddof=std_ddof))

    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(
            "Standard deviation must be finite and positive"
        )

    stats = {
        "n_samples": int(values.size),
        "mean": float(np.mean(values)),
        "std": sd,
        "skew": float(
            skew(values, bias=stats_bias)
        ),
        "kurt": float(
            kurtosis(
                values,
                fisher=False,
                bias=stats_bias,
            )
        ),
    }

    if not all(
        np.isfinite(stats[key])
        for key in ("mean", "std", "skew", "kurt")
    ):
        raise ValueError("Non-finite statistics were obtained")

    return stats


def signed_log_transform(values, sigma, k, strength):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")

    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(
            "log_strength must be finite and positive"
        )

    values = np.asarray(values, dtype=float)
    effective_k = strength * k

    with np.errstate(over="raise", invalid="raise"):
        transformed = (
            np.sign(values)
            * np.log1p(effective_k * np.abs(values) / sigma)
        )

    if not np.isfinite(transformed).all():
        raise ValueError(
            "Transformed values contain NaN or infinity"
        )

    return transformed


def inverse_signed_log_transform(values, sigma, k, strength):
    values = np.asarray(values, dtype=float)
    effective_k = strength * k

    with np.errstate(over="raise", invalid="raise"):
        restored = (
            np.sign(values)
            * (sigma / effective_k)
            * np.expm1(np.abs(values))
        )

    if not np.isfinite(restored).all():
        raise ValueError(
            "Inverse-transformed values contain NaN or infinity"
        )

    return restored


def make_legend_label(stats, unit=None):
    suffix = f" {unit}" if unit else ""

    return (
        f"mean = {stats['mean']:.4f}{suffix}\n"
        f"std = {stats['std']:.4f}{suffix}\n"
        f"skew = {stats['skew']:.4f}\n"
        f"kurt = {stats['kurt']:.4f}"
    )


def draw_histogram(
    ax,
    values,
    stats,
    title,
    xlabel,
    color,
    unit=None,
):
    limit = float(np.max(np.abs(values)))

    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Invalid histogram range")

    edges = np.linspace(-limit, limit, bins + 1)

    ax.hist(
        values,
        bins=edges,
        density=True,
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.35,
        label=make_legend_label(stats, unit=unit),
    )

    ax.set_xlim(-1.02 * limit, 1.02 * limit)
    ax.set_ylim(bottom=0)

    ax.set_title(
        title,
        fontsize=font_size + 1,
        pad=5,
    )
    ax.set_xlabel(
        xlabel,
        fontsize=font_size,
        labelpad=3,
    )
    ax.set_ylabel(
        "Probability density",
        fontsize=font_size,
        labelpad=4,
    )

    ax.tick_params(
        axis="both",
        labelsize=font_size - 1,
        pad=3,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )
    ax.set_axisbelow(True)

    ax.legend(
        loc=legend_location,
        fontsize=legend_font_size,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
        handlelength=0.9,
        handletextpad=0.5,
        borderpad=0.5,
        labelspacing=0.3,
    )


def make_output_base(strength):
    strength_tag = (
        f"{strength:.8g}"
        .replace(".", "p")
        .replace("-", "m")
    )

    return Path(
        f"{out_stem}_{species_target}_x_signedlog_"
        f"lambda{strength_tag}_lag{lag_frames}f"
    )


def plot_before_after(
    x_before,
    x_after,
    stats_before,
    stats_after,
    strength,
    output_base,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size,
        sharex=False,
        sharey=False,
    )

    draw_histogram(
        ax=axes[0],
        values=x_before,
        stats=stats_before,
        title="Before transformation",
        xlabel=r"$\Delta x$ ($\mathrm{\AA}$)",
        color=before_color,
        unit="Å",
    )

    draw_histogram(
        ax=axes[1],
        values=x_after,
        stats=stats_after,
        title=(
            "After signed-log transformation "
            rf"($\lambda={strength:g}$)"
        ),
        xlabel=r"Transformed $\Delta x$ (dimensionless)",
        color=after_color,
    )

    # No suptitle or reserved top area.
    fig.tight_layout(
        pad=layout_pad,
        h_pad=subplot_h_pad,
    )

    png_path = output_base.with_suffix(".png")
    fig.savefig(
        png_path,
        dpi=figure_dpi,
        bbox_inches="tight",
        pad_inches=save_pad_inches,
        facecolor="white",
    )
    print(f"Saved: {png_path}")

    # if save_pdf:
    #     pdf_path = output_base.with_suffix(".pdf")
    #     fig.savefig(
    #         pdf_path,
    #         bbox_inches="tight",
    #         pad_inches=save_pad_inches,
    #         facecolor="white",
    #     )
    #     print(f"Saved: {pdf_path}")

    plt.close(fig)


def save_statistics(
    output_base,
    stats_before,
    stats_after,
    sigma,
    k,
    strength,
    lag_ps,
    inverse_max_error,
):
    csv_path = output_base.with_suffix(".csv")

    fieldnames = [
        "stage",
        "species",
        "component",
        "lag_frames",
        "lag_ps",
        "n_samples",
        "mean",
        "std",
        "skew",
        "kurt",
        "kurtosis_definition",
        "std_ddof",
        "stats_bias",
        "transform_sigma",
        "transform_k",
        "log_strength",
        "effective_k",
        "inverse_max_abs_error",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for stage, stats in (
            ("before", stats_before),
            ("after", stats_after),
        ):
            writer.writerow({
                "stage": stage,
                "species": species_target,
                "component": "x",
                "lag_frames": lag_frames,
                "lag_ps": lag_ps,
                **stats,
                "kurtosis_definition": "Pearson (normal=3)",
                "std_ddof": std_ddof,
                "stats_bias": stats_bias,
                "transform_sigma": sigma,
                "transform_k": k,
                "log_strength": strength,
                "effective_k": strength * k,
                "inverse_max_abs_error": inverse_max_error,
            })

    print(f"Saved: {csv_path}")


def validate_settings():
    if not input_paths:
        raise ValueError("input_paths is empty")

    if not isinstance(lag_frames, int) or lag_frames < 1:
        raise ValueError(
            "lag_frames must be a positive integer"
        )

    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")

    if not np.isfinite(dt_frame_ps) or dt_frame_ps <= 0:
        raise ValueError(
            "dt_frame_ps must be finite and positive"
        )

    if std_ddof not in (0, 1):
        raise ValueError("std_ddof must be 0 or 1")

    strengths = (
        strength_candidates
        if compare_strengths
        else [log_strength]
    )

    if not strengths:
        raise ValueError("No transformation strengths specified")

    if any(
        not np.isfinite(s) or s <= 0
        for s in strengths
    ):
        raise ValueError(
            "All transformation strengths must be "
            "finite and positive"
        )

    return list(dict.fromkeys(float(s) for s in strengths))


def main():
    strengths = validate_settings()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    all_vectors = []
    expected_vectors = 0

    for path in input_paths:
        disp, frames, n_target = (
            compute_species_displacement_vectors_unwrap(
                path,
                species_target,
                lag_frames,
            )
        )

        all_vectors.append(disp)
        expected_vectors += (
            n_target * max(frames - lag_frames, 0)
        )

        print(
            f"[{path}] frames={frames}, "
            f"{species_target} count={n_target}, "
            f"vectors={len(disp):,}"
        )

    disp_vec = np.concatenate(all_vectors, axis=0)

    if len(disp_vec) < 2:
        raise ValueError(
            "At least two displacement samples are required"
        )

    if not np.isfinite(disp_vec).all():
        raise ValueError(
            "Displacements contain NaN or infinity"
        )

    if len(disp_vec) != expected_vectors:
        raise ValueError(
            "Unexpected displacement sample count"
        )

    # x component only.
    x_before = disp_vec[:, 0]
    stats_before = calculate_statistics(x_before)

    sigma_x = stats_before["std"]
    k_x = stats_before["kurt"]
    lag_ps = lag_frames * dt_frame_ps

    print(f"\nSamples: {len(x_before):,}")
    print(f"Lag: {lag_frames} frames = {lag_ps:g} ps")

    print("\n[Before transformation]")
    for key in ("mean", "std", "skew", "kurt"):
        print(f"{key} = {stats_before[key]:.8g}")

    for strength in strengths:
        x_after = signed_log_transform(
            x_before,
            sigma=sigma_x,
            k=k_x,
            strength=strength,
        )

        stats_after = calculate_statistics(x_after)

        restored = inverse_signed_log_transform(
            x_after,
            sigma=sigma_x,
            k=k_x,
            strength=strength,
        )

        inverse_max_error = float(
            np.max(np.abs(restored - x_before))
        )

        if not np.allclose(
            restored,
            x_before,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "Inverse-transform consistency check failed "
                f"for lambda={strength:g}"
            )

        print(
            "\n[After transformation: "
            f"lambda={strength:g}]"
        )
        print(f"sigma_x = {sigma_x:.8g} Å")
        print(f"K_x = {k_x:.8g}")
        print(f"effective_k = {strength * k_x:.8g}")

        for key in ("mean", "std", "skew", "kurt"):
            print(f"{key} = {stats_after[key]:.8g}")

        print(
            "Inverse-transform max absolute error: "
            f"{inverse_max_error:.6e} Å"
        )

        output_base = make_output_base(strength)
        output_base.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_before_after(
            x_before=x_before,
            x_after=x_after,
            stats_before=stats_before,
            stats_after=stats_after,
            strength=strength,
            output_base=output_base,
        )

        if save_statistics_csv:
            save_statistics(
                output_base=output_base,
                stats_before=stats_before,
                stats_after=stats_after,
                sigma=sigma_x,
                k=k_x,
                strength=strength,
                lag_ps=lag_ps,
                inverse_max_error=inverse_max_error,
            )


if __name__ == "__main__":
    main()