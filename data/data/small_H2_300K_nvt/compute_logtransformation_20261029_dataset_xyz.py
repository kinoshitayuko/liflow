# -*- coding: utf-8 -*-
"""
XYZから対象原子の変位ベクトルを計算し、
xyz各成分の符号付き対数変換前後を2行×3列で描画する。

前提:
- 座標はすでにunwrappedされている。
- フレーム間で原子数・原子順序・原子種が変わらない。
- 異なる入力ファイルをまたぐ変位は計算しない。
"""

import csv
import re
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew


# =========================================================
# User parameters
# =========================================================
input_paths = ["output.xyz"]
species_target = "He"
# species_target = "Ti"

lag_frames = 50
bins = 80

time_step_ps = 0.01
dump_stride = 1
dt_frame_ps = time_step_ps * dump_stride

out_stem = "positions_300K"

# u_c = sign(delta_c)
#       * log(1 + log_strength * K_c * abs(delta_c) / sigma_c)
log_strength = 0.1

# Trueの場合、strength_candidatesの各値について図を保存する。
compare_strengths = True
strength_candidates = [0.1]
# strength_candidates = [0.05, 0.1, 0.2, 0.5, 1.0]

# Figure settings
figure_size = (15.0, 6.0)
figure_dpi = 300

font_size = 11
legend_font_size = 9
legend_location = "upper right"

before_color = "#1f77b4"
after_color = "#ff7f0e"

layout_pad = 0.6
subplot_h_pad = 1.0
subplot_w_pad = 0.8
save_pad_inches = 0.03

# Statistics
std_ddof = 1
stats_bias = True

# Pearson kurtosis: normal distribution -> 3.
save_pdf = False
save_statistics_csv = False

# Inverse-transform consistency check
inverse_rtol = 1e-10
inverse_atol = 1e-12

COMPONENTS = ("x", "y", "z")
# =========================================================


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
                raise ValueError(
                    f"Invalid atom count in {path}"
                )

            comment = f.readline()
            if not comment:
                raise ValueError(
                    f"Missing XYZ comment line in {path}"
                )

            species = []
            pos = np.empty((natoms, 3), dtype=np.float64)

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
    path,
    species_target,
    lag_frames,
):
    """Input coordinates must already be unwrapped."""
    frames = iter_xyz_frames(path)
    first = next(frames, None)

    if first is None:
        raise ValueError(f"No XYZ frames in {path}")

    species_ref = first["species"]

    idx = [
        i
        for i, species in enumerate(species_ref)
        if species == species_target
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
        else np.empty((0, 3), dtype=np.float64)
    )

    return disp, n_frames, len(idx)


def calculate_statistics(values):
    values = np.asarray(
        values, dtype=np.float64
    ).reshape(-1)

    if values.size < 2:
        raise ValueError(
            "At least two samples are required"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Values contain NaN or infinity"
        )

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
        raise ValueError(
            "Non-finite statistics were obtained"
        )

    return stats


def validate_transform_parameters(sigma, k, strength):
    for name, value in (
        ("sigma", sigma),
        ("k", k),
        ("log_strength", strength),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be finite and positive"
            )

    effective_k = strength * k

    if not np.isfinite(effective_k) or effective_k <= 0:
        raise ValueError(
            "strength * k must be finite and positive"
        )

    return effective_k


def signed_log_transform(values, sigma, k, strength):
    effective_k = validate_transform_parameters(
        sigma, k, strength
    )

    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Input values contain NaN or infinity"
        )

    with np.errstate(
        over="raise",
        invalid="raise",
        divide="raise",
    ):
        transformed = (
            np.sign(values)
            * np.log1p(
                effective_k * np.abs(values) / sigma
            )
        )

    if not np.isfinite(transformed).all():
        raise ValueError(
            "Transformed values contain NaN or infinity"
        )

    return transformed


def inverse_signed_log_transform(
    values,
    sigma,
    k,
    strength,
):
    effective_k = validate_transform_parameters(
        sigma, k, strength
    )

    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Input values contain NaN or infinity"
        )

    with np.errstate(
        over="raise",
        invalid="raise",
        divide="raise",
    ):
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


def make_common_histogram_edges(component_results, stage):
    limit = max(
        float(
            np.max(
                np.abs(component_results[c][stage])
            )
        )
        for c in COMPONENTS
    )

    if not np.isfinite(limit) or limit <= 0:
        raise ValueError(
            f"Invalid histogram range for stage={stage}"
        )

    edges = np.linspace(
        -limit,
        limit,
        bins + 1,
    )

    return edges, limit


def draw_histogram(
    ax,
    values,
    stats,
    title,
    xlabel,
    color,
    edges,
    limit,
    unit=None,
):
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

    ax.set_xlim(
        -1.02 * limit,
        1.02 * limit,
    )
    #ax.set_ylim(bottom=0)

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
        f"{out_stem}_{species_target}_xyz_signedlog_"
        f"lambda{strength_tag}_lag{lag_frames}f"
    )


def plot_before_after(
    component_results,
    strength,
    output_base,
):
    edges_before, limit_before = (
        make_common_histogram_edges(
            component_results,
            stage="before",
        )
    )

    edges_after, limit_after = (
        make_common_histogram_edges(
            component_results,
            stage="after",
        )
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=figure_size,
        sharex="row",
        sharey="row",
        squeeze=False,
    )

    for column, component in enumerate(COMPONENTS):
        result = component_results[component]

        draw_histogram(
            ax=axes[0, column],
            values=result["before"],
            stats=result["stats_before"],
            title=(
                f"Before transformation: "
                f"{component} component"
            ),
            xlabel=(
                rf"$\Delta {component}$ "
                r"($\mathrm{\AA}$)"
            ),
            color=before_color,
            edges=edges_before,
            limit=limit_before,
            unit="Å",
        )

        draw_histogram(
            ax=axes[1, column],
            values=result["after"],
            stats=result["stats_after"],
            title=(
                f"After signed-log: {component} component "
                rf"($\lambda={strength:g}$)"
            ),
            xlabel=(
                rf"Transformed $\Delta {component}$ "
                "(dimensionless)"
            ),
            color=after_color,
            edges=edges_after,
            limit=limit_after,
        )

    # # 上下段それぞれで、3成分の縦軸範囲を統一する。
    # for row in range(2):
    #     ymax = max(
    #         ax.get_ylim()[1]
    #         for ax in axes[row]
    #     )

    #     for ax in axes[row]:
    #         ax.set_ylim(0, ymax)
    
    # 上下段それぞれについて、実際の棒の高さから上限を設定する。
    for row in range(2):
        peak_height = max(
            patch.get_height()
            for ax in axes[row]
            for patch in ax.patches
        )

        # 最大ピークの上に10%の余白を設ける。
        ymax = peak_height * 1.10

        for ax in axes[row]:
            ax.set_ylim(0, ymax)
    

    fig.tight_layout(
        pad=layout_pad,
        h_pad=subplot_h_pad,
        w_pad=subplot_w_pad,
    )

    # ファイル名中の小数点があっても末尾を削らない。
    png_path = Path(f"{output_base}.png")

    fig.savefig(
        png_path,
        dpi=figure_dpi,
        bbox_inches="tight",
        pad_inches=save_pad_inches,
        facecolor="white",
    )
    print(f"Saved: {png_path}")

    if save_pdf:
        pdf_path = Path(f"{output_base}.pdf")

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=save_pad_inches,
            facecolor="white",
        )
        print(f"Saved: {pdf_path}")

    plt.close(fig)


def save_statistics(
    output_base,
    component_results,
    strength,
    lag_ps,
):
    csv_path = Path(f"{output_base}.csv")

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

        for component in COMPONENTS:
            result = component_results[component]

            for stage in ("before", "after"):
                stats = result[f"stats_{stage}"]

                writer.writerow({
                    "stage": stage,
                    "species": species_target,
                    "component": component,
                    "lag_frames": lag_frames,
                    "lag_ps": lag_ps,
                    **stats,
                    "kurtosis_definition":
                        "Pearson (normal=3)",
                    "std_ddof": std_ddof,
                    "stats_bias": stats_bias,
                    "transform_sigma": result["sigma"],
                    "transform_k": result["k"],
                    "log_strength": strength,
                    "effective_k":
                        strength * result["k"],
                    "inverse_max_abs_error":
                        result["inverse_max_error"],
                })

    print(f"Saved: {csv_path}")


def validate_settings():
    if not input_paths:
        raise ValueError("input_paths is empty")

    for path in input_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

    if not isinstance(lag_frames, int) or lag_frames < 1:
        raise ValueError(
            "lag_frames must be a positive integer"
        )

    if not isinstance(bins, int) or bins < 1:
        raise ValueError(
            "bins must be a positive integer"
        )

    if not np.isfinite(dt_frame_ps) or dt_frame_ps <= 0:
        raise ValueError(
            "dt_frame_ps must be finite and positive"
        )

    if std_ddof not in (0, 1):
        raise ValueError(
            "std_ddof must be 0 or 1"
        )

    strengths = (
        strength_candidates
        if compare_strengths
        else [log_strength]
    )

    if not strengths:
        raise ValueError(
            "No transformation strengths specified"
        )

    if any(
        not np.isfinite(s) or s <= 0
        for s in strengths
    ):
        raise ValueError(
            "All transformation strengths must be "
            "finite and positive"
        )

    return list(
        dict.fromkeys(float(s) for s in strengths)
    )


def print_statistics(stats):
    for key in ("mean", "std", "skew", "kurt"):
        print(f"  {key} = {stats[key]:.8g}")


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

    disp_vec = np.concatenate(
        all_vectors,
        axis=0,
    )

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

    lag_ps = lag_frames * dt_frame_ps

    print(f"\nSpecies: {species_target}")
    print(f"Displacement vectors: {len(disp_vec):,}")
    print(f"Samples per component: {len(disp_vec):,}")
    print(
        f"Lag: {lag_frames} frames = {lag_ps:g} ps"
    )
    print("Kurtosis definition: Pearson (normal=3)")

    # 変換前統計はlambdaによらないため、一度だけ計算する。
    before_results = {}

    print("\n[Before transformation]")

    for column, component in enumerate(COMPONENTS):
        values = disp_vec[:, column]

        try:
            stats = calculate_statistics(values)
        except ValueError as exc:
            raise ValueError(
                f"Before statistics failed "
                f"for component={component}: {exc}"
            ) from exc

        before_results[component] = {
            "before": values,
            "stats_before": stats,
            "sigma": stats["std"],
            "k": stats["kurt"],
        }

        print(f"\nComponent: {component}")
        print_statistics(stats)

    for strength in strengths:
        component_results = {}

        print(
            "\n"
            "=================================================="
        )
        print(
            f"[After transformation: lambda={strength:g}]"
        )

        for component in COMPONENTS:
            source = before_results[component]

            values_before = source["before"]
            sigma = source["sigma"]
            k = source["k"]

            values_after = signed_log_transform(
                values_before,
                sigma=sigma,
                k=k,
                strength=strength,
            )

            stats_after = calculate_statistics(
                values_after
            )

            restored = inverse_signed_log_transform(
                values_after,
                sigma=sigma,
                k=k,
                strength=strength,
            )

            inverse_max_error = float(
                np.max(
                    np.abs(restored - values_before)
                )
            )

            if not np.allclose(
                restored,
                values_before,
                rtol=inverse_rtol,
                atol=inverse_atol,
            ):
                raise ValueError(
                    "Inverse-transform consistency check failed "
                    f"for component={component}, "
                    f"lambda={strength:g}"
                )

            component_results[component] = {
                **source,
                "after": values_after,
                "stats_after": stats_after,
                "inverse_max_error": inverse_max_error,
            }

            print(f"\nComponent: {component}")
            print(f"  sigma_{component} = {sigma:.8g} Å")
            print(f"  K_{component} = {k:.8g}")
            print(
                f"  effective_k = {strength * k:.8g}"
            )
            print_statistics(stats_after)

            print(
                "  Inverse-transform max absolute error: "
                f"{inverse_max_error:.6e} Å"
            )

        output_base = make_output_base(strength)

        output_base.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_before_after(
            component_results=component_results,
            strength=strength,
            output_base=output_base,
        )

        if save_statistics_csv:
            save_statistics(
                output_base=output_base,
                component_results=component_results,
                strength=strength,
                lag_ps=lag_ps,
            )


if __name__ == "__main__":
    main()