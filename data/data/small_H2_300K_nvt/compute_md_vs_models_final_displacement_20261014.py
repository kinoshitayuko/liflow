# -*- coding: utf-8 -*-
"""Compare reference MD and final displacement samples from two models.
Requires unwrapped XYZ coordinates and extracted prior-vs-final CSV files.
Only final_dx/dy/dz are used; initial-prior columns are never read.
"""
import re
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- user parameters --------
md_xyz_path = Path("output.xyz")
md_target_species = "He"
current_model_csv_paths = [
    Path("../../../scripts/traj_output_TiO2SiO2_H2/"
         "20261019_flow_trace_propose_300K_seed42_prior_vs_final_atoms31.csv"),
    
    # Path("../../../scripts/traj_output_TiO2SiO2_H2/"
    #      "新しいモデルの実際の抽出CSV名.csv"),
]
# REQUIRED: add the actual extracted CSV from the newly trained model.
new_model_csv_paths = [
    # Path("../../../scripts/traj_output_TiO2SiO2_H2/ACTUAL_NEW_MODEL_EXTRACTED_CSV.csv"),
]
current_model_atom_index = 31
new_model_atom_index = 31
lag_frames = 50
time_step_ps = 0.01
dump_stride = 1
dt_frame_ps = time_step_ps * dump_stride
bins = 80
x_limits = (-10.0, 10.0)  # Display limits only; None shows the full range.
chunksize = 100_000
output_png = Path("20261019_MD_vs_current_vs_new_model_final_xyz_hist_finetune.png")
md_label = "Reference MD (all He)"
current_model_label = "Current-prior model (CSV atom 31)"
new_model_label = "New-prior model (CSV atom 31)"
md_color = "#7166ff"
current_model_color = "#ff6666"
new_model_color = "#2ca866"
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
            if not f.readline():
                raise ValueError(f"Incomplete XYZ frame in {path}: missing comment")
            species = []
            positions = np.empty((natoms, 3), dtype=float)
            for i in range(natoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Bad atom line in {path}: need species x y z")
                species.append(parts[0])
                positions[i] = [float(v) for v in parts[1:4]]
            yield species, positions


def read_md_displacements(path, target_species, lag):
    """Pool individual selected-atom displacements; do not average atoms."""
    if not path.is_file():
        raise FileNotFoundError(f"MD XYZ not found: {path}")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    buffer = deque(maxlen=lag + 1)
    displacements = []
    reference_species = None
    atom_indices = None
    n_frames = 0
    for species, positions in iter_xyz_frames(path):
        if reference_species is None:
            reference_species = species.copy()
            atom_indices = [i for i, symbol in enumerate(species) if symbol == target_species]
            if not atom_indices:
                raise ValueError(f"No atoms labeled {target_species!r} found in {path}")
            print(f"[MD selection] species={target_species}, atom_indices={atom_indices}, n_atoms={len(atom_indices)}")
        elif species != reference_species:
            raise ValueError(f"Atom count/species ordering changed across frames in {path}")
        coordinates = positions[atom_indices].copy()
        if not np.isfinite(coordinates).all():
            raise ValueError(f"Non-finite MD coordinates at frame {n_frames}")
        buffer.append(coordinates)
        n_frames += 1
        if len(buffer) == lag + 1:
            displacements.append(buffer[-1] - buffer[0])
    if not displacements:
        raise ValueError(f"No MD displacement samples: frames={n_frames}, lag_frames={lag}")
    values = np.asarray(displacements, dtype=float).reshape(-1, 3)
    if not np.isfinite(values).all():
        raise ValueError("MD displacement samples contain NaN or infinity")
    return values, n_frames, atom_indices


def read_final_samples(paths, atom_index, chunk_size):
    required = ["atom_index", "final_dx", "final_dy", "final_dz"]
    parts = []
    seen_paths = set()
    for path in paths:
        path = Path(path)
        if path.resolve() in seen_paths:
            raise ValueError(f"The same CSV was specified more than once: {path}")
        seen_paths.add(path.resolve())
        if not path.is_file():
            raise FileNotFoundError(f"Final-displacement CSV not found: {path}")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing = set(required) - set(columns)
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {sorted(missing)}. "
                "Use the extracted prior-vs-final CSV, not the raw flow-trace CSV."
            )
        n_read = n_selected = 0
        for chunk in pd.read_csv(path, usecols=required, chunksize=chunk_size):
            n_read += len(chunk)
            selected = chunk.loc[chunk["atom_index"].eq(atom_index), required[1:]]
            if not selected.empty:
                values = selected.to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Final samples contain NaN or infinity in {path}")
                parts.append(values)
                n_selected += len(values)
        print(f"[CSV] {path}: rows={n_read:,}, atom {atom_index} samples={n_selected:,}")
        if n_selected == 0:
            print(f"Warning: no rows for atom {atom_index} in this CSV.")
    if not parts:
        raise ValueError(f"No final samples for atom_index={atom_index}")
    return np.concatenate(parts, axis=0)


def print_summary(label, vectors):
    means = np.mean(vectors, axis=0)
    stds = np.std(vectors, axis=0, ddof=0)
    rms = np.sqrt(np.mean(vectors ** 2, axis=0))
    print(f"\n[{label}] vector samples={len(vectors):,}; samples per axis={len(vectors):,}")
    for i, axis in enumerate("xyz"):
        print(f"  {axis}: mean={means[i]:.6f} Å, std={stds[i]:.6f} Å, RMS={rms[i]:.6f} Å")


# def plot_histograms(md_vectors, current_vectors, new_vectors, lag_ps):
#     groups = [
#         (md_vectors, md_label, md_color, 0.58),
#         (current_vectors, current_model_label, current_model_color, 0.50),
#         (new_vectors, new_model_label, new_model_color, 0.50),
#     ]
#     observed_limit = float(max(np.max(np.abs(v)) for v, _, _, _ in groups))
#     if x_limits is not None:
#         observed_limit = max(observed_limit, abs(x_limits[0]), abs(x_limits[1]))
#     if observed_limit == 0:
#         observed_limit = 1.0
#     edges = np.linspace(-np.nextafter(observed_limit, np.inf),
#                         np.nextafter(observed_limit, np.inf), bins + 1)
#     fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), sharex=True, sharey=True)
#     for i, axis in enumerate("xyz"):
#         ax = axes[i]
#         for vectors, label, color, alpha in groups:
#             ax.hist(vectors[:, i], bins=edges, density=True,
#                     color=color, alpha=alpha, label=label)
#         ax.set_title(f"Component: Δ{axis}", fontsize=18)
#         ax.set_xlabel(f"Δ{axis} (Å)", fontsize=16)
#         ax.tick_params(labelsize=12)
#         ax.grid(True, ls="--", alpha=0.35)
#         ax.set_axisbelow(True)
#         if x_limits is not None:
#             ax.set_xlim(*x_limits)
#     axes[0].set_ylabel("Probability density", fontsize=16)
#     fig.suptitle(
#         f"{md_target_species}: MD displacement ({lag_ps:g} ps) vs final model displacements\n"
#         f"Samples per axis: MD={len(md_vectors):,}, current={len(current_vectors):,}, new={len(new_vectors):,}",
#         fontsize=13,
#     )
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
#                ncol=3, fontsize=11, frameon=False)
#     fig.tight_layout(rect=(0, 0.09, 1, 1), pad=1.0)
#     output_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_png, dpi=180, bbox_inches="tight")
#     plt.close(fig)
#     print(f"\nSaved: {output_png}")
#     if x_limits is not None:
#         lo, hi = x_limits
#         for vectors, label, _, _ in groups:
#             if ((vectors < lo) | (vectors > hi)).any():
#                 print(f"Note [{label}]: some samples are outside display limits; all samples were used for normalization.")

def plot_histograms(md_vectors, current_vectors, new_vectors, lag_ps):
    groups = [
        (md_vectors, md_label, md_color, 0.58),
        (current_vectors, current_model_label, current_model_color, 0.50),
    ]

    # 新しいモデルのデータがある場合だけ追加する
    if new_vectors is not None:
        groups.append(
            (new_vectors, new_model_label, new_model_color, 0.50)
        )

    # 全データ・全軸に共通のbin境界を設定する
    observed_limit = float(
        max(
            np.max(np.abs(vectors))
            for vectors, _, _, _ in groups
        )
    )

    if x_limits is not None:
        observed_limit = max(
            observed_limit,
            abs(x_limits[0]),
            abs(x_limits[1]),
        )

    if observed_limit == 0:
        observed_limit = 1.0

    limit = float(np.nextafter(observed_limit, np.inf))
    edges = np.linspace(-limit, limit, bins + 1)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18.0, 5.2),
        sharex=True,
        sharey=True,
    )

    for i, axis in enumerate("xyz"):
        ax = axes[i]

        for vectors, label, color, alpha in groups:
            ax.hist(
                vectors[:, i],
                bins=edges,
                density=True,
                color=color,
                alpha=alpha,
                label=label,
            )

        ax.set_title(f"Component: Δ{axis}", fontsize=18)
        ax.set_xlabel(f"Δ{axis} (Å)", fontsize=16)
        ax.tick_params(labelsize=12)
        ax.grid(True, ls="--", alpha=0.35)
        ax.set_axisbelow(True)

        if x_limits is not None:
            ax.set_xlim(*x_limits)

    axes[0].set_ylabel("Probability density", fontsize=16)

    # 標本数の表示も、比較するモデル数に合わせる
    sample_text = (
        f"Samples per axis: MD={len(md_vectors):,}, "
        f"current={len(current_vectors):,}"
    )

    if new_vectors is not None:
        sample_text += f", new={len(new_vectors):,}"

    fig.suptitle(
        f"{md_target_species}: MD displacement ({lag_ps:g} ps) "
        f"vs final model displacements\n"
        f"{sample_text}",
        fontsize=13,
    )

    # 共通凡例：2種類なら2列、3種類なら3列
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(groups),
        fontsize=11,
        frameon=False,
    )

    fig.tight_layout(
        rect=(0, 0.09, 1, 1),
        pad=1.0,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {output_png}")

    if x_limits is not None:
        lo, hi = x_limits

        for vectors, label, _, _ in groups:
            if ((vectors < lo) | (vectors > hi)).any():
                print(
                    f"Note [{label}]: some samples are outside "
                    "display limits; all samples were used "
                    "for normalization."
                )





# def main():
#     if not current_model_csv_paths:
#         raise ValueError("Set current_model_csv_paths to the current model's extracted CSV files.")
#     if not new_model_csv_paths:
#         raise ValueError("Set new_model_csv_paths to the newly trained model's actual extracted CSV files.")
#     if current_model_atom_index < 0 or new_model_atom_index < 0:
#         raise ValueError("CSV atom indices must be >= 0")
#     if lag_frames < 1 or bins < 1 or chunksize < 1 or dt_frame_ps <= 0:
#         raise ValueError("lag_frames, bins, chunksize and dt_frame_ps must be positive")
#     if x_limits is not None and (len(x_limits) != 2 or x_limits[0] >= x_limits[1]):
#         raise ValueError("x_limits must be None or an increasing pair")
#     current_paths = {Path(p).resolve() for p in current_model_csv_paths}
#     new_paths = {Path(p).resolve() for p in new_model_csv_paths}
#     if current_paths & new_paths:
#         raise ValueError("The same CSV is in both model groups. Check the model CSV paths.")
#     for path in [md_xyz_path, *current_model_csv_paths, *new_model_csv_paths]:
#         if not Path(path).is_file():
#             raise FileNotFoundError(f"Input file not found: {path}")

#     md_vectors, n_frames, indices = read_md_displacements(md_xyz_path, md_target_species, lag_frames)
#     print("\nLoading current model final displacements...")
#     current_vectors = read_final_samples(current_model_csv_paths, current_model_atom_index, chunksize)
#     print("\nLoading new model final displacements...")
#     new_vectors = read_final_samples(new_model_csv_paths, new_model_atom_index, chunksize)
#     lag_ps = lag_frames * dt_frame_ps
#     print(f"\n[MD] frames={n_frames:,}, atom_indices={indices}")
#     print(f"[MD samples] {n_frames-lag_frames:,} time origins × {len(indices)} atoms = {len(md_vectors):,}")
#     print(f"[Lag] {lag_frames} frames × {dt_frame_ps:g} ps/frame = {lag_ps:g} ps")
#     print("[CSV columns] final_dx, final_dy, final_dz only; initial prior components are unused.")
#     print("[Comparison] pooled MD He atoms vs each model's selected CSV atom; not a matched-atom comparison.")
#     print_summary(md_label, md_vectors)
#     print_summary(current_model_label, current_vectors)
#     print_summary(new_model_label, new_vectors)
#     for label, values in [(current_model_label, current_vectors), (new_model_label, new_vectors)]:
#         if len(values) < 100:
#             print(f"Note [{label}]: few samples; histogram shape may be noisy.")
#     plot_histograms(md_vectors, current_vectors, new_vectors, lag_ps)



def main():
    if not current_model_csv_paths:
        raise ValueError(
            "Set current_model_csv_paths to the current model's "
            "extracted CSV files."
        )

    # 空リストなら、新しいモデルは比較しない
    use_new_model = bool(new_model_csv_paths)

    if current_model_atom_index < 0:
        raise ValueError("current_model_atom_index must be >= 0")

    if use_new_model and new_model_atom_index < 0:
        raise ValueError("new_model_atom_index must be >= 0")

    if (
        lag_frames < 1
        or bins < 1
        or chunksize < 1
        or dt_frame_ps <= 0
    ):
        raise ValueError(
            "lag_frames, bins, chunksize and dt_frame_ps "
            "must be positive"
        )

    if x_limits is not None and (
        len(x_limits) != 2 or x_limits[0] >= x_limits[1]
    ):
        raise ValueError(
            "x_limits must be None or an increasing pair"
        )

    # 同じCSVを新旧両方に指定していないか確認する
    current_paths = {
        Path(p).resolve() for p in current_model_csv_paths
    }

    new_paths = {
        Path(p).resolve() for p in new_model_csv_paths
    }

    if current_paths & new_paths:
        raise ValueError(
            "The same CSV is in both model groups. "
            "Check the model CSV paths."
        )

    # 指定したファイルが存在しない場合は、エラーにする
    for path in [
        md_xyz_path,
        *current_model_csv_paths,
        *new_model_csv_paths,
    ]:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

    md_vectors, n_frames, indices = read_md_displacements(
        md_xyz_path,
        md_target_species,
        lag_frames,
    )

    print("\nLoading current model final displacements...")

    current_vectors = read_final_samples(
        current_model_csv_paths,
        current_model_atom_index,
        chunksize,
    )

    new_vectors = None

    if use_new_model:
        print("\nLoading new model final displacements...")

        new_vectors = read_final_samples(
            new_model_csv_paths,
            new_model_atom_index,
            chunksize,
        )

        print("\n[Mode] MD + current model + new model")

    else:
        print("\n[Mode] MD + current model")
        print(
            "[New model] Skipped: new_model_csv_paths is empty."
        )

    lag_ps = lag_frames * dt_frame_ps

    print(
        f"\n[MD] frames={n_frames:,}, "
        f"atom_indices={indices}"
    )

    print(
        f"[MD samples] "
        f"{n_frames - lag_frames:,} time origins "
        f"× {len(indices)} atoms "
        f"= {len(md_vectors):,}"
    )

    print(
        f"[Lag] {lag_frames} frames "
        f"× {dt_frame_ps:g} ps/frame "
        f"= {lag_ps:g} ps"
    )

    print(
        "[CSV columns] final_dx, final_dy, final_dz only; "
        "initial prior components are unused."
    )

    print_summary(md_label, md_vectors)
    print_summary(current_model_label, current_vectors)

    model_groups = [
        (current_model_label, current_vectors),
    ]

    if use_new_model:
        print_summary(new_model_label, new_vectors)
        model_groups.append(
            (new_model_label, new_vectors)
        )

    for label, values in model_groups:
        if len(values) < 100:
            print(
                f"Note [{label}]: few samples; "
                "histogram shape may be noisy."
            )

    plot_histograms(
        md_vectors,
        current_vectors,
        new_vectors,
        lag_ps,
    )



if __name__ == "__main__":
    main()
