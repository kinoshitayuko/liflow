# export LD_LIBRARY_PATH=/home/user/anaconda3/envs/liflow/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=0
# 最初にこれやる↑
#export CUBLAS_WORKSPACE_CONFIG=:4096:8



# -*- coding: utf-8 -*-
"""
Test one-step Propagator generation from reference MD structures.

Atom ordering:
    Use output.xyz ordering for elements, positions and atom indices.

POSCAR:
    Used only for composition checking and the fixed simulation cell.

Disabled:
    Corrector
    Center-of-mass removal
    H2 bond-length fixing
    Retries

Outputs:
    <output_dir>/md_vs_propagator_one_step.csv
    <output_dir>/origin_frame_mapping.csv
    <output_dir>/run_settings.json

    <flow_debug_directory>/he_displacements.csv

MD coordinates must already be unwrapped.
No PBC correction is applied to reference displacement calculations.
"""

import csv
import json
import random
from collections import Counter, deque
from pathlib import Path

import ase.io
import numpy as np
import torch

from liflow.model.modules import FlowModule
from liflow.utils.inference import FlowSimulator
from liflow.utils.prior import get_prior


# ============================================================
# User parameters
# ============================================================

data_dir = Path("../data/data/small_H2_300K_nvt")

structure_path = data_dir / "TiO2SiO2_init_POSCAR.sid"
element_index_path = data_dir / "element_index.npy"
md_xyz_path = data_dir / "output.xyz"

propagate_ckpt = Path(
    # "ckpt/P_TiO2SiO2_H2_20261106_test_2.ckpt"
    "ckpt/P_TiO2SiO2_H2_20260804_2.ckpt"
)

output_dir = Path(
    "propagator_from_reference_md_20261109_fixed_weighted_2"
)

temperature = 300
seed = 43

flow_steps = 10
solver = "euler"

# Must match the physical duration of one Propagator step.
lag_frames = 50
md_frame_interval_ps = 0.01

# Starting frames: 0, 50, 100, ...
origin_stride = 50

# Maximum number of independent starting structures.
# 1000 structures × 2 H atoms = 2000 generated displacement vectors.
max_origins = 2000

target_species = "He"
expected_target_atom_count = 2

scale_Li_index = 1
scale_frame_index = 0

# Cell matching tolerance when the XYZ includes cell information.
cell_rtol = 1.0e-5
cell_atol = 1.0e-5

# ============================================================


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def validate_parameters():
    required_paths = {
        "POSCAR": structure_path,
        "Element index": element_index_path,
        "Reference MD": md_xyz_path,
        "Propagator checkpoint": propagate_ckpt,
    }

    for label, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path.resolve()}"
            )

    for label, value in (
        ("lag_frames", lag_frames),
        ("origin_stride", origin_stride),
        ("max_origins", max_origins),
        ("flow_steps", flow_steps),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"{label} must be an integer >= 1"
            )

    if (
        not np.isfinite(md_frame_interval_ps)
        or md_frame_interval_ps <= 0
    ):
        raise ValueError(
            "md_frame_interval_ps must be finite and positive"
        )

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "temperature must be finite and positive"
        )

    if solver not in ("euler", "heun"):
        raise ValueError(
            f"Unsupported solver: {solver}"
        )


def check_md_cell(frame, reference_cell, frame_index):
    """Use the POSCAR cell; reject conflicting XYZ cell information."""
    if frame.cell.rank == 0:
        return

    if frame.cell.rank != 3:
        raise ValueError(
            "MD frame contains an incomplete cell: "
            f"frame={frame_index}, cell_rank={frame.cell.rank}"
        )

    if not np.allclose(
        frame.cell.array,
        reference_cell,
        rtol=cell_rtol,
        atol=cell_atol,
    ):
        raise ValueError(
            "MD cell differs from the fixed POSCAR cell: "
            f"frame={frame_index}\n"
            f"POSCAR cell:\n{reference_cell}\n"
            f"MD cell:\n{frame.cell.array}"
        )


def print_summary(label, vectors):
    vectors = np.asarray(vectors, dtype=float)

    means = vectors.mean(axis=0)
    stds = vectors.std(axis=0)
    rms = np.sqrt(np.mean(vectors ** 2, axis=0))
    magnitudes = np.linalg.norm(vectors, axis=1)

    print(f"\n[{label}] vectors={len(vectors):,}")

    for axis_index, axis in enumerate("xyz"):
        print(
            f"  {axis}: "
            f"mean={means[axis_index]:.6f} A, "
            f"std={stds[axis_index]:.6f} A, "
            f"RMS={rms[axis_index]:.6f} A"
        )

    print(
        "  P(|displacement| > 4 A)="
        f"{np.mean(magnitudes > 4.0):.6f}"
    )


def main():
    validate_parameters()
    set_seed(seed)

    torch.set_float32_matmul_precision("high")

    # POSCAR provides the fixed simulation cell.
    template = ase.io.read(structure_path)
    reference_cell = template.cell.array.copy()

    if (
        template.cell.rank != 3
        or not np.isfinite(reference_cell).all()
    ):
        raise ValueError(
            "POSCAR must provide a finite, full-rank simulation cell"
        )

    # All atom labels and indices follow MD ordering.
    md_first = ase.io.read(md_xyz_path, index=0)

    poscar_composition = Counter(
        template.get_chemical_symbols()
    )
    md_composition = Counter(
        md_first.get_chemical_symbols()
    )

    if poscar_composition != md_composition:
        raise ValueError(
            "MD and POSCAR compositions differ:\n"
            f"POSCAR={dict(poscar_composition)}\n"
            f"MD={dict(md_composition)}"
        )

    atomic_numbers = md_first.get_atomic_numbers().copy()
    md_symbols = md_first.get_chemical_symbols()

    target_indices = np.array(
        [
            index
            for index, symbol in enumerate(md_symbols)
            if symbol == target_species
        ],
        dtype=int,
    )

    if len(target_indices) != expected_target_atom_count:
        raise ValueError(
            f"Expected {expected_target_atom_count} "
            f"{target_species}-labelled atoms, "
            f"found {len(target_indices)}: "
            f"{target_indices.tolist()}"
        )

    check_md_cell(
        md_first,
        reference_cell,
        frame_index=0,
    )

    element_idx = np.load(element_index_path)

    print("[Input]")
    print(f"  POSCAR: {structure_path.resolve()}")
    print(f"  MD: {md_xyz_path.resolve()}")
    print(f"  Checkpoint: {propagate_ckpt.resolve()}")

    print("\n[Structure]")
    print(f"  Number of atoms: {len(atomic_numbers)}")
    print(f"  Composition: {dict(md_composition)}")
    print("  Atom ordering: output.xyz")
    print(
        f"  Diffusing H labels: {target_species}, "
        f"indices={target_indices.tolist()} (0-based)"
    )

    lag_ps = lag_frames * md_frame_interval_ps

    print("\n[Test settings]")
    print(f"  Temperature: {temperature} K")
    print(f"  MD lag: {lag_frames} frames = {lag_ps:g} ps")
    print(f"  Starting-frame stride: {origin_stride}")
    print(f"  Maximum starting structures: {max_origins}")
    print(f"  Solver: {solver}")
    print(f"  Flow integration steps: {flow_steps}")
    print("  Corrector: OFF")
    print("  COM removal: OFF")
    print("  Bond fixing: OFF")
    print("  Retries: OFF")
    print("  Reference coordinates must be unwrapped.")

    propagate_model = FlowModule.load_from_checkpoint(
        str(propagate_ckpt)
    ).eval()

    cfg_prior = propagate_model.cfg.propagate_prior

    propagate_prior = get_prior(
        cfg_prior.class_name,
        **cfg_prior.params,
        seed=seed,
    )

    simulator = FlowSimulator(
        propagate_model=propagate_model,
        propagate_prior=propagate_prior,
        atomic_numbers=atomic_numbers,
        element_idx=element_idx,
        lattice=reference_cell,
        temp=temperature,
        correct_model=None,
        correct_prior=None,
        pbc=True,
        scale_Li_index=scale_Li_index,
        scale_frame_index=scale_frame_index,
        fix_frame_species=None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        output_dir / "md_vs_propagator_one_step.csv"
    )
    mapping_path = (
        output_dir / "origin_frame_mapping.csv"
    )
    settings_path = (
        output_dir / "run_settings.json"
    )

    # Protect previous results from accidental overwriting.
    for path in (
        comparison_path,
        mapping_path,
        settings_path,
    ):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path.resolve()}\n"
                "Change output_dir or move the previous results."
            )

    settings = {
        "structure_path": str(structure_path.resolve()),
        "md_xyz_path": str(md_xyz_path.resolve()),
        "propagate_ckpt": str(propagate_ckpt.resolve()),
        "element_index_path": str(element_index_path.resolve()),
        "atom_ordering": "output.xyz",
        "target_species": target_species,
        "target_atom_indices": target_indices.tolist(),
        "temperature_K": temperature,
        "seed": seed,
        "lag_frames": lag_frames,
        "md_frame_interval_ps": md_frame_interval_ps,
        "lag_ps": lag_ps,
        "origin_stride": origin_stride,
        "max_origins": max_origins,
        "flow_steps": flow_steps,
        "solver": solver,
        "corrector": False,
        "fix_com": False,
        "bond_fix": False,
        "num_retries": 0,
        "reference_coordinates_assumed_unwrapped": True,
        "cell_A": reference_cell.tolist(),
        "stage_csv": str(
            simulator.displacement_csv_path.resolve()
        ),
        "status": "running",
    }

    with settings_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            settings,
            f,
            indent=2,
            ensure_ascii=False,
        )

    comparison_fields = [
        "origin_id",
        "run_id",
        "start_frame",
        "end_frame",
        "atom_index",
        "md_dx",
        "md_dy",
        "md_dz",
        "pred_dx",
        "pred_dy",
        "pred_dz",
    ]

    mapping_fields = [
        "origin_id",
        "run_id",
        "start_frame",
        "end_frame",
    ]

    buffer = deque(maxlen=lag_frames + 1)
    n_origins = 0
    md_samples = []
    pred_samples = []

    frames = ase.io.iread(
        md_xyz_path,
        index=":",
    )

    try:
        with (
            comparison_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as comparison_file,
            mapping_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as mapping_file,
        ):
            comparison_writer = csv.writer(
                comparison_file
            )
            comparison_writer.writerow(
                comparison_fields
            )

            mapping_writer = csv.writer(
                mapping_file
            )
            mapping_writer.writerow(
                mapping_fields
            )

            for frame_index, frame in enumerate(frames):
                current_numbers = (
                    frame.get_atomic_numbers()
                )

                if not np.array_equal(
                    current_numbers,
                    atomic_numbers,
                ):
                    raise ValueError(
                        "Atom count/order/species changed "
                        "within output.xyz: "
                        f"frame={frame_index}"
                    )

                check_md_cell(
                    frame,
                    reference_cell,
                    frame_index,
                )

                positions = (
                    frame.get_positions().copy()
                )

                if not np.isfinite(positions).all():
                    raise ValueError(
                        "Nonfinite MD coordinates: "
                        f"frame={frame_index}"
                    )

                buffer.append(positions)

                if len(buffer) < lag_frames + 1:
                    continue

                start_frame = (
                    frame_index - lag_frames
                )

                if start_frame % origin_stride != 0:
                    continue

                start_positions = buffer[0].copy()
                reference_end = buffer[-1].copy()

                # Always restart from the reference MD structure.
                traj = simulator.run(
                    positions=start_positions,
                    steps=1,
                    flow_steps=flow_steps,
                    solver=solver,
                    verbose=False,
                    fix_com=False,
                    num_retries=0,
                    corrector_every=1,
                )

                if len(traj) != 2:
                    raise RuntimeError(
                        "One-step Propagator generation failed: "
                        f"origin_id={n_origins}, "
                        f"start_frame={start_frame}"
                    )

                generated_end = np.asarray(
                    traj[1],
                    dtype=float,
                )

                md_disp = (
                    reference_end - start_positions
                )
                pred_disp = (
                    generated_end - start_positions
                )

                if (
                    not np.isfinite(md_disp).all()
                    or not np.isfinite(pred_disp).all()
                ):
                    raise ValueError(
                        "Nonfinite displacement: "
                        f"start_frame={start_frame}"
                    )

                run_id = simulator._csv_run_id

                for atom_index in target_indices:
                    comparison_writer.writerow(
                        [
                            n_origins,
                            run_id,
                            start_frame,
                            frame_index,
                            int(atom_index),
                            *md_disp[atom_index].tolist(),
                            *pred_disp[atom_index].tolist(),
                        ]
                    )

                mapping_writer.writerow(
                    [
                        n_origins,
                        run_id,
                        start_frame,
                        frame_index,
                    ]
                )

                md_samples.append(
                    md_disp[target_indices].copy()
                )
                pred_samples.append(
                    pred_disp[target_indices].copy()
                )

                n_origins += 1

                if n_origins % 50 == 0:
                    comparison_file.flush()
                    mapping_file.flush()

                    print(
                        "[Progress] "
                        f"origins={n_origins}/{max_origins}, "
                        f"last_start_frame={start_frame}"
                    )

                if n_origins >= max_origins:
                    break

    finally:
        frames.close()

    if n_origins == 0:
        raise ValueError(
            "No starting structures were selected. "
            "Check MD length, lag_frames and origin_stride."
        )

    md_vectors = np.asarray(
        md_samples,
        dtype=float,
    ).reshape(-1, 3)

    pred_vectors = np.asarray(
        pred_samples,
        dtype=float,
    ).reshape(-1, 3)

    settings["status"] = "completed"
    settings["completed_origins"] = n_origins
    settings["displacement_vectors_per_dataset"] = (
        len(md_vectors)
    )

    with settings_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            settings,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print_summary(
        "Reference MD at selected starting frames",
        md_vectors,
    )
    print_summary(
        "Propagator from reference MD structures",
        pred_vectors,
    )

    print("\n[Completed]")
    print(f"  Starting structures: {n_origins}")
    print(
        "  Diffusing H indices: "
        f"{target_indices.tolist()}"
    )
    print(
        "  Samples per axis: "
        f"MD={len(md_vectors):,}, "
        f"prediction={len(pred_vectors):,}"
    )

    if n_origins < max_origins:
        print(
            "  Note: reached the end of MD before "
            "max_origins was reached."
        )

    print("\n[Output files]")
    print(
        f"  Comparison CSV: {comparison_path.resolve()}"
    )
    print(
        f"  Frame mapping: {mapping_path.resolve()}"
    )
    print(
        f"  Settings: {settings_path.resolve()}"
    )
    print(
        "  Stage CSV: "
        f"{simulator.displacement_csv_path.resolve()}"
    )


if __name__ == "__main__":
    main()