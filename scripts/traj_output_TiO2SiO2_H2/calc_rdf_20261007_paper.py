"""RDF visualization script for LGPS-type and TiO2SiO2-H2 systems."""

import argparse
import os
from pathlib import Path
import ase.io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ═════════════════════════════════════════════
# フォントサイズの一括設定エリア
# ═════════════════════════════════════════════
BASE_FONT_SIZE = 18  # 全体の基準フォントサイズ

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": BASE_FONT_SIZE,  # 全体基本サイズ
        "axes.titlesize": BASE_FONT_SIZE + 1,  # タイトル
        "axes.labelsize": BASE_FONT_SIZE,  # 軸ラベル
        "xtick.labelsize": BASE_FONT_SIZE - 2,  # X軸目盛り
        "ytick.labelsize": BASE_FONT_SIZE - 2,  # Y軸目盛り
        "legend.fontsize": BASE_FONT_SIZE - 2,  # 凡例
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


# ──────────────────────────────────────────────────────────────
# Core RDF functions  (liflow の calculate_average_rdf と同方式)
# ──────────────────────────────────────────────────────────────


def minimum_image(dr: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """直方体セルの最小イメージ規則を適用する。"""
    for i in range(3):
        dr[..., i] -= np.round(dr[..., i] / cell[i]) * cell[i]
    return dr


def calculate_rdf_single_frame(
    positions: np.ndarray,
    cell: np.ndarray,
    rmax: float,
    nbins: int,
) -> tuple:
    """1フレームの全原子ペア RDF を計算する。

    Parameters
    ----------
    positions : (N, 3) Cartesian coords [Å]
    cell      : (3,) orthorhombic cell lengths [Å]
    rmax      : float
    nbins     : int

    Returns
    -------
    r_centers : (nbins,)
    g_r       : (nbins,)
    """
    n = len(positions)
    edges = np.linspace(0.0, rmax, nbins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr_bin = edges[1] - edges[0]

    counts = np.zeros(nbins, dtype=np.float64)

    for i in range(n):
        dr = positions - positions[i]  # (N, 3)
        dr = minimum_image(dr, cell)
        dist = np.linalg.norm(dr, axis=-1)  # (N,)
        dist[i] = rmax + 1.0  # 自分自身を除外

        hist, _ = np.histogram(dist, bins=edges)
        counts += hist

    # 規格化: g(r) = counts / (N * rho * 4*pi*r^2 * dr)
    volume = cell[0] * cell[1] * cell[2]
    rho = n / volume
    shell_vol = 4.0 * np.pi * r_centers**2 * dr_bin
    shell_vol = np.where(shell_vol > 0, shell_vol, 1e-30)
    g_r = counts / (n * rho * shell_vol)

    return r_centers, g_r


def calculate_average_rdf(
    traj: np.ndarray,
    cell: np.ndarray,
    rmax: float = 10.0,
    nbins: int = 100,
) -> tuple:
    """軌跡全体で時間平均した RDF を計算する。

    Parameters
    ----------
    traj : (T, N, 3)
    cell : (3,) orthorhombic cell lengths [Å]
    """
    g_sum = np.zeros(nbins)
    r_centers = None

    for frame in traj:
        r_centers, g_r = calculate_rdf_single_frame(frame, cell, rmax, nbins)
        g_sum += g_r

    return r_centers, g_sum / len(traj)


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────


def load_xyz_trajectory(
    xyz_path: str,
    skip_frames: int = 5,
    stride: int = 1,
    exclude_numbers: list = None,
    cell_override: np.ndarray = None,
    tail_frames: int = 0,
) -> tuple:
    """XYZ ファイルから軌跡を読み込む。"""
    atoms_list = ase.io.read(xyz_path, index=":")
    print(f"  {len(atoms_list)} frames loaded from {Path(xyz_path).name}")

    if tail_frames > 0:
        atoms_list = atoms_list[:-tail_frames]
        print(f"  Dropped last {tail_frames} frames (tail)")

    atoms_list = atoms_list[skip_frames::stride]
    print(
        f"  Using {len(atoms_list)} frames  (skip={skip_frames}, stride={stride})"
    )

    for atoms in atoms_list:
        if cell_override is not None:
            atoms.set_cell(cell_override)
            atoms.set_pbc([True, True, True])
        elif not np.any(atoms.get_cell()):
            raise ValueError("セル情報がなく --lattice も未指定です。")
        else:
            atoms.set_pbc([True, True, True])
        atoms.wrap()

    if exclude_numbers is not None and len(exclude_numbers) > 0:
        ref_numbers = atoms_list[0].get_atomic_numbers()
        mask = np.ones(len(ref_numbers), dtype=bool)
        for z in exclude_numbers:
            mask &= ref_numbers != z
        print(
            f"  Excluding atomic numbers {exclude_numbers}: {mask.sum()} / {len(mask)} atoms used"
        )
        traj = np.array([a.get_positions()[mask] for a in atoms_list])
    else:
        traj = np.array([a.get_positions() for a in atoms_list])

    cell_matrix = atoms_list[-1].get_cell().array
    cell_diag = np.array(
        [cell_matrix[0, 0], cell_matrix[1, 1], cell_matrix[2, 2]]
    )

    if np.all(cell_diag > 0):
        return traj, cell_diag
    else:
        return traj, None


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────


def plot_rdf_comparison(
    r_ref,
    g_ref,
    r_infer,
    g_infer,
    rdf_mae: float,
    output_path: str,
    label_ref: str = "Dataset",
    label_infer: str = "Inference",
    title: str = "RDF Comparison",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))

    ax.plot(
        r_ref, g_ref, color="#1f77b4", lw=2.0, label=label_ref, alpha=0.9
    )
    ax.plot(
        r_infer,
        g_infer,
        color="#d62728",
        lw=1.8,
        linestyle="--",
        label=label_infer,
        alpha=0.9,
    )

    ax.set_xlabel(r"$r$ (Å)")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title(f"{title}\nRDF MAE = {rdf_mae:.4f}", pad=10)

    # 凡例を角丸の四角枠（背景白・透過度高め）で表示
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.9,
    )
    legend.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")

    ax.set_xlim(0.0, None)
    ax.set_ylim(0.0, None)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot averaged RDF from reference and inference XYZ trajectories."
    )

    parser.add_argument(
        "--ref-xyz",
        type=str,
        help="Reference XYZ trajectory (e.g., AIMD/MLIP output)",
        default="../../data/data/small_H2_300K_nvt/output.xyz",
    )
    parser.add_argument(
        "--infer-xyz",
        type=str,
        help="Inference XYZ trajectory (e.g., LiFlow output)",
        #default="traj_300K_42_20260805_2_10n.xyz",
        # default="traj_300K_44_20260817_finetune_1_10n.xyz",
        default="traj_300K_42_20261007_1.xyz",
    )
    parser.add_argument(
        "--lattice",
        type=float,
        nargs=3,
        metavar=("A", "B", "C"),
        help="Orthorhombic cell lengths [Å]. XYZ にセル情報が含まれない場合に指定する。",
        default=[13.659831, 13.659831, 13.659831],
    )
    parser.add_argument(
        "--ref-skip-frames",
        type=int,
        default=0,
        help="Reference の最初の N フレームを捨てる",
    )
    parser.add_argument(
        "--infer-skip-frames",
        type=int,
        default=0,
        help="Inference の最初の N フレームを捨てる",
    )
    parser.add_argument(
        "--ref-tail-frames",
        type=int,
        default=0,
        help="Reference の末尾から N フレームを捨てる (default: 0)",
    )
    parser.add_argument(
        "--infer-tail-frames",
        type=int,
        default=0,
        help="Inference の末尾から N フレームを捨てる (default: 0)",
    )
    parser.add_argument(
        "--ref-stride",
        type=int,
        default=50,
        help="Reference の抽出ストライド (default: 50)",
    )
    parser.add_argument(
        "--infer-stride",
        type=int,
        default=1,
        help="Inference の抽出ストライド (default: 1)",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=5.0,
        help="RDF の最大距離 [Å] (default: 5.0)",
    )
    parser.add_argument(
        "--nbins", type=int, default=100, help="RDF のビン数 (default: 100)"
    )
    parser.add_argument(
        "--label-ref", type=str, default="Dataset"
    )
    parser.add_argument(
        "--label-infer", type=str, default="Inference"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="RDF: After Fine-tuning (10ns Rollout)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rdf_comparison_300K_20261007_1ns_after.png",
    )

    args = parser.parse_args()

    # Reference を読み込む
    print("\n[Reference trajectory]")
    traj_ref, cell_from_ref = load_xyz_trajectory(
        args.ref_xyz,
        skip_frames=args.ref_skip_frames,
        stride=args.ref_stride,
        exclude_numbers=[],
        cell_override=np.array(args.lattice) if args.lattice else None,
        tail_frames=args.ref_tail_frames,
    )

    # セル決定
    if args.lattice is not None:
        cell = np.array(args.lattice)
        print(f"  Cell (manual): {cell} Å")
    elif cell_from_ref is not None:
        cell = cell_from_ref
        print(f"  Cell (from XYZ): {cell} Å")
    else:
        raise ValueError(
            "セル情報が XYZ から読み取れませんでした。--lattice A B C で手動指定してください。"
        )

    # Inference を読み込む
    print("\n[Inference trajectory]")
    traj_infer, _ = load_xyz_trajectory(
        args.infer_xyz,
        skip_frames=args.infer_skip_frames,
        stride=args.infer_stride,
        exclude_numbers=[],
        cell_override=cell,
        tail_frames=args.infer_tail_frames,
    )

    # RDF 計算
    print(f"\n[Computing RDF]  rmax={args.rmax} Å, nbins={args.nbins}")
    print("  Computing reference RDF...")
    r_ref, g_ref = calculate_average_rdf(
        traj_ref, cell, rmax=args.rmax, nbins=args.nbins
    )

    print("  Computing inference RDF...")
    r_infer, g_infer = calculate_average_rdf(
        traj_infer, cell, rmax=args.rmax, nbins=args.nbins
    )

    # MAE 計算
    rdf_mae = float(np.mean(np.abs(g_ref - g_infer)))
    print(f"\n  RDF MAE = {rdf_mae:.6f}")

    # プロット
    plot_rdf_comparison(
        r_ref,
        g_ref,
        r_infer,
        g_infer,
        rdf_mae=rdf_mae,
        output_path=args.output,
        label_ref=args.label_ref,
        label_infer=args.label_infer,
        title=args.title,
    )


if __name__ == "__main__":
    main()