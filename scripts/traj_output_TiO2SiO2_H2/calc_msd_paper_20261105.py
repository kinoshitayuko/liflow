import os
from pathlib import Path

from ase.io import read
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ═════════════════════════════════════════════
# フォントサイズ
# ═════════════════════════════════════════════
BASE_FONT_SIZE = 18

plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE + 2,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 2,
        "ytick.labelsize": BASE_FONT_SIZE - 2,
        "legend.fontsize": BASE_FONT_SIZE - 2,
        "figure.titlesize": BASE_FONT_SIZE + 4,
    }
)


# ─────────────────────────────────────────────
# MSD計算
# ─────────────────────────────────────────────
def get_atom_indices(traj, species="He"):
    return [
        i
        for i, symbol in enumerate(traj[0].get_chemical_symbols())
        if symbol == species
    ]


def calculate_msd(traj, atom_indices, msd_time_window, msd_time_shift):
    """時間起点をずらしながらMSDを計算して平均する。"""
    n_traj = len(traj)
    traj_ini = 0
    traj_fin = msd_time_window + 1
    n_count = 0
    msd_list = []

    while traj_fin <= n_traj:
        n_count += 1
        traj_tmp = traj[traj_ini:traj_fin]
        initial_positions = traj_tmp[0].get_positions()[atom_indices]

        msd_tmp = [
            np.mean(
                np.sum(
                    (
                        atoms.get_positions()[atom_indices]
                        - initial_positions
                    ) ** 2,
                    axis=1,
                )
            )
            for atoms in traj_tmp
        ]

        msd_list.append(msd_tmp)
        traj_ini += msd_time_shift
        traj_fin += msd_time_shift

    print(f"  Number of windows: {n_count}")
    return np.mean(msd_list, axis=0)


def calculate_msd_from_file(
    traj_file,
    species,
    msd_time_window,
    msd_time_shift,
    timestep_ps,
    traj_start=0,
):
    if not os.path.exists(traj_file):
        print(f"  [WARN] ファイルが見つかりません: {traj_file}")
        return None, None

    print(f"  読み込み中: {traj_file}")
    traj = read(traj_file, index=":")

    traj = traj[traj_start:]
    print(f"  使用フレーム: {traj_start} 〜 {traj_start + len(traj) - 1}")

    # 複数ファイルの一括処理が途中で止まらないように確認
    if len(traj) < msd_time_window + 1:
        print(
            f"  [WARN] フレーム不足のためスキップ: "
            f"{len(traj)} frames "
            f"(必要: {msd_time_window + 1})"
        )
        return None, None

    atom_indices = get_atom_indices(traj, species)
    print(f"  {species} 原子数: {len(atom_indices)}, フレーム数: {len(traj)}")

    if len(atom_indices) == 0:
        print(f"  [WARN] {species} 原子が存在しないためスキップ")
        return None, None

    msd = calculate_msd(
        traj,
        atom_indices,
        msd_time_window,
        msd_time_shift,
    )

    time = np.arange(len(msd)) * timestep_ps
    return time, msd


def calculate_diffusion_coefficient(msd, timestep_ps, fit_start):
    """MSDの線形フィットから自己拡散係数[cm²/s]を計算する。"""
    x_fit_ps = np.arange(len(msd))[fit_start:] * timestep_ps
    y_fit = msd[fit_start:]

    slope, intercept, _, _, _ = stats.linregress(x_fit_ps, y_fit)

    D = slope / 6
    D *= 1.0e-4

    return D, slope, intercept, x_fit_ps


# ═════════════════════════════════════════════
# 設定：計算条件は元コードと同じ
# ═════════════════════════════════════════════
target_temp = 300

# ── LiFlow推論ファイル：ここに追加する ──
liflow_files = [
    # "traj_300K_47_20261105_finetune_10ns_before.xyz",
    # "traj_300K_48_20261105_finetune_10ns_before.xyz",
    # "traj_300K_49_20261105_finetune_10ns_before.xyz",
    # "traj_300K_50_20261105_finetune_10ns_before.xyz",
    # "traj_300K_47_20261105_finetune_10ns_after.xyz",
    # "traj_300K_48_20261105_finetune_10ns_after.xyz",
    # "traj_300K_49_20261105_finetune_10ns_after.xyz",
    # # "traj_300K_50_20261105_finetune_10ns_after.xyz",
    #"traj_300K_42_20261106_logtransformation_1.xyz"
    "traj_300K_42_20261109_logtransformation_4.xyz"
]

liflow_dt_ps = 500 / 1e3
liflow_msd_time_window = 300
liflow_msd_time_shift = 100
liflow_fit_start = 80
liflow_traj_start = 800

# ── Dataset (AIMD) ──
dataset_file = "../../data/data/small_H2_300K_nvt/output.xyz"
dataset_dt_ps = 10 / 1e3
dataset_msd_time_window = 15000
dataset_msd_time_shift = 5000
dataset_fit_start = 4000
dataset_traj_start = 40000

# ── 共通設定 ──
species = "He"

# すべてのグラフで元コードと同じラベル・色を使用
inference_label = "Inference"
inference_color = "red"

# plot_title = "MSD: Before Fine-tuning (1ns Rollout)"
plot_title = "MSD: After Log Transformation (1ns Rollout)"

# 出力先
output_dir = Path("msd_comparison_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 元コードと同様、保存後にグラフを表示
show_plots = True


# ═════════════════════════════════════════════
# DatasetのMSD：1回だけ計算
# ═════════════════════════════════════════════
print("=" * 52)
print(f"  {target_temp}K MSD Comparison")
print("=" * 52)

print("\n[Dataset (AIMD)]")

times_dataset, msd_dataset = calculate_msd_from_file(
    dataset_file,
    species,
    dataset_msd_time_window,
    dataset_msd_time_shift,
    dataset_dt_ps,
    traj_start=dataset_traj_start,
)

if times_dataset is not None:
    D_dataset, slope_d, int_d, xfit_d = calculate_diffusion_coefficient(
        msd_dataset,
        dataset_dt_ps,
        dataset_fit_start,
    )

    print(f"  D (Dataset) = {D_dataset:.4e} cm²/s")


# ═════════════════════════════════════════════
# 推論ファイルごとに計算・描画・保存
# ═════════════════════════════════════════════
for file_index, traj_file in enumerate(liflow_files, start=1):
    print(f"\n[{file_index}/{len(liflow_files)}] {traj_file}")

    times_lf, msd_lf = calculate_msd_from_file(
        traj_file,
        species,
        liflow_msd_time_window,
        liflow_msd_time_shift,
        liflow_dt_ps,
        traj_start=liflow_traj_start,
    )

    if times_lf is None:
        continue

    D_lf, slope_lf, int_lf, xfit_lf = calculate_diffusion_coefficient(
        msd_lf,
        liflow_dt_ps,
        liflow_fit_start,
    )

    print(f"  D ({inference_label}) = {D_lf:.4e} cm²/s")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Dataset
    if times_dataset is not None:
        ax.plot(
            times_dataset,
            msd_dataset,
            color="black",
            linestyle="-",
            linewidth=2,
            label="Dataset",
        )

        ax.plot(
            xfit_d,
            slope_d * xfit_d + int_d,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Fit Dataset (D={D_dataset:.2e} cm²/s)",
        )

    # Inference
    ax.plot(
        times_lf,
        msd_lf,
        color=inference_color,
        linestyle="--",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=10,
        label=inference_label,
    )

    ax.plot(
        xfit_lf,
        slope_lf * xfit_lf + int_lf,
        color=inference_color,
        linestyle=":",
        linewidth=1.5,
        label=f"Fit {inference_label} (D={D_lf:.2e} cm²/s)",
    )

    ax.set_xlabel(r"Lag time, $\tau$ (ps)")
    ax.set_ylabel(r"MSD($\tau$) ($\mathrm{\AA^2}$)")
    ax.set_title(plot_title, pad=10)

    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout(pad=0.5)

    # 入力ファイル名と通し番号から出力名を作る
    input_stem = Path(traj_file).stem
    save_filename = output_dir / (
        f"msd_comparison_{target_temp}K_"
        f"{file_index:02d}_{input_stem}.png"
    )

    fig.savefig(save_filename, dpi=300)
    print(f"  比較グラフを保存しました: {save_filename}")

    if show_plots:
        plt.show()

    plt.close(fig)