import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ase.io import read
import os


# ─────────────────────────────────────────────
# MSD 計算関数（calc_msd.py ベース）
# ─────────────────────────────────────────────

def get_atom_indices(traj, species="He"):
    return [
        i for i, symbol in enumerate(traj[0].get_chemical_symbols())
        if symbol == species
    ]


def calculate_msd(traj, atom_indices, msd_time_window, msd_time_shift):
    """
    時間起点をずらしながら MSD を計算して平均する。

    Parameters
    ----------
    traj            : ASE Atoms のリスト（unwrapped 座標を想定）
    atom_indices    : 対象原子のインデックスリスト
    msd_time_window : ウィンドウ長 [steps]
    msd_time_shift  : 時間起点のシフト間隔 [steps]

    Returns
    -------
    msd : np.ndarray, shape (msd_time_window + 1,)  [Å²]
    """
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
                    (atoms.get_positions()[atom_indices] - initial_positions) ** 2,
                    axis=1
                )
            )
            for atoms in traj_tmp
        ]
        msd_list.append(msd_tmp)
        traj_ini += msd_time_shift
        traj_fin += msd_time_shift

    print(f"  Number of windows: {n_count}")
    return np.mean(msd_list, axis=0)


def calculate_msd_from_file(traj_file, species, msd_time_window, msd_time_shift, timestep_ps):
    """
    xyz ファイルを読み込んで MSD と時間軸を返す。
    ファイルが存在しない場合は (None, None) を返す。
    """
    if not os.path.exists(traj_file):
        print(f"  [WARN] ファイルが見つかりません: {traj_file}")
        return None, None

    print(f"  読み込み中: {traj_file}")
    traj = read(traj_file, index=":")
    atom_indices = get_atom_indices(traj, species)
    print(f"  {species} 原子数: {len(atom_indices)},  フレーム数: {len(traj)}")

    msd = calculate_msd(traj, atom_indices, msd_time_window, msd_time_shift)
    time = np.arange(len(msd)) * timestep_ps
    return time, msd


def calculate_diffusion_coefficient(msd, timestep_ps, fit_start):
    """
    MSD の線形フィットから自己拡散係数 D を計算する。

    Parameters
    ----------
    msd         : np.ndarray [Å²]
    timestep_ps : タイムステップ [ps]
    fit_start   : フィット開始インデックス（ballistic 領域を除外）

    Returns
    -------
    D               : 自己拡散係数 [cm²/s]
    slope           : フィット直線の傾き [Å²/ps]
    intercept       : フィット直線の切片 [Å²]
    x_fit_ps        : フィット範囲の時間軸 [ps]
    """
    x_fit_ps = np.arange(len(msd))[fit_start:] * timestep_ps  # [ps]
    y_fit = msd[fit_start:]
    slope, intercept, _, _, _ = stats.linregress(x_fit_ps, y_fit)
    D = slope / 6      # [Å²/ps]
    D *= 1.0e-4        # [cm²/s]
    return D, slope, intercept, x_fit_ps


# ═════════════════════════════════════════════
# 設定（ここを環境に合わせて変更してください）
# ═════════════════════════════════════════════

target_temp = 300  # 比較する温度 (K)

# ── LiFlow 推論ファイル ──────────────────────
# label: ファイルパス  の辞書で複数ファイルを一括処理できます
liflow_files = {
    "LiFlow (default, sigma_H=10)": "traj_300K_42_20260514_0.5_1.xyz",
    # "LiFlow (sigma_Li=20)":        "traj_300K_42_20260513_0.1_10.xyz",
    # "LiFlow (sigma_Li=30)":        "traj_300K_42_20260513_0.1_30.xyz",
}
liflow_dt_ps = 500 / 1e3  # 500 fs → 0.500 ps

# LiFlow 用 MSD パラメータ（0.500 ps/step, 約 150 ps 軌跡）
liflow_msd_time_window = 300  # 300 steps × 0.500 ps ≈ 150.000 ps
liflow_msd_time_shift  = 1    # 1 step ずつずらして起点を最大化
liflow_fit_start       = 5    # 最初の 5 steps（≈ 5 ps）を除外

# ── Dataset (AIMD) ファイル ──────────────────
dataset_file   = "../../data/data/small_H2_300K_nvt/output.xyz"
dataset_dt_ps  = 10 / 1e3   # 10 fs → 0.010 ps

# Dataset 用 MSD パラメータ（0.010 ps/step）
# positions_300K.xyz のフレーム数に合わせて変更してください
# 確認方法: traj = read(dataset_file, index=":"); print(len(traj))
dataset_msd_time_window = 15000  # 15000 steps × 0.010 ps = 150.000 ps
dataset_msd_time_shift  = 50    # 50 steps ごとに起点をシフト
dataset_fit_start       = 500   # 最初の 500 steps（≈ 5 ps）を除外

# ── 共通設定 ─────────────────────────────────
species = "He"
liflow_colors = ["red", "blue", "green", "orange", "purple"]

# ═════════════════════════════════════════════
# 計算実行
# ═════════════════════════════════════════════

print(f"{'='*52}")
print(f"  {target_temp}K  MSD Comparison")
print(f"{'='*52}")

# Dataset (AIMD) の MSD 計算
print("\n[Dataset (AIMD)]")
times_dataset, msd_dataset = calculate_msd_from_file(
    dataset_file, species,
    dataset_msd_time_window, dataset_msd_time_shift, dataset_dt_ps
)
if times_dataset is not None:
    D_dataset, slope_d, int_d, xfit_d = calculate_diffusion_coefficient(
        msd_dataset, dataset_dt_ps, dataset_fit_start
    )
    print(f"  D (Dataset) = {D_dataset:.4e} cm²/s")

# LiFlow の MSD 計算（複数ファイル対応）
liflow_results = {}
for label, traj_file in liflow_files.items():
    print(f"\n[{label}]")
    times_lf, msd_lf = calculate_msd_from_file(
        traj_file, species,
        liflow_msd_time_window, liflow_msd_time_shift, liflow_dt_ps
    )
    if times_lf is not None:
        D_lf, slope_lf, int_lf, xfit_lf = calculate_diffusion_coefficient(
            msd_lf, liflow_dt_ps, liflow_fit_start
        )
        print(f"  D ({label}) = {D_lf:.4e} cm²/s")
        liflow_results[label] = (times_lf, msd_lf, D_lf, slope_lf, int_lf, xfit_lf)

# ═════════════════════════════════════════════
# プロット（MSD + フィット直線の重ね書き）
# ═════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))

# Dataset (AIMD) をプロット
if times_dataset is not None:
    ax.plot(
        times_dataset, msd_dataset,
        color="black", linestyle="-", linewidth=2,
        label="Dataset (AIMD)"
    )
    ax.plot(
        xfit_d, slope_d * xfit_d + int_d,
        color="black", linestyle=":", linewidth=1.5,
        label=f"Fit Dataset (D={D_dataset:.2e} cm²/s)"
    )

# LiFlow をプロット
for (label, result), color in zip(liflow_results.items(), liflow_colors):
    times_lf, msd_lf, D_lf, slope_lf, int_lf, xfit_lf = result
    ax.plot(
        times_lf, msd_lf,
        color=color, linestyle="--", linewidth=2,
        marker="o", markersize=4, markevery=10,
        label=label
    )
    ax.plot(
        xfit_lf, slope_lf * xfit_lf + int_lf,
        color=color, linestyle=":", linewidth=1.5,
        label=f"Fit {label.split('(')[0].strip()} (D={D_lf:.2e} cm²/s)"
    )

ax.set_xlabel("Time (ps)", fontsize=14)
ax.set_ylabel(r"MSD ($\mathrm{\AA^2}$)", fontsize=14)
ax.set_title(f"H Mean Squared Displacement at {target_temp} K", fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

save_filename = f"msd_comparison_{target_temp}K_20260514.png"
plt.savefig(save_filename, dpi=300)
print(f"\n比較グラフを {save_filename} に保存しました。")
plt.show()