import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ase.io import read
import os


# ─────────────────────────────────────────────
# MSD 計算関数
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


def calculate_msd_from_traj(traj, species, msd_time_window, msd_time_shift, timestep_ps, label=""):
    """
    読み込み済みの traj から MSD と時間軸を返す。
    """
    atom_indices = get_atom_indices(traj, species)
    print(f"  [{label}] {species} 原子数: {len(atom_indices)},  使用フレーム数: {len(traj)}")
    msd = calculate_msd(traj, atom_indices, msd_time_window, msd_time_shift)
    time = np.arange(len(msd)) * timestep_ps
    return time, msd


def calculate_diffusion_coefficient(msd, timestep_ps, fit_start):
    """
    MSD の線形フィットから自己拡散係数 D を計算する。

    Returns
    -------
    D         : 自己拡散係数 [cm²/s]
    slope     : フィット直線の傾き [Å²/ps]
    intercept : フィット直線の切片 [Å²]
    x_fit_ps  : フィット範囲の時間軸 [ps]
    """
    x_fit_ps = np.arange(len(msd))[fit_start:] * timestep_ps
    y_fit = msd[fit_start:]
    slope, intercept, _, _, _ = stats.linregress(x_fit_ps, y_fit)
    D = slope / 6 * 1.0e-4  # Å²/ps → cm²/s
    return D, slope, intercept, x_fit_ps


# ═════════════════════════════════════════════
# 設定
# ═════════════════════════════════════════════

target_temp   = 300
species       = "He"
dataset_file  = "../../data/data/small_H2_300K_nvt/output.xyz"
dataset_dt_ps = 10 / 1e3   # 10 fs → 0.010 ps

# ── トラジェクトリの長さ ──────────────────────
# Full  : 100,000 フレーム = 1 ns
# Subset:  50,000 フレーム = 500 ps（前半のみ）
subset_frames = 50000   # 500 ps 分

# ── MSD ウィンドウ（両者共通） ────────────────
# 200 ps = 20000 steps × 0.010 ps
msd_time_window = 20000   # [steps]  → 200 ps
msd_time_shift  = 50      # 起点シフト間隔 [steps]
fit_start       = 0       # ballistic 領域を除外する場合はここを増やす（例: 500）

# ═════════════════════════════════════════════
# トラジェクトリ読み込み（1 回だけ）
# ═════════════════════════════════════════════

if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"ファイルが見つかりません: {dataset_file}")

print(f"読み込み中: {dataset_file}")
traj_full = read(dataset_file, index=":")
n_frames_total = len(traj_full)
print(f"  総フレーム数: {n_frames_total}  ({n_frames_total * dataset_dt_ps:.0f} ps)")

if n_frames_total < subset_frames:
    raise ValueError(
        f"フレーム数が不足しています: {n_frames_total} < {subset_frames}"
    )

# 後半 500 ps だけ切り出し
traj_subset = traj_full[subset_frames:]

# ═════════════════════════════════════════════
# MSD 計算（ウィンドウ 200 ps で統一）
# ═════════════════════════════════════════════

print(f"\n{'='*52}")
print(f"  {target_temp}K  MSD Comparison")
print(f"  Window: {msd_time_window * dataset_dt_ps:.0f} ps  |  Shift: {msd_time_shift * dataset_dt_ps:.3f} ps")
print(f"{'='*52}")

# ── Full trajectory（1 ns） ─────────────────
print(f"\n[Full trajectory  ({n_frames_total * dataset_dt_ps:.0f} ps)]")
times_full, msd_full = calculate_msd_from_traj(
    traj_full, species,
    msd_time_window, msd_time_shift, dataset_dt_ps,
    label="Full"
)
D_full, slope_full, int_full, xfit_full = calculate_diffusion_coefficient(
    msd_full, dataset_dt_ps, fit_start
)
print(f"  D (Full 1 ns) = {D_full:.4e} cm²/s")

# ── Subset trajectory（前半 500 ps） ────────
print(f"\n[Subset trajectory (first {subset_frames * dataset_dt_ps:.0f} ps)]")
times_sub, msd_sub = calculate_msd_from_traj(
    traj_subset, species,
    msd_time_window, msd_time_shift, dataset_dt_ps,
    label="500ps Subset"
)
D_sub, slope_sub, int_sub, xfit_sub = calculate_diffusion_coefficient(
    msd_sub, dataset_dt_ps, fit_start
)
print(f"  D (500 ps)    = {D_sub:.4e} cm²/s")

# ═════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))

# Full trajectory
ax.plot(
    times_full, msd_full,
    color="black", linestyle="-", linewidth=2,
    label=f"AIMD Full (1 ns, {n_frames_total} frames)"
)
ax.plot(
    xfit_full, slope_full * xfit_full + int_full,
    color="black", linestyle=":", linewidth=1.5,
    label=f"Fit Full (D = {D_full:.2e} cm²/s)"
)

# Subset (500 ps)
ax.plot(
    times_sub, msd_sub,
    color="royalblue", linestyle="--", linewidth=2,
    marker="o", markersize=4, markevery=500,
    label=f"AIMD Subset (second half of 500 ps, {subset_frames} frames)"
)
ax.plot(
    xfit_sub, slope_sub * xfit_sub + int_sub,
    color="royalblue", linestyle=":", linewidth=1.5,
    label=f"Fit Subset (D = {D_sub:.2e} cm²/s)"
)

ax.set_xlabel("Time (ps)", fontsize=14)
ax.set_ylabel(r"MSD ($\mathrm{\AA^2}$)", fontsize=14)
ax.set_title(
    f"{species} MSD at {target_temp} K — Full 1 ns vs Second Half of 500 ps\n"
    f"(MSD window: {msd_time_window * dataset_dt_ps:.0f} ps)",
    fontsize=15
)
ax.legend(fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

save_filename = f"msd_full_vs_second half of 500ps_{target_temp}K_200ps.png"
plt.savefig(save_filename, dpi=300)
print(f"\n比較グラフを {save_filename} に保存しました。")
plt.show()