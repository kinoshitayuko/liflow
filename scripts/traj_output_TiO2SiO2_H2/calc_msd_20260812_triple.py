import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ase.io import read
import os

# ─────────────────────────────────────────────
# MSD 計算関数（既存コードのまま変更なし）
# ─────────────────────────────────────────────

def get_atom_indices(traj, species="He"):
    return [
        i for i, symbol in enumerate(traj[0].get_chemical_symbols())
        if symbol == species
    ]

def calculate_msd(traj, atom_indices, msd_time_window, msd_time_shift):
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

def calculate_msd_from_file(traj_file, species, msd_time_window, msd_time_shift,
                             timestep_ps, traj_start=0):
    if not os.path.exists(traj_file):
        print(f"  [WARN] ファイルが見つかりません: {traj_file}")
        return None, None

    print(f"  読み込み中: {traj_file}")
    traj = read(traj_file, index=":")

    traj = traj[traj_start:]
    print(f"  使用フレーム: {traj_start} 〜 {traj_start + len(traj) - 1}")

    atom_indices = get_atom_indices(traj, species)
    print(f"  {species} 原子数: {len(atom_indices)},  フレーム数: {len(traj)}")

    msd = calculate_msd(traj, atom_indices, msd_time_window, msd_time_shift)
    time = np.arange(len(msd)) * timestep_ps
    return time, msd

def calculate_diffusion_coefficient(msd, timestep_ps, fit_start, fit_end=None):
    x_fit_ps = np.arange(len(msd))[fit_start:fit_end] * timestep_ps
    y_fit = msd[fit_start:fit_end]
    slope, intercept, _, _, _ = stats.linregress(x_fit_ps, y_fit)
    D = slope / 6      # [Å²/ps]
    D *= 1.0e-4        # [cm²/s]
    return D, slope, intercept, x_fit_ps


# ═════════════════════════════════════════════
# 設定（3つのファイルのパラメータを辞書で一括管理）
# ═════════════════════════════════════════════

target_temp = 300
species = "He"

# 3つのファイルの設定をここに定義します。
# 実際の3つ目のファイルパス（File 3）に合わせて書き換えてください。
file_configs = {
    "Dataset (AIMD 300K)": {
        "file": "../../data/data/small_H2_300K_nvt/output.xyz",
        "dt_ps": 10 / 1e3,               # 0.010 ps
        "msd_time_window": 15000,
        "msd_time_shift": 5000,
        "fit_start": 4000,
        "fit_end": 15000,
        "traj_start": 50000,
        "style": {"color": "black", "ls": "-", "marker": None}
    },
    "Dataset (AIMD 450K)": {
        "file": "../../data/data/small_H2_450K_nvt/output.xyz",
        "dt_ps": 10 / 1e3,              # 0.500 ps
        "msd_time_window": 15000,
        "msd_time_shift": 5000,
        "fit_start": 4000,
        "fit_end": 15000,
        "traj_start": 50000,
        "style": {"color": "red", "ls": "--", "marker": "o"}
    },
    "Dataset (AIMD 600K)": {                # 🌟 ここに3つ目のファイルを指定
        "file": "../../data/data/small_H2_600K_nvt/output.xyz",
        "dt_ps": 10 / 1e3,              # 0.500 ps
        "msd_time_window": 15000,
        "msd_time_shift": 5000,
        "fit_start": 4000,
        "fit_end": 15000,
        "traj_start": 50000,
        "style": {"color": "blue", "ls": "--", "marker": "s"}
    }
}


# ═════════════════════════════════════════════
# 計算実行
# ═════════════════════════════════════════════

print(f"{'='*52}")
print(f"  {target_temp}K  MSD Comparison")
print(f"{'='*52}")

results = {}

for label, config in file_configs.items():
    print(f"\n[{label}]")
    times, msd = calculate_msd_from_file(
        traj_file=config["file"], 
        species=species,
        msd_time_window=config["msd_time_window"], 
        msd_time_shift=config["msd_time_shift"], 
        timestep_ps=config["dt_ps"], 
        traj_start=config["traj_start"]
    )
    
    if times is not None:
        D, slope, intercept, xfit = calculate_diffusion_coefficient(
            msd, config["dt_ps"], config["fit_start"], config["fit_end"]
        )
        print(f"  D ({label}) = {D:.4e} cm²/s")
        # 計算結果と描画スタイルを保存
        results[label] = {
            "times": times, "msd": msd, 
            "D": D, "slope": slope, "intercept": intercept, "xfit": xfit,
            "style": config["style"]
        }


# ═════════════════════════════════════════════
# プロット（MSD + フィット直線の重ね書き）
# ═════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))

for label, res in results.items():
    style = res["style"]
    
    # MSDカーブのプロット
    ax.plot(
        res["times"], res["msd"],
        color=style["color"], linestyle=style["ls"], linewidth=2,
        marker=style["marker"], markersize=4 if style["marker"] else 0,
        markevery=10 if style["marker"] else None,
        label=label
    )
    
    # フィット直線のプロット
    ax.plot(
        res["xfit"], res["slope"] * res["xfit"] + res["intercept"],
        color=style["color"], linestyle=":", linewidth=1.5,
        label=f"Fit {label.split('(')[0].strip()} (D={res['D']:.2e} cm²/s)"
    )

ax.set_xlabel("Lag time, τ (ps)", fontsize=14)
ax.set_ylabel(r"MSD(τ) ($\mathrm{\AA^2}$)", fontsize=14)
ax.set_title(f"H Mean Squared Displacement at {target_temp} K", fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

save_filename = f"msd_comparison_{target_temp}K_3files_20260817.png"
plt.savefig(save_filename, dpi=300)
print(f"\n比較グラフを {save_filename} に保存しました。")
plt.show()