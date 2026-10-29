"""
650K データセット(正解) vs 推論結果 の VACF 比較スクリプト (対象原子のみ)
- dataset      : positions_650K.xyz       (raw timestep = 15 fs)
- inference    : traj_650K_44_20260513.xyz (raw timestep = 1.005 ps)
- Δt (VACFの1ラグあたりの物理時間) = 1 ps を目標とする
    -> dataset は 67 step ごと (67 * 15 fs = 1.005 ps ≈ 1 ps) にサンプリングして速度・VACFを計算
    -> inference は 1 step ごと (timestep 自体が 1.005 ps ≈ 1 ps) に計算
- 速度は x,y,z をまとめた3次元ベクトルとして扱い、内積 v(t0)・v(t0+lag) でVACFを計算
  (x,y,z成分別には計算しない)
- VACFは対象原子(拡散させたいゲスト分子, 例: H)のみに絞って計算する ("tagged particle VACF")
  フレームワーク原子(Ti, O, Si など)は含めない
- CSV出力なし。グラフのみ出力 (横軸: 時間 0-10 ps, 縦軸: 正規化VACF 0-1 付近)
"""

import numpy as np
import matplotlib.pyplot as plt

DATASET_FILE = "positions_650K.xyz"
INFERENCE_FILE = "../../../scripts/traj_output_LGPS/traj_650K_44_20260513.xyz"

DATASET_RAW_DT_PS = 0.015      # 15 fs
INFERENCE_RAW_DT_PS = 1.005    # 1.005 ps

DATASET_STRIDE = 1            # 67 step ごとにVACFを計算 (67*15fs≈1.005ps)
INFERENCE_STRIDE = 1           # 1 step ごとにVACFを計算

#MAX_TIME_PS = 10.0             # グラフ横軸の上限 (VACF計算のラグもこれに合わせて決める)
MAX_TIME_PS = 10.0             # グラフ横軸の上限 (VACF計算のラグもこれに合わせて決める)

# VACFの対象とする原子種 (xyzファイル中の元素記号と一致させる)
# 拡散を見たいゲスト分子(例: H2 -> "H")のみを指定する。
# フレームワーク原子(例: "Ti", "O", "Si")は含めないこと。
TARGET_ELEMENTS = ["Li"]


def read_xyz_trajectory(filename):
    """標準的なマルチフレームxyzファイルを読み込み、
    positions: (n_frames, n_atoms, 3), elements: (n_atoms,) の元素記号リストを返す"""
    frames = []
    elements = None

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i].strip()
        if line == "":
            i += 1
            continue
        n_atoms = int(line)
        i += 1  # comment line
        i += 1
        coords = np.zeros((n_atoms, 3), dtype=np.float64)
        frame_elements = []
        for a in range(n_atoms):
            parts = lines[i].split()
            frame_elements.append(parts[0])
            coords[a, 0] = float(parts[1])
            coords[a, 1] = float(parts[2])
            coords[a, 2] = float(parts[3])
            i += 1
        frames.append(coords)
        if elements is None:
            elements = frame_elements  # 元素の並びは全フレームで同一と仮定

    return np.array(frames), np.array(elements)


def filter_target_atoms(positions, elements, target_elements):
    """指定した元素記号に一致する原子だけを抽出する
    positions: (n_frames, n_atoms, 3)"""
    mask = np.isin(elements, target_elements)
    n_selected = mask.sum()
    if n_selected == 0:
        raise ValueError(
            f"TARGET_ELEMENTS={target_elements} に一致する原子が見つかりません。"
            f"xyzファイル中の元素記号を確認してください (例: {np.unique(elements)[:10]})"
        )
    return positions[:, mask, :], n_selected


def compute_velocities(positions, dt):
    """中心差分で速度を計算 (端は前進/後退差分)。
    positions: (n_frames, n_atoms, 3), dt: フレーム間の物理時間 [ps]"""
    velocities = np.zeros_like(positions)
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    return velocities


def compute_vacf(velocities, max_lag):
    """velocities: (n_frames, n_atoms, 3) -- 対象原子のみを渡すこと
    3次元ベクトルの内積 v(t0)・v(t0+lag) を原子・時間起点で平均し、
    正規化 (C(0)=1) して返す。"""
    n_frames = velocities.shape[0]
    max_lag = min(max_lag, n_frames - 1)

    vacf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        n_t0 = n_frames - lag
        dots = np.sum(velocities[:n_t0] * velocities[lag:lag + n_t0], axis=2)  # (n_t0, n_target_atoms)
        vacf[lag] = dots.mean()

    vacf /= vacf[0]
    return vacf


def build_vacf_curve(xyz_file, raw_dt_ps, stride, max_time_ps, target_elements):
    positions, elements = read_xyz_trajectory(xyz_file)
    positions, n_selected = filter_target_atoms(positions, elements, target_elements)
    print(f"[{xyz_file}] 対象原子({target_elements})の数: {n_selected} / 全原子数: {len(elements)}")

    coarse_positions = positions[::stride]
    effective_dt = raw_dt_ps * stride

    velocities = compute_velocities(coarse_positions, effective_dt)

    max_lag = int(np.ceil(max_time_ps / effective_dt))
    vacf = compute_vacf(velocities, max_lag)

    time_axis = np.arange(len(vacf)) * effective_dt
    return time_axis, vacf


def main():
    t_dataset, vacf_dataset = build_vacf_curve(
        DATASET_FILE, DATASET_RAW_DT_PS, DATASET_STRIDE, MAX_TIME_PS, TARGET_ELEMENTS
    )
    t_inference, vacf_inference = build_vacf_curve(
        INFERENCE_FILE, INFERENCE_RAW_DT_PS, INFERENCE_STRIDE, MAX_TIME_PS, TARGET_ELEMENTS
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_dataset, vacf_dataset, marker="o", label="Dataset (positions_650K.xyz)")
    ax.plot(t_inference, vacf_inference, marker="s", label="Inference (traj_650K_44_20260513.xyz)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlim(0, MAX_TIME_PS)
    ax.set_ylim(top=1.05)
    ax.set_xlabel("Time lag [ps]")
    ax.set_ylabel("Normalized VACF")
    ax.set_title(f"VACF comparison at 650 K (target atoms: {TARGET_ELEMENTS})")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    # fig.savefig("vacf_650K_dataset.png", dpi=200)
    # print("Saved: vacf_650K_dataset.png")
    
    fig.savefig("vacf_650K_comparison_2.png", dpi=200)
    print("Saved: vacf_650K_comparison_2.png")


if __name__ == "__main__":
    main()
