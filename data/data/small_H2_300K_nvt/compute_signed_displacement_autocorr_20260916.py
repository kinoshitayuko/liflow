"""650K/300K データセット(正解) vs 推論結果 の 符号付き変位自己相関 (Signed DACF) 比較スクリプト

- 1-step の粗視化時間幅 Δt = 0.5 ps を基準とし、各ステップの成分別符号付き変位 (Δx, Δy, Δz) を算出
- dataset      : 10 fs 刻みの場合、50 step ごとサンプリングして 0.5 ps 変位を計算
- inference    : 0.5 ps 刻みの場合、1 step ごとに変位を計算
- x, y, z それぞれについて全時間原点平均による自己相関関数 C_a(lag) を計算・正規化 (C_a(0) = 1.0)
- グラフ出力: x, y, z の 3 分割サブプロット (横軸: Time lag [ps], 縦軸: Signed Autocorrelation)
"""

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# ユーザー設定
# ==========================================
DATASET_FILE = "output.xyz"
INFERENCE_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_44_20260817_finetune_1.xyz"

DATASET_RAW_DT_PS = 0.010  # 10 fs
INFERENCE_RAW_DT_PS = 0.5  # 0.5 ps

# 1-step 変位の評価基準とする物理時間幅 (ps)
TARGET_DT_PS = 0.5

# グラフ横軸の上限 (ps) : まずは 5.0 ps (10 lags) 程度を推奨
MAX_TIME_PS = 5.0

# 対象原子種 (xyzファイル中の元素記号)
TARGET_ELEMENTS = ["He"]


# ==========================================
# 軌道読み込み & 抽出
# ==========================================
def read_xyz_trajectory(filename):
  """標準的なマルチフレームxyzファイルを読み込み、

  positions: (n_frames, n_atoms, 3), elements: (n_atoms,) の元素記号リストを返す
  """
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
      elements = frame_elements

  return np.array(frames), np.array(elements)


def filter_target_atoms(positions, elements, target_elements):
  """指定した元素記号に一致する原子だけを抽出する

  positions: (n_frames, n_atoms, 3)
  """
  mask = np.isin(elements, target_elements)
  n_selected = mask.sum()
  if n_selected == 0:
    raise ValueError(
        f"TARGET_ELEMENTS={target_elements} に一致する原子が見つかりません。"
        f"xyzファイル中の元素記号を確認してください (例: {np.unique(elements)[:10]})"
    )
  return positions[:, mask, :], n_selected


# ==========================================
# 符号付き変位 & 自己相関の計算
# ==========================================
def compute_signed_displacements(positions):
  """隣接フレーム間の1-step変位ベクトル Δr_t = (Δx, Δy, Δz) をそのまま返す (符号付き)

  positions: (n_frames, n_atoms, 3)
  returns: displacements: (n_frames - 1, n_atoms, 3)
  """
  return positions[1:] - positions[:-1]


def compute_signed_dacf(displacements, max_lag):
  """displacements: (n_steps, n_atoms, 3) -- 符号付き変位系列

  x, y, z それぞれについて全開始原点 t0 での積を計算し、分散で正規化 (C(0) = 1.0) して返す。
  returns: dacf: (max_lag + 1, 3)  [0: x, 1: y, 2: z]
  """
  n_steps, n_atoms, n_dims = displacements.shape
  max_lag = min(max_lag, n_steps - 1)
  dacf = np.zeros((max_lag + 1, n_dims))

  for comp in range(n_dims):
    series = displacements[:, :, comp]  # (n_steps, n_atoms)

    mu_a = series.mean()
    var_a = ((series - mu_a) ** 2).mean()

    if var_a == 0:
      continue

    series_centered = series - mu_a

    for lag in range(max_lag + 1):
      n_t0 = n_steps - lag
      # 全原子・全時間起点での積の平均
      cov = (series_centered[:n_t0] * series_centered[lag : lag + n_t0]).mean()
      dacf[lag, comp] = cov / var_a

  return dacf


def build_dacf_curves(
    xyz_file, raw_dt_ps, target_dt_ps, max_time_ps, target_elements
):
  positions, elements = read_xyz_trajectory(xyz_file)
  positions, n_selected = filter_target_atoms(
      positions, elements, target_elements
  )
  print(
      f"[{xyz_file}] 対象原子({target_elements})の数: {n_selected} /"
      f" 全原子数: {len(elements)}"
  )

  # TARGET_DT_PS (0.5 ps) に対応する stride を算出
  stride = max(1, int(round(target_dt_ps / raw_dt_ps)))
  effective_dt = raw_dt_ps * stride
  print(
      f"  -> raw_dt: {raw_dt_ps:.4f} ps, stride: {stride}, effective_dt:"
      f" {effective_dt:.4f} ps"
  )

  coarse_positions = positions[::stride]
  displacements = compute_signed_displacements(coarse_positions)

  # 各成分の統計量をコンソールに表示
  labels = ["x", "y", "z"]
  for c in range(3):
    vals = displacements[:, :, c]
    print(
        f"  -> Δ{labels[c]}: mean = {vals.mean():+.4f} Å, std = {vals.std():.4f}"
        " Å"
    )

  max_lag = int(np.ceil(max_time_ps / effective_dt))
  dacf = compute_signed_dacf(displacements, max_lag)

  time_axis = np.arange(len(dacf)) * effective_dt
  return time_axis, dacf


# ==========================================
# メインルーチン
# ==========================================
def main():
  # 1. データセットの計算
  t_dataset, dacf_dataset = build_dacf_curves(
      DATASET_FILE,
      DATASET_RAW_DT_PS,
      TARGET_DT_PS,
      MAX_TIME_PS,
      TARGET_ELEMENTS,
  )

  # 2. 推論結果の計算 (実行する場合はコメントアウトを解除)
  t_inference, dacf_inference = build_dacf_curves(
      INFERENCE_FILE,
      INFERENCE_RAW_DT_PS,
      TARGET_DT_PS,
      MAX_TIME_PS,
      TARGET_ELEMENTS,
  )

  # 3. プロット (x, y, z の 3 分割表示)
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
  dim_names = ["x", "y", "z"]

  for c in range(3):
    ax = axes[c]
    # Dataset
    ax.plot(
        t_dataset,
        dacf_dataset[:, c],
        marker="o",
        color="tab:blue",
        label="Dataset (output.xyz)",
    )

    # Inference (コメントアウトを解除して使用)
    ax.plot(
        t_inference,
        dacf_inference[:, c],
        marker="s",
        color="tab:orange",
        label="Inference (DW-CFM)",
    )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, MAX_TIME_PS)
    ax.set_ylim(-0.4, 1.05)
    ax.set_xlabel("Time lag [ps]")
    ax.set_title(f"Δ{dim_names[c]}-axis Autocorrelation")
    ax.grid(alpha=0.3)
    if c == 0:
      ax.set_ylabel("Signed Displacement Autocorr")
      ax.legend()

  fig.suptitle(
      f"Signed Displacement Autocorrelation (Δt={TARGET_DT_PS} ps, target:"
      f" {TARGET_ELEMENTS})",
      fontsize=13,
      y=1.02,
  )
  fig.tight_layout()

  out_png = "signed_displacement_autocorr_comparison_20260916.png"
  fig.savefig(out_png, dpi=200, bbox_inches="tight")
  print(f"Saved: {out_png}")


if __name__ == "__main__":
  main()