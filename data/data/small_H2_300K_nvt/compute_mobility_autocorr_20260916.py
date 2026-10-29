"""650K/300K データセット(正解) vs 推論結果 の 移動度自己相関 (MACF) 比較スクリプト (対象原子のみ)

- 1-step の粗視化時間幅 Δt = 0.5 ps を基準とし、各ステップの移動度 m_t = |Δr_t| を算出
- dataset      : 10 fs 刻みの場合、50 step ごとサンプリングして 0.5 ps 変位を計算
- inference    : 0.5 ps 刻みの場合、1 step ごとに変位を計算
- 移動度 m_t の全時間原点平均による自己相関関数 C_m(lag) を計算・正規化 (C_m(0) = 1.0)
- グラフ出力 (横軸: Time lag [ps], 縦軸: Normalized Mobility Autocorrelation)
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
TARGET_DT_PS = 0.010
# TARGET_DT_PS = 0.5

# グラフ横軸の上限 (ps) : まずは 10 ps (20 lags) 程度を推奨
# MAX_TIME_PS = 10.0
MAX_TIME_PS = 2.0

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
# 移動度 (Mobility) & 自己相関の計算
# ==========================================
def compute_mobilities(positions):
  """隣接フレーム間の1-step変位ベクトル Δr_t の大きさ (ノルム) を計算

  positions: (n_frames, n_atoms, 3)
  returns: mobilities: (n_frames - 1, n_atoms)
  """
  disp_vectors = positions[1:] - positions[:-1]  # (n_frames - 1, n_atoms, 3)
  mobilities = np.linalg.norm(disp_vectors, axis=2)  # (n_frames - 1, n_atoms)
  return mobilities


def compute_macf(mobilities, max_lag):
  """mobilities: (n_steps, n_atoms) -- 1-step 変位の大きさ系列

  全開始原点 t0 での積を計算し、分散で正規化 (C_m(0) = 1.0) して返す。
  """
  n_steps, n_atoms = mobilities.shape
  max_lag = min(max_lag, n_steps - 1)

  # 全ステップ・全原子での平均と分散
  m_mean = mobilities.mean()
  m_var = ((mobilities - m_mean) ** 2).mean()

  if m_var == 0:
    return np.zeros(max_lag + 1)

  m_centered = mobilities - m_mean

  macf = np.zeros(max_lag + 1)
  for lag in range(max_lag + 1):
    n_t0 = n_steps - lag
    # t0 と t0+lag の積を全原子・全時間起点で平均
    cov = (m_centered[:n_t0] * m_centered[lag : lag + n_t0]).mean()
    macf[lag] = cov / m_var

  return macf


def build_macf_curve(
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
  mobilities = compute_mobilities(coarse_positions)

  # 統計量の簡易表示（尖度や歪度の確認に役立ちます）
  print(
      f"  -> 1-step mobility (Δt={effective_dt:.2f} ps): mean ="
      f" {mobilities.mean():.4f}, std = {mobilities.std():.4f}"
  )

  max_lag = int(np.ceil(max_time_ps / effective_dt))
  macf = compute_macf(mobilities, max_lag)

  time_axis = np.arange(len(macf)) * effective_dt
  return time_axis, macf


# ==========================================
# メインルーチン
# ==========================================
def main():
  # 1. データセットの計算
  t_dataset, macf_dataset = build_macf_curve(
      DATASET_FILE,
      DATASET_RAW_DT_PS,
      TARGET_DT_PS,
      MAX_TIME_PS,
      TARGET_ELEMENTS,
  )

#   # 2. 推論結果の計算 (実行する場合はコメントアウトを解除してください)
#   t_inference, macf_inference = build_macf_curve(
#       INFERENCE_FILE, INFERENCE_RAW_DT_PS, TARGET_DT_PS, MAX_TIME_PS, TARGET_ELEMENTS
#   )

  # 3. プロット
  fig, ax = plt.subplots(figsize=(7, 5))
  ax.plot(
      t_dataset,
      macf_dataset,
      marker="o",
      color="tab:blue",
      label="Dataset (output.xyz)",
  )

#   # 推論結果プロット用
#   ax.plot(
#       t_inference, macf_inference,
#       marker="s", color="tab:orange",
#       label="Inference (DW-CFM fine-tune)"
#   )

  ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

  ax.set_xlim(0, MAX_TIME_PS)
  ax.set_ylim(-0.2, 1.05)
  ax.set_xlabel("Time lag [ps]")
  ax.set_ylabel("Normalized Mobility Autocorrelation")
  ax.set_title(
      f"Mobility Autocorrelation (Δt={TARGET_DT_PS} ps, target:"
      f" {TARGET_ELEMENTS})"
  )
  ax.legend()
  ax.grid(alpha=0.3)

  fig.tight_layout()
  out_png = "mobility_autocorr_comparison_dataset_20260916_10fs.png"
  fig.savefig(out_png, dpi=200)
  print(f"Saved: {out_png}")


if __name__ == "__main__":
  main()