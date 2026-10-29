# # compute_sigma_LGPS.py
# from pathlib import Path
# import numpy as np
# import ase.io

# # ── 設定 ──────────────────────────────────────────────────────
# XYZ_PATH    = Path("positions_1400K.xyz")  # ← 適宜変更
# TIME_DELAY  = 67          # フレーム間隔
# OUTPUT_NPY  = Path("sigma_LGPS_1400K_67step.npy")   # 保存先

# # Li の原子番号 = 3, Ge = 32, P = 15, S = 16
# DIFF_ATOMIC_NUMS  = {3}          # Li（拡散原子）
# FRAME_ATOMIC_NUMS = {32, 15, 16} # Ge, P, S（フレーム原子）

# # ── データ読み込み ─────────────────────────────────────────────
# print("読み込み中...")
# frames = ase.io.read(XYZ_PATH, index=":")
# N_frames = len(frames)
# N_atoms  = len(frames[0])
# print(f"  フレーム数: {N_frames}, 原子数: {N_atoms}")

# # 全座標を numpy 配列に変換 [T, N, 3]
# positions = np.stack([f.get_positions() for f in frames], axis=0)

# # ── σ 計算 ────────────────────────────────────────────────────
# # 有効なペア数
# n_pairs = N_frames - TIME_DELAY

# if n_pairs <= 0:
#     raise ValueError(f"フレーム数({N_frames}) が TIME_DELAY({TIME_DELAY}) 以下です")

# print(f"  有効ペア数: {n_pairs}")

# # Δr = r(t + 67) - r(t)  → [n_pairs, N, 3]
# delta_r = positions[TIME_DELAY:] - positions[:n_pairs]  # PBCなし（必要ならあとで補正）

# # σ = RMS over all pairs → [N, 3]
# sigma = np.sqrt(np.mean(delta_r ** 2, axis=0))   # [N, 3]

# # スカラー σ（各原子の等方的な標準偏差）= √(σx² + σy² + σz²) / √3
# sigma_scalar = np.sqrt(np.mean(sigma ** 2, axis=-1))  # [N]

# # ── 原子種別に集計 ────────────────────────────────────────────
# atomic_numbers = frames[0].get_atomic_numbers()  # [N]

# diff_mask  = np.isin(atomic_numbers, list(DIFF_ATOMIC_NUMS))
# frame_mask = np.isin(atomic_numbers, list(FRAME_ATOMIC_NUMS))

# print("\n── σ（RMS変位, 67step） ──────────────────────")
# print(f"  全原子    σ_scalar 平均: {sigma_scalar.mean():.4f} Å")
# print(f"  Li（diff） σ_scalar 平均: {sigma_scalar[diff_mask].mean():.4f} Å  "
#       f"(n={diff_mask.sum()})")
# print(f"  Ge/P/S（frame）σ_scalar 平均: {sigma_scalar[frame_mask].mean():.4f} Å  "
#       f"(n={frame_mask.sum()})")
# print(f"\n  σ_x / σ_y / σ_z (Li 平均):")
# print(f"    x: {sigma[diff_mask, 0].mean():.4f} Å")
# print(f"    y: {sigma[diff_mask, 1].mean():.4f} Å")
# print(f"    z: {sigma[diff_mask, 2].mean():.4f} Å")
# print(f"\n  σ_x / σ_y / σ_z (frame 平均):")
# print(f"    x: {sigma[frame_mask, 0].mean():.4f} Å")
# print(f"    y: {sigma[frame_mask, 1].mean():.4f} Å")
# print(f"    z: {sigma[frame_mask, 2].mean():.4f} Å")

# # ── 保存 ─────────────────────────────────────────────────────
# np.save(OUTPUT_NPY, sigma)   # [N, 3]
# print(f"\n  保存完了: {OUTPUT_NPY}  shape={sigma.shape}")

# compute_sigma_LGPS.py
from pathlib import Path
import numpy as np
import ase.io

# ── 設定 ──────────────────────────────────────────────────────
# ★ 変更点①：複数ファイルのリストに変更
XYZ_PATHS = [
    Path("positions_650K.xyz"),
    Path("positions_900K.xyz"),
    Path("positions_1150K.xyz"),
    Path("positions_1400K.xyz"),
]
TIME_DELAY = 67
OUTPUT_NPY = Path("sigma_LGPS_alltemp_67step.npy")

# POSCAR から格子定数を取得（XYZに格子情報がないため）
poscar = ase.io.read("Li10GeS2P12_AIMD_stoichiometric_650K_POSCAR.sid")
cell   = poscar.get_cell().diagonal()
print(f"格子定数: {cell}")

DIFF_ATOMIC_NUMS  = {3}
FRAME_ATOMIC_NUMS = {32, 15, 16}

# ★ 変更点②：温度ごとの sigma を集める
sigma_list = []

for XYZ_PATH in XYZ_PATHS:   # ← ループを追加しただけ

    # ── データ読み込み（元コードそのまま）────────────────────
    print(f"\n読み込み中: {XYZ_PATH.name}")
    frames   = ase.io.read(XYZ_PATH, index=":")
    N_frames = len(frames)
    N_atoms  = len(frames[0])
    print(f"  フレーム数: {N_frames}, 原子数: {N_atoms}")

    positions = np.stack([f.get_positions() for f in frames], axis=0)

    # ── σ 計算（元コードそのまま）────────────────────────────
    n_pairs = N_frames - TIME_DELAY
    if n_pairs <= 0:
        raise ValueError(f"フレーム数({N_frames}) が TIME_DELAY({TIME_DELAY}) 以下です")
    print(f"  有効ペア数: {n_pairs}")

    delta_r = positions[TIME_DELAY:] - positions[:n_pairs]

    # ★ PBC補正を追加
    delta_r -= np.round(delta_r / cell) * cell

    sigma        = np.sqrt(np.mean(delta_r ** 2, axis=0))   # [N, 3]
    sigma_scalar = np.sqrt(np.mean(sigma ** 2, axis=-1))     # [N]

    # ── 原子種別集計（元コードそのまま）──────────────────────
    atomic_numbers = frames[0].get_atomic_numbers()
    diff_mask  = np.isin(atomic_numbers, list(DIFF_ATOMIC_NUMS))
    frame_mask = np.isin(atomic_numbers, list(FRAME_ATOMIC_NUMS))

    print(f"\n── σ（RMS変位, 67step） ──────────────────────")
    print(f"  全原子    σ_scalar 平均: {sigma_scalar.mean():.4f} Å")
    print(f"  Li（diff） σ_scalar 平均: {sigma_scalar[diff_mask].mean():.4f} Å  (n={diff_mask.sum()})")
    print(f"  Ge/P/S（frame）σ_scalar 平均: {sigma_scalar[frame_mask].mean():.4f} Å  (n={frame_mask.sum()})")
    print(f"\n  σ_x / σ_y / σ_z (Li 平均):")
    print(f"    x: {sigma[diff_mask, 0].mean():.4f} Å")
    print(f"    y: {sigma[diff_mask, 1].mean():.4f} Å")
    print(f"    z: {sigma[diff_mask, 2].mean():.4f} Å")
    print(f"\n  σ_x / σ_y / σ_z (frame 平均):")
    print(f"    x: {sigma[frame_mask, 0].mean():.4f} Å")
    print(f"    y: {sigma[frame_mask, 1].mean():.4f} Å")
    print(f"    z: {sigma[frame_mask, 2].mean():.4f} Å")

    sigma_list.append(sigma)   # ← ループ末尾に追加

# ★ 変更点③：4温度の単純平均を取って保存
sigma_mean = np.mean(sigma_list, axis=0)   # [N, 3]
np.save(OUTPUT_NPY, sigma_mean)
print(f"\n  保存完了（全温度平均）: {OUTPUT_NPY}  shape={sigma_mean.shape}")

# ── 全温度平均の集計出力（追加部分）─────────────────────────
sigma_mean_scalar = np.sqrt(np.mean(sigma_mean ** 2, axis=-1))  # [N]

print("\n" + "="*50)
print("【全温度平均（650〜1400K）の σ まとめ】")
print("="*50)
print(f"  Li（diff）σ_scalar 平均: {sigma_mean_scalar[diff_mask].mean():.4f} Å  (n={diff_mask.sum()})")
print(f"    x: {sigma_mean[diff_mask, 0].mean():.4f} Å")
print(f"    y: {sigma_mean[diff_mask, 1].mean():.4f} Å")
print(f"    z: {sigma_mean[diff_mask, 2].mean():.4f} Å")
print(f"  Ge/P/S（frame）σ_scalar 平均: {sigma_mean_scalar[frame_mask].mean():.4f} Å  (n={frame_mask.sum()})")
print(f"    x: {sigma_mean[frame_mask, 0].mean():.4f} Å")
print(f"    y: {sigma_mean[frame_mask, 1].mean():.4f} Å")
print(f"    z: {sigma_mean[frame_mask, 2].mean():.4f} Å")