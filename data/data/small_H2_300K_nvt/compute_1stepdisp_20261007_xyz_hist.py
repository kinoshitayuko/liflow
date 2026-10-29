import os
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# フォントサイズの一括設定エリア
# =========================================================
BASE_FONT_SIZE = 14

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE + 1,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 2,
        "ytick.labelsize": BASE_FONT_SIZE - 2,
        "legend.fontsize": BASE_FONT_SIZE - 2,
    }
)

# =========================================================
# 設定エリア
# =========================================================
TRUE_XYZ_FILE = "output.xyz"
# GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20261007_1_10n.xyz"
GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20260924_1_10n.xyz"
OUTPUT_PNG = "displacement_hist_xyz_20261007_10ns_before.png"  # 保存ファイル名

species_target = 'He'  # 対象原子種（Hの場合は 'H' に変更）

# タイムステップ（lag）
LAG_TRUE = 50
LAG_GEN = 1

# 各成分の最大変位(Å)とビン数：-MAX〜+MAX で等分割
MAX_DISP_COMPONENT = 10.0  # 成分ごとの表示範囲（-8.0 Å 〜 +8.0 Å）
NUM_BINS = 50


# =========================================================
# XYZファイルから指定ラグ(lag)の3成分変位(Δx, Δy, Δz)を取得
# =========================================================
def get_component_displacements(filepath, target_species, lag=1):
    if not os.path.exists(filepath):
        print(f"警告: ファイルが見つかりません -> {filepath}")
        return np.empty((0, 3))

    frames = []
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                n_atoms = int(line)
            except ValueError:
                break

            f.readline()  # コメント行スキップ

            coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                if parts[0] == target_species:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if len(coords) > 0:
                frames.append(coords)

    frames = np.array(frames)  # shape: (frames, n_target, 3)

    if len(frames) < lag + 1:
        print(f"エラー: {filepath} のフレーム数が不足しています。")
        return np.empty((0, 3))

    # 各フレーム間の差分 (frames - lag, n_target, 3) -> (全サンプル数, 3)
    diffs = (frames[lag:] - frames[:-lag]).reshape(-1, 3)
    print(f"[{filepath}] {target_species} 成分別変位取得数: {len(diffs)} 件")
    return diffs


# =========================================================
# メイン処理：各成分 (x, y, z) のヒストグラム描画
# =========================================================
if __name__ == "__main__":
    true_diffs = get_component_displacements(TRUE_XYZ_FILE, species_target, lag=LAG_TRUE)
    gen_diffs = get_component_displacements(GEN_XYZ_FILE, species_target, lag=LAG_GEN)

    # ダミーフォールバック
    if len(true_diffs) == 0 or len(gen_diffs) == 0:
        print("テスト用ダミーデータで実行します。")
        true_diffs = np.random.normal(loc=0.0, scale=1.2, size=(10000, 3))
        gen_diffs = np.random.normal(loc=0.0, scale=0.9, size=(10000, 3))

    # 対称なビンを作成 (-MAX 〜 +MAX)
    bins = np.linspace(-MAX_DISP_COMPONENT, MAX_DISP_COMPONENT, NUM_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    components = ["x", "y", "z"]

    # 1行3列のグラフ配置（上段のヒストグラムのみ）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for i, comp in enumerate(components):
        ax = axes[i]

        t_data = true_diffs[:, i]
        g_data = gen_diffs[:, i]

        hist_true, _ = np.histogram(t_data, bins=bins, density=True)
        hist_gen, _ = np.histogram(g_data, bins=bins, density=True)

        # ヒストグラムの描画
        ax.bar(
            bin_centers, hist_true, width=bin_width, alpha=0.5,
            label="Dataset", color="blue"
        )
        ax.bar(
            bin_centers, hist_gen, width=bin_width, alpha=0.5,
            label="Inference", color="red"
        )

        ax.set_title(f"Component: $\\Delta {comp}$")
        ax.set_xlabel(f"$\\Delta {comp}$ (Å)")
        if i == 0:
            ax.set_ylabel("Probability Density")
        ax.set_xlim(-MAX_DISP_COMPONENT, MAX_DISP_COMPONENT)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"ヒストグラム画像を {OUTPUT_PNG} に保存しました。")
    plt.close()