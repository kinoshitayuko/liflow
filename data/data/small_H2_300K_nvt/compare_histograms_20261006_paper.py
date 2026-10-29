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
        "legend.fontsize": BASE_FONT_SIZE - 3,
    }
)

# =========================================================
# 設定エリア
# =========================================================
TRUE_XYZ_FILE = "output.xyz"
GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20260924_1.xyz"
OUTPUT_PNG = "histogram_ratio_xyz_20261006.png"

species_target = 'He'  # 対象原子種（Hの場合は 'H' に変更）

# タイムステップ（lag）
LAG_TRUE = 50
LAG_GEN = 1

# 各成分の最大変位(Å)とビン数：-MAX〜+MAX で等分割
MAX_DISP_COMPONENT = 8.0  # 成分ごとの最大値（-8.0 Å 〜 +8.0 Å）
NUM_BINS = 50

# 重みクリッピング設定
MAX_WEIGHT = 10.0
MIN_WEIGHT = 0.1


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
# メイン処理：各成分 (x, y, z) の計算と描画
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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # 2行3列のグラフ配置（上段: 確率密度, 下段: 密度比）
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
    )

    ratio_dict = {}

    for i, comp in enumerate(components):
        ax_hist = axes[0, i]
        ax_ratio = axes[1, i]

        t_data = true_diffs[:, i]
        g_data = gen_diffs[:, i]

        hist_true, _ = np.histogram(t_data, bins=bins, density=True)
        hist_gen, _ = np.histogram(g_data, bins=bins, density=True)

        ratio = hist_true / (hist_gen + 1e-8)
        ratio_clipped = np.clip(ratio, MIN_WEIGHT, MAX_WEIGHT)
        ratio_dict[comp] = ratio_clipped

        # 上段：確率密度
        ax_hist.bar(
            bin_centers, hist_true, width=bin_width, alpha=0.45,
            label="Dataset", color="blue"
        )
        ax_hist.bar(
            bin_centers, hist_gen, width=bin_width, alpha=0.45,
            label="Inference", color="red"
        )
        ax_hist.set_title(f"Component: $\\Delta {comp}$")
        ax_hist.set_ylabel("Probability Density" if i == 0 else "")
        ax_hist.set_xlim(-MAX_DISP_COMPONENT, MAX_DISP_COMPONENT)
        ax_hist.grid(True, linestyle="--", alpha=0.4)
        ax_hist.legend(loc="upper right")

        # 下段：密度比
        ax_ratio.plot(bin_centers, ratio, marker="o", markersize=3.5, color=colors[i], label=f"Ratio ({comp})")
        ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.7)
        ax_ratio.set_yscale("log")
        ax_ratio.set_xlabel(f"$\\Delta {comp}$ (Å)")
        ax_ratio.set_ylabel("Ratio (True / Gen)" if i == 0 else "")
        ax_ratio.set_xlim(-MAX_DISP_COMPONENT, MAX_DISP_COMPONENT)
        ax_ratio.grid(True, linestyle="--", alpha=0.4)
        ax_ratio.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"グラフを {OUTPUT_PNG} に保存しました。")
    plt.close()

    # 4. 配列の出力
    print("\n# =========================================================")
    print(f"HIST_BINS_NP = np.linspace(-{MAX_DISP_COMPONENT}, {MAX_DISP_COMPONENT}, {NUM_BINS + 1})")
    for comp in components:
        formatted = ", ".join([f"{w:.3f}" for w in ratio_dict[comp]])
        print(f"HIST_WEIGHTS_{comp.upper()}_NP = np.array([{formatted}])")
    print("# =========================================================\n")