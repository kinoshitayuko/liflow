import os
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# フォントサイズの一括設定エリア
# =========================================================
BASE_FONT_SIZE = 18  # 全体の基準フォントサイズ

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": BASE_FONT_SIZE,  # 全体基本サイズ
        "axes.titlesize": BASE_FONT_SIZE + 1,  # タイトル
        "axes.labelsize": BASE_FONT_SIZE,  # 軸ラベル
        "xtick.labelsize": BASE_FONT_SIZE - 2,  # X軸目盛り
        "ytick.labelsize": BASE_FONT_SIZE - 2,  # Y軸目盛り
        "legend.fontsize": BASE_FONT_SIZE - 2,  # 凡例
        "figure.titlesize": BASE_FONT_SIZE + 3,
    }
)

# =========================================================
# 設定エリア
# =========================================================
TRUE_XYZ_FILE = "output.xyz"  # 真のデータ（データセット）
GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20260805_2_10n.xyz"  # モデルが生成したデータ
OUTPUT_PNG = "histogram_ratio_20260827_10ns.png"  # 保存する画像ファイル名

species_target = 'He'  # ターゲットにする原子種

# 各データのタイムステップ（lag）設定
LAG_TRUE = 50  # データセットは50step刻み
LAG_GEN = 1  # 生成データは1step刻み

# ヒストグラムの最大変位(Å)と分割数
MAX_DISPLACEMENT = 15.0
NUM_BINS = 50

# 重みクリッピング設定
MAX_WEIGHT = 10.0
MIN_WEIGHT = 0.1


# =========================================================
# XYZファイルから指定ラグ(lag)の変位(|Δr|)を計算する関数
# =========================================================
def get_displacements(filepath, target_species, lag=1):
    if not os.path.exists(filepath):
        print(f"警告: ファイルが見つかりません -> {filepath}")
        return np.array([])

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

            f.readline()  # コメント行をスキップ

            coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                species = parts[0]
                if species == target_species:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coords.append([x, y, z])

            if len(coords) > 0:
                frames.append(coords)

    frames = np.array(frames)  # shape: (フレーム数, ターゲット原子数, 3)

    if len(frames) < lag + 1:
        print(
            f"エラー: {filepath} には {lag + 1} フレーム以上のデータが必要です（現在: {len(frames)}）。"
        )
        return np.array([])

    diffs = frames[lag:] - frames[:-lag]
    mags = np.linalg.norm(diffs, axis=-1).flatten()

    print(
        f"[{filepath}] から {target_species} 原子の lag={lag} 変位データを {len(mags)} 件取得しました。"
    )
    return mags


# =========================================================
# メイン処理：データの読み込みと可視化
# =========================================================
if __name__ == "__main__":
    # 1. データの取得
    true_data = get_displacements(TRUE_XYZ_FILE, species_target, lag=LAG_TRUE)
    gen_data = get_displacements(GEN_XYZ_FILE, species_target, lag=LAG_GEN)

    # テスト用フォールバック
    if len(true_data) == 0 or len(gen_data) == 0:
        print(
            "データの読み込みに失敗したため、テスト用のダミーデータでグラフを描画します。"
        )
        true_data = np.random.exponential(scale=1.5, size=10000)
        gen_data = np.random.normal(loc=1.0, scale=1.0, size=10000)
        gen_data = np.abs(gen_data)

    # 2. ヒストグラムと比率の計算
    bins = np.linspace(0, MAX_DISPLACEMENT, NUM_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    hist_true, _ = np.histogram(true_data, bins=bins, density=True)
    hist_gen, _ = np.histogram(gen_data, bins=bins, density=True)

    ratio = hist_true / (hist_gen + 1e-8)
    ratio_clipped = np.clip(ratio, MIN_WEIGHT, MAX_WEIGHT)

    # 3. グラフの描画（2段レイアウト）
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1]},
    )

    # 上段：ヒストグラム
    ax1.bar(
        bin_centers,
        hist_true,
        width=bin_width,
        alpha=0.45,
        # label=f"Dataset (lag={LAG_TRUE})",
        label=f"Dataset",
        color="blue",
        # color="cornflowerblue",
        # edgecolor="royalblue",
        linewidth=1.0,
    )
    ax1.bar(
        bin_centers,
        hist_gen,
        width=bin_width,
        alpha=0.45,
        # label=f"Inference (lag={LAG_GEN})",
        label=f"Inference",
        color="red",
        #edgecolor="crimson",
        linewidth=1.0,
    )
    ax1.set_ylabel("Probability Density")
    ax1.set_title(
        rf"Displacement Histogram Comparison (10ns Rollout)",
        pad=10,
    )
    ax1.set_xlabel(r"Displacement Magnitude $|\Delta r|$ (Å)")
    ax1.set_xlim(0, MAX_DISPLACEMENT)
    ax1.tick_params(labelbottom=True)  # sharex=True でも上段に目盛り数値を表示
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 下段：比率プロット
    ax2.plot(
        bin_centers,
        ratio,
        marker="o",
        color="green",
        markersize=4.5,
        linestyle="-",
        label="Dataset / Inference Ratio",
    )
    ax2.axhline(
        y=1.0, color="black", linestyle="--", alpha=0.7, label="Ratio = 1.0"
    )
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Displacement Magnitude $|\Delta r|$ (Å)")
    ax2.set_ylabel("Ratio (True / Pred)")
    ax2.set_title("Displacement Histogram Ratio (10ns Rollout)", pad=10)
    ax2.set_xlim(0, MAX_DISPLACEMENT)
    #ax2.set_ylim(0.05, 50.0)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(pad=0.5)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"グラフを {OUTPUT_PNG} に保存しました。")
    plt.close()

    # 4. PyTorch コピペ用配列の出力
    print("\n--- 以下の配列をPyTorchのコードにコピペしてください ---")
    print("ratio_weights_np = np.array([")
    print("    " + ", ".join([f"{w:.3f}" for w in ratio_clipped]))
    print("])")
    print("--------------------------------------------------\n")