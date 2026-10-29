import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np

# =========================================================
# フォントサイズの一括設定エリア
# =========================================================
BASE_FONT_SIZE = 16  # 全体の基準フォントサイズ

plt.rcParams.update(
    {
        'font.size': BASE_FONT_SIZE,  # 全体基本サイズ
        'axes.titlesize': BASE_FONT_SIZE + 2,  # タイトル (16pt)
        'axes.labelsize': BASE_FONT_SIZE + 1,  # 軸ラベル (15pt)
        'xtick.labelsize': BASE_FONT_SIZE,  # X軸目盛り (14pt)
        'ytick.labelsize': BASE_FONT_SIZE,  # Y軸目盛り (14pt)
        'legend.fontsize': BASE_FONT_SIZE,  # 凡例 (14pt)
        'figure.titlesize': BASE_FONT_SIZE + 4,  # 図全体のタイトル
    }
)

# =========================================================
# 設定エリア
# =========================================================
TRUE_XYZ_FILE = "output.xyz"  # 真のデータ（データセット）
#GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20260805_2_10n.xyz"  # モデル生成データ
#GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20260803_14.xyz"  # モデル生成データ
GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_45_20260817_finetune_1_10n.xyz"  # モデル生成データ
OUTPUT_PNG = "histogram_1stepdisp_20260829_10ns_after.png"  # 出力画像ファイル名

species_target = 'He'  # ターゲット原子種

# 各データのタイムステップ（lag）設定
LAG_TRUE = 50  # データセット側のlag
LAG_GEN = 1  # 生成データ側のlag

# ヒストグラムの最大変位(Å)と分割数
MAX_DISPLACEMENT = 15.0
NUM_BINS = 50


# =========================================================
# XYZファイルから指定ラグ(lag)の変位とメタデータを取得する関数
# =========================================================
def get_displacements_with_info(filepath, target_species, lag=1):
    if not os.path.exists(filepath):
        print(f"警告: ファイルが見つかりません -> {filepath}")
        return np.array([]), {}

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
                species = parts[0]
                if species == target_species:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coords.append([x, y, z])

            if len(coords) > 0:
                frames.append(coords)

    frames = np.array(frames)  # shape: (n_frames, n_target, 3)

    if len(frames) < lag + 1:
        print(
            f"エラー: {filepath} には {lag + 1} フレーム以上必要です (現在: {len(frames)})。"
        )
        return np.array([]), {}

    n_frames = frames.shape[0]
    n_target = frames.shape[1]
    expected_samples = (n_frames - lag) * n_target

    diffs = frames[lag:] - frames[:-lag]
    mags = np.linalg.norm(diffs, axis=-1).flatten()

    info = {
        'n_frames': n_frames,
        'n_target': n_target,
        'expected_samples': expected_samples,
        'lag': lag,
    }

    print(
        f"[{filepath}] {target_species} (lag={lag}) : {len(mags)} 件取得 (mean={np.mean(mags):.4f} Å, std={np.std(mags):.4f} Å)"
    )
    return mags, info


# =========================================================
# メイン処理
# =========================================================
if __name__ == "__main__":
    # 1. データの読み込み
    true_data, true_info = get_displacements_with_info(
        TRUE_XYZ_FILE, species_target, lag=LAG_TRUE
    )
    gen_data, gen_info = get_displacements_with_info(
        GEN_XYZ_FILE, species_target, lag=LAG_GEN
    )

    # テスト用フォールバック
    if len(true_data) == 0 or len(gen_data) == 0:
        print(
            "データ読み込みに失敗したため、テスト用のダミーデータで描画します。"
        )
        true_data = np.random.exponential(scale=1.5, size=10000)
        gen_data = np.random.normal(loc=1.2, scale=0.8, size=10000)
        gen_data = np.abs(gen_data)
        true_info = {
            'n_frames': 250,
            'n_target': 50,
            'expected_samples': 10000,
            'lag': LAG_TRUE,
        }
        gen_info = {
            'n_frames': 10001,
            'n_target': 1,
            'expected_samples': 10000,
            'lag': LAG_GEN,
        }

    # 統計量の計算
    mu_true, std_true = np.mean(true_data), np.std(true_data)
    mu_gen, std_gen = np.mean(gen_data), np.std(gen_data)

    # 2. ヒストグラムの計算
    bins = np.linspace(0, MAX_DISPLACEMENT, NUM_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    hist_true, _ = np.histogram(true_data, bins=bins, density=True)
    hist_gen, _ = np.histogram(gen_data, bins=bins, density=True)

    # 3. グラフ描画（文字が大きくなったため画像サイズを少し拡張）
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.bar(
        bin_centers,
        hist_true,
        width=bin_width,
        alpha=0.45,
        label='Dataset',
        # color='cornflowerblue',
        # edgecolor='royalblue',
        color='blue',
        linewidth=1.0,
    )
    ax.bar(
        bin_centers,
        hist_gen,
        width=bin_width,
        alpha=0.45,
        label='Inference',
        # color='lightcoral',
        # edgecolor='crimson',
        color='red',
        linewidth=1.0,
    )

    ax.set_xlabel(r'Displacement Magnitude $|\Delta r|$ (Å)')
    ax.set_ylabel('Probability Density')
    # ax.set_title('Displacement Histogram Comparison (Species: $\mathrm{H}_2$)', pad=10)
    #ax.set_title('Baseline (10ns Rollout)', pad=10)
    ax.set_title('Fine-tuned (10ns Rollout)', pad=10)

    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, MAX_DISPLACEMENT)

    # # 統計情報テキストボックス (AnchoredText)
    # stats_text = (
    #     f"[Dataset]\n"
    #     f" mean={mu_true:.4f} Å, std={std_true:.4f} Å\n\n"
    #     f"[Inference]\n"
    #     f" mean={mu_gen:.4f} Å, std={std_gen:.4f} Å"
    # )

    # # prop に BASE_FONT_SIZE を連動させて設定
    # at = AnchoredText(
    #     stats_text,
    #     prop=dict(size=BASE_FONT_SIZE - 1.5),  # 12.5pt相当
    #     frameon=True,
    #     loc="upper right",
    #     bbox_to_anchor=(0.98, 0.81),  # 凡例サイズ拡大に合わせて配置位置を少し下に調整
    #     bbox_transform=ax.transAxes,
    #     pad=0.5,
    #     borderpad=0.0,
    # )
    # at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    # at.patch.set_alpha(0.85)
    # ax.add_artist(at)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    print(f"\nグラフを {OUTPUT_PNG} に保存しました。")
    plt.close()