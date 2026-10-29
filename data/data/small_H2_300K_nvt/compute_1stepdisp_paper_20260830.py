import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from scipy.stats import kurtosis

# =========================================================
# フォントサイズの一括設定エリア
# =========================================================
BASE_FONT_SIZE = 16  # 全体の基準フォントサイズ


plt.rcParams.update(
    {
        'font.size': BASE_FONT_SIZE,
        'axes.titlesize': BASE_FONT_SIZE + 2,
        'axes.labelsize': BASE_FONT_SIZE + 1,
        'xtick.labelsize': BASE_FONT_SIZE,
        'ytick.labelsize': BASE_FONT_SIZE,
        'legend.fontsize': BASE_FONT_SIZE,
        'figure.titlesize': BASE_FONT_SIZE + 4,
    }
)


# =========================================================
# 設定エリア
# =========================================================
TRUE_XYZ_FILE = "output.xyz"
#GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20260805_2_10n.xyz"
#GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_45_20260817_finetune_1_10n.xyz"  # モデル生成データ
#GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20261007_1.xyz"  # モデル生成データ
# GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20261007_1_10n.xyz"  # モデル生成データ
#GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20261019_propose_1.xyz"  # モデル生成データ
GEN_XYZ_FILE = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20261014_finetune_1.xyz"  # モデル生成データ
OUTPUT_PNG = "histogram_1stepdisp_20261019_after_finetune_1ns.png"

species_target = 'He'

LAG_TRUE = 50
LAG_GEN = 1

MAX_DISPLACEMENT = 15.0
NUM_BINS = 50

# サンプリング用シード（等間隔サンプリングの開始オフセットに使用）
RANDOM_SEED = 42


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

    frames = np.array(frames)

    if len(frames) < lag + 1:
        print(f"エラー: {filepath} には {lag + 1} フレーム以上必要です (現在: {len(frames)})。")
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

    print(f"[{filepath}] {target_species} (lag={lag}) : {len(mags)} 件取得 (mean={np.mean(mags):.4f} Å, std={np.std(mags):.4f} Å)")
    return mags, info


# =========================================================
# 等間隔サンプリング（開始オフセットのみシードで固定）
# =========================================================
def systematic_sample(data, n_target, seed=RANDOM_SEED):
    n_total = len(data)
    if n_total <= n_target:
        print(f"注意: 元データ数({n_total})がターゲット数({n_target})以下のため、全件を使用します。")
        return data.copy()

    step = n_total / n_target  # 実数ステップ（等間隔）
    rng = np.random.default_rng(seed)
    offset = rng.uniform(0, step)  # 開始位置だけランダム化（再現性はseedで確保）

    indices = (offset + np.arange(n_target) * step).astype(int)
    indices = np.clip(indices, 0, n_total - 1)
    # 重複が出た場合は後方にずらして重複回避（稀なケース対策）
    indices = np.unique(indices)
    while len(indices) < n_target:
        extra = indices[-1] + 1
        if extra >= n_total:
            break
        indices = np.append(indices, extra)
        indices = np.unique(indices)

    sampled = data[indices[:n_target]]
    return sampled


# =========================================================
# メイン処理
# =========================================================
if __name__ == "__main__":
    true_data, true_info = get_displacements_with_info(TRUE_XYZ_FILE, species_target, lag=LAG_TRUE)
    gen_data, gen_info = get_displacements_with_info(GEN_XYZ_FILE, species_target, lag=LAG_GEN)

    if len(true_data) == 0 or len(gen_data) == 0:
        print("データ読み込みに失敗したため、テスト用のダミーデータで描画します。")
        rng_dummy = np.random.default_rng(RANDOM_SEED)
        true_data = rng_dummy.exponential(scale=1.5, size=10000)
        gen_data = np.abs(rng_dummy.normal(loc=1.2, scale=0.8, size=2000))

    # ターゲット件数 = Inference側の件数
    n_target_samples = len(gen_data)
    n_true_total = len(true_data)

    print(f"\nDataset全件数: {n_true_total}, Inference件数(目標サンプル数): {n_target_samples}")

    true_data_sampled = systematic_sample(true_data, n_target_samples, seed=RANDOM_SEED)
    print(f"Datasetから等間隔サンプリング後: {len(true_data_sampled)} 件 (seed={RANDOM_SEED})")

    # mu_true, std_true = np.mean(true_data_sampled), np.std(true_data_sampled)
    # mu_gen, std_gen = np.mean(gen_data), np.std(gen_data)
    mu_true, std_true = np.mean(true_data_sampled), np.std(true_data_sampled)
    mu_gen, std_gen = np.mean(gen_data), np.std(gen_data)

    # # Fisherの超過尖度：
    # # 正規分布なら 0、正なら裾が重い／尖りやすい、負なら相対的に平坦
    # kurt_true = kurtosis(true_data_sampled, fisher=True, bias=False, nan_policy='omit')
    # kurt_gen = kurtosis(gen_data, fisher=True, bias=False, nan_policy='omit')

    # print(f"Dataset   : excess kurtosis={kurt_true:.4f}")
    # print(f"Inference : excess kurtosis={kurt_gen:.4f}")
    
    # 大変位イベントの頻度：
    # 物理的には、サイト間移動に対応する距離へ変更する
    TAIL_THRESHOLD = 4.0  # Å
    tail_true = np.mean(true_data_sampled > TAIL_THRESHOLD)
    tail_gen = np.mean(gen_data > TAIL_THRESHOLD)
    
    print("\n=== Distribution comparison ===")
    print(
        # f"Dataset   : mean={mean_true:.3f}, std={std_true:.3f}, "
        # f"median={median_true:.3f}, q95={q95_true:.3f}, q99={q99_true:.3f}, "
        f"P(Δr>{TAIL_THRESHOLD:.1f} Å)={tail_true:.4f}"
    )
    print(
        # f"Inference : mean={mean_gen:.3f}, std={std_gen:.3f}, "
        # f"median={median_gen:.3f}, q95={q95_gen:.3f}, q99={q99_gen:.3f}, "
        f"P(Δr>{TAIL_THRESHOLD:.1f} Å)={tail_gen:.4f}"
    )

    
    

    bins = np.linspace(0, MAX_DISPLACEMENT, NUM_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # density=False -> 実カウント数
    hist_true, _ = np.histogram(true_data_sampled, bins=bins, density=False)
    hist_gen, _ = np.histogram(gen_data, bins=bins, density=False)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.bar(bin_centers, hist_true, width=bin_width, alpha=0.45,
           label='Dataset', color='blue', linewidth=1.0)
    ax.bar(bin_centers, hist_gen, width=bin_width, alpha=0.45,
           label='Inference', color='red', linewidth=1.0)

    ax.set_xlabel(r'Displacement Magnitude $|\Delta r|$ (Å)')
    ax.set_ylabel('Count')
    #ax.set_title('Baseline (10ns Rollout)', pad=10)
    # ax.set_title('Proposed (1ns Rollout)', pad=10)
    ax.set_title('After Fine-tuning (1ns Rollout)', pad=10)

    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, MAX_DISPLACEMENT)
    
    # stats_text = (
    # "Excess kurtosis\n"
    # f"Dataset: {kurt_true:.3f}\n"
    # f"Inference: {kurt_gen:.3f}"
    # )
    
    stats_text = (
    f"P(|Δr| > {TAIL_THRESHOLD:.0f} Å)\n"
    f"Dataset   = {tail_true:.3f}\n"
    f"Inference = {tail_gen:.3f}"
    )
    
    anchored_text = AnchoredText(
        stats_text,
        loc='lower right',
        bbox_to_anchor=(0.987, 0.62),  # 凡例サイズ拡大に合わせて配置位置を少し下に調整
        bbox_transform=ax.transAxes,
        prop=dict(size=BASE_FONT_SIZE - 0),
        frameon=True,
        pad=0.5,
        borderpad=0.0,
    )

    anchored_text.patch.set_alpha(0.85)
    anchored_text.patch.set_facecolor('white')
    ax.add_artist(anchored_text)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    print(f"\nグラフを {OUTPUT_PNG} に保存しました。")
    plt.close()
