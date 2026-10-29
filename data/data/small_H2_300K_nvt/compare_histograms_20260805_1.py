import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# 設定エリア
# =========================================================
TRUE_XYZ_FILE = "output.xyz"  # 真のデータ（データセット）
GEN_XYZ_FILE  = "../../../scripts/traj_output_TiO2SiO2_H2/traj_300K_42_20260805_2_10n.xyz" # モデルが生成したデータ
OUTPUT_PNG    = "histogram_ratio_20260828_10ns.png" # 保存する画像ファイル名

species_target = 'He'  # ターゲットにする原子種

# 各データのタイムステップ（lag）設定
LAG_TRUE = 50  # データセットは50step刻み
LAG_GEN  = 1   # 生成データは1step刻み

# ヒストグラムの最大変位(Å)と分割数
MAX_DISPLACEMENT = 15.0
NUM_BINS = 50


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
                break # ファイル終端
            
            line = line.strip()
            if not line:
                continue
                
            try:
                n_atoms = int(line)
            except ValueError:
                break
                
            f.readline() # コメント行を読み飛ばす
            
            coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                species = parts[0]
                if species == target_species:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coords.append([x, y, z])
                    
            if len(coords) > 0:
                frames.append(np.array(coords))

    frames = np.array(frames) # shape: (フレーム数, ターゲット原子数, 3)
    
    if len(frames) < lag + 1:
        print(f"エラー: {filepath} には {lag + 1} フレーム以上のデータが必要です（現在のフレーム数: {len(frames)}）。")
        return np.array([])
        
    # lag ステップ離れたフレーム間の変位ベクトルを計算
    # (例: lag=50 の場合、フレーム50 - フレーム0, フレーム51 - フレーム1...)
    diffs = frames[lag:] - frames[:-lag] 
    
    # ベクトルの大きさ（ノルム）を計算して平坦化
    mags = np.linalg.norm(diffs, axis=-1).flatten()
    
    print(f"[{filepath}] から {target_species} 原子の lag={lag} 変位データを {len(mags)} 件取得しました。")
    return mags


# =========================================================
# メイン処理：データの読み込みと可視化
# =========================================================
if __name__ == "__main__":
    # 1. データの取得（それぞれのlagを指定）
    true_data = get_displacements(TRUE_XYZ_FILE, species_target, lag=LAG_TRUE)
    gen_data  = get_displacements(GEN_XYZ_FILE, species_target, lag=LAG_GEN)
    
    # テスト用フォールバック
    if len(true_data) == 0 or len(gen_data) == 0:
        print("データの読み込みに失敗したため、テスト用のダミーデータでグラフを描画します。")
        true_data = np.random.exponential(scale=1.5, size=10000)
        gen_data = np.random.normal(loc=1.0, scale=1.0, size=10000)

    # 2. ヒストグラムと比率の計算
    bins = np.linspace(0, MAX_DISPLACEMENT, NUM_BINS + 1)
    
    hist_true, _ = np.histogram(true_data, bins=bins, density=True)
    hist_gen,  _ = np.histogram(gen_data, bins=bins, density=True)
    
    ratio = hist_true / (hist_gen + 1e-8)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # 3. グラフの描画
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上段：ヒストグラム
    ax1.bar(bin_centers, hist_true, width=bin_width, alpha=0.5, label=f'True Data (lag={LAG_TRUE})', color='blue')
    ax1.bar(bin_centers, hist_gen, width=bin_width, alpha=0.5, label=f'Generated Data (lag={LAG_GEN})', color='red')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'Displacement Histogram Comparison (Species: {species_target})')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 下段：比率
    ax2.plot(bin_centers, ratio, marker='o', color='green', markersize=4, linestyle='-')
    ax2.axhline(y=1.0, color='black', linestyle='--', label='Ratio = 1.0 (Match)')
    ax2.set_yscale('log') # 対数スケール化
    ax2.set_xlabel('Displacement Magnitude |Δr| (Å)')
    ax2.set_ylabel('Ratio (True / Gen)')
    ax2.set_title('Ratio (Next epoch learning weight)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 縦軸の上限設定（10倍程度まで見えれば十分）
    #ax2.set_ylim(0, min(10.0, np.max(ratio) * 1.1))

    plt.tight_layout()
    #plt.show()
    # PNGファイルとして保存
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    print(f"グラフを {OUTPUT_PNG} に保存しました。")
    plt.close()

    # （さっきのスクリプトの最後に追加）

    # 上限を 10.0、下限を 0.1 に制限する
    MAX_WEIGHT = 10.0
    MIN_WEIGHT = 0.1
    ratio_clipped = np.clip(ratio, MIN_WEIGHT, MAX_WEIGHT)

    # PyTorchのコードにコピペしやすい形で出力する
    print("\n--- 以下の配列をPyTorchのコードにコピペしてください ---")
    print("ratio_weights_np = np.array([")
    print("    " + ", ".join([f"{w:.3f}" for w in ratio_clipped]))
    print("])")
    print("--------------------------------------------------\n")