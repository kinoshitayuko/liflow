import pandas as pd


def create_window_csv():
    # 温度ごとのデータセット設定
    datasets = [
        {"name": "small_H2_300K_nvt", "temp": 300},
        {"name": "small_H2_450K_nvt", "temp": 450},
        {"name": "small_H2_600K_nvt", "temp": 600},
    ]

    window_size = 51     # t_end - t_start
    n_windows = 99950    # 各データセットあたりの行数
    stride = 1

    rows = []

    for dataset in datasets:
        for i in range(n_windows):
            t_start = i * stride
            t_end = t_start + window_size

            rows.append({
                "name": dataset["name"],
                "temp": dataset["temp"],
                "t_start": t_start,
                "t_end": t_end,
                "prior_Li": 1,
                "prior_frame": 0,
            })

    df = pd.DataFrame(rows)

    output_file = "train_1000ps_20260817_1.csv"
    df.to_csv(output_file, index=False)

    return df, output_file


if __name__ == "__main__":
    df, output_file = create_window_csv()

    print(f"CSVを出力しました: {output_file}")
    print(f"総行数: {len(df):,}")
    print("\n温度ごとの行数:")
    print(df.groupby(["name", "temp"]).size())

    print("\n先頭5行:")
    print(df.head())





# #CSVにdisplacementとweightの列を追加するpythonスクリプト


# import numpy as np
# import pandas as pd
# from pathlib import Path
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. 設定項目（環境に合わせて変更してください）
# # ==========================================
# CSV_FILE = "train_1000ps_20260728_1.csv"           # 読み込む対象の95000行のCSVファイル
# OUTPUT_CSV = "train_1000ps_20260728_1_weighted_3.csv" # 出力するCSVファイル名
# XYZ_DIR = Path(".")                # output.xyz が置かれているディレクトリ
# TARGET_SPECIES = "He"              # 変位を計算する対象の元素記号

# # 重み付けとヒストグラムの設定
# TOP_PERCENTILE = 0.99              # 上位何%を「大きく動いた(Rare)」とみなすか（0.90 = 上位10%）
# WEIGHT_NORMAL = 1.0                # 通常の重み
# WEIGHT_RARE = 99.0                  # 上位10%に対する重み（1:9にすることで学習時の提示割合が50:50になる）
# HISTOGRAM_FILE = "train_1000ps_20260728_1_displacement_histogram_3.png" # 保存するヒストグラムの画像ファイル名

# # ==========================================
# # 2. XYZファイルを読み込み、座標と対象原子のインデックスを返す関数
# # ==========================================
# def load_xyz_coords_and_indices(filepath, target_species):
#     """XYZファイルを読み込み、(座標配列, 対象元素のインデックスリスト) を返す"""
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
    
#     if not lines:
#         return None, []
    
#     n_atoms = int(lines[0].strip())
#     n_frames = len(lines) // (n_atoms + 2)
#     coords = np.zeros((n_frames, n_atoms, 3))
    
#     target_indices = []
    
#     for i in range(n_frames):
#         start = i * (n_atoms + 2) + 2
#         for a in range(n_atoms):
#             parts = lines[start+a].split()
            
#             # 最初のフレームを読み込む時に、元素記号を確認してインデックスを記録
#             if i == 0:
#                 species = parts[0]
#                 if species == target_species:
#                     target_indices.append(a)
                    
#             coords[i, a] = [float(parts[1]), float(parts[2]), float(parts[3])]
            
#     return coords, target_indices

# # ==========================================
# # 3. メイン処理
# # ==========================================
# def main():
#     print(f"CSVを読み込んでいます: {CSV_FILE}")
#     df = pd.read_csv(CSV_FILE)
    
#     displacement_list = np.zeros(len(df))
    
#     for name, group in df.groupby("name"):
#         #xyz_path = XYZ_DIR / f"{name}.xyz"
#         xyz_path = XYZ_DIR / f"output.xyz"
#         print(f"XYZファイルを読み込んでいます: {xyz_path}")
        
#         coords, target_indices = load_xyz_coords_and_indices(xyz_path, TARGET_SPECIES)
        
#         if coords is None:
#             raise FileNotFoundError(f"{xyz_path} が見つからないか、空です。")
#         if not target_indices:
#             raise ValueError(f"{xyz_path} の中に '{TARGET_SPECIES}' 原子が見つかりませんでした。")
        
#         print(f" -> '{TARGET_SPECIES}' 原子を {len(target_indices)} 個検出しました。")
        
#         # 各行(t_start, t_end)について変位を計算
#         for idx in group.index:
#             t_start = int(df.at[idx, "t_start"])
#             t_end = int(df.at[idx, "t_end"])
            
#             r_start = coords[t_start, target_indices]
#             r_end = coords[t_end, target_indices]
            
#             #距離計算して、平均とる
#             distances = np.linalg.norm(r_end - r_start, axis=1)
#             displacement_list[idx] = np.mean(distances)

#     # DataFrameに変位を追加
#     df["displacement"] = displacement_list
#     print("変位(displacement)の計算が完了しました。")

#     # ==========================================
#     # 4. ヒストグラムの作成と保存
#     # ==========================================
#     print("ヒストグラムを作成しています...")
#     # 上位10%の境界となる変位の値を計算
#     threshold = df["displacement"].quantile(TOP_PERCENTILE)
    
#     plt.figure(figsize=(10, 6))
#     plt.hist(df["displacement"], bins=100, color='skyblue', edgecolor='black')
    
#     # 境界線を赤の点線で引く
#     top_pct_label = round((1.0 - TOP_PERCENTILE) * 100)
#     plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, 
#                 label=f'Top {top_pct_label}% Threshold ({threshold:.2f} Å)')
    
#     plt.title(f"Displacement Histogram (Target: {TARGET_SPECIES})")
#     plt.xlabel("Displacement (Å)")
#     plt.ylabel("Counts")
#     plt.legend()
#     plt.grid(axis='y', alpha=0.75)
#     plt.savefig(HISTOGRAM_FILE, dpi=300)
#     plt.close()
#     print(f"ヒストグラムを {HISTOGRAM_FILE} に保存しました。")

#     # ==========================================
#     # 5. 上位10%に基づく重み(weight)の計算
#     # ==========================================
#     print("重み(weight)を計算しています...")
    
#     # 全てに基本の重み(1.0)を設定
#     df["weight"] = WEIGHT_NORMAL
    
#     # 境界値以上の行（上位10%）の重みを書き換え(9.0)
#     df.loc[df["displacement"] >= threshold, "weight"] = WEIGHT_RARE
    
#     # 重みの平均が1.0になるように正規化（ロスや学習率のスケールを保つため必須）
#     df["weight"] = df["weight"] / df["weight"].mean()
    
#     # ==========================================
#     # 6. 保存
#     # ==========================================
#     df.to_csv(OUTPUT_CSV, index=False)
#     print(f"完了しました！ {OUTPUT_CSV} に保存しました。")
    
#     # 簡単な統計を表示
#     num_rare = len(df[df["displacement"] >= threshold])
#     print("\n【集計結果】")
#     print(f"全サンプル数: {len(df)}")
#     print(f"閾値 ({top_pct_label}%): {threshold:.4f} Å")
#     print(f"大きく動いた(Rare)サンプル数: {num_rare} ({num_rare/len(df)*100:.1f}%)")
#     print("\n【正規化後の重みの値】")
#     print(f"通常(Normal)の重み: {df.loc[df['displacement'] < threshold, 'weight'].iloc[0]:.4f}")
#     print(f"Rareの重み       : {df.loc[df['displacement'] >= threshold, 'weight'].iloc[0]:.4f}")

# if __name__ == "__main__":
#     main()


# すでに作成済みのcsvファイルを読み込んで、ヒストグラムの画像だけを生成・保存するスクリプト

# import pandas as pd
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. 設定項目
# # ==========================================
# # displacement列が既に計算されているCSVファイルを指定してください
# INPUT_CSV = "train_1000ps_20260728_1_weighted.csv" 
# HISTOGRAM_FILE = "displacement_histogram_multi_thresholds.png"
# TARGET_SPECIES = "He"

# def main():
#     print(f"CSVを読み込んでいます: {INPUT_CSV}")
#     df = pd.read_csv(INPUT_CSV)
    
#     if "displacement" not in df.columns:
#         raise ValueError("CSVに 'displacement' 列が見つかりません。先に変位を計算したCSVを用意してください。")

#     print("ヒストグラムの画像を作成しています...")
    
#     # 描画したい上位パーセンタイルのリスト（1%, 2%, 5%, 10%, 20%）
#     top_pcts = [1, 2, 5, 10, 20]
#     # それぞれの線の色
#     colors = ['#663399', '#4169E1', '#E75480', '#D2691E', '#CD5C5C']
    
#     plt.figure(figsize=(10, 6))
    
#     # 添付図の色合いに似せた設定
#     plt.hist(df["displacement"], bins=50, color='#5F9EA0', edgecolor='white')
    
#     # 各パーセンタイルの閾値を計算して縦線を引く
#     for pct, color in zip(top_pcts, colors):
#         val = df["displacement"].quantile(1.0 - pct/100.0)
#         plt.axvline(val, color=color, linestyle='--', linewidth=2, 
#                     label=f'top {pct}% threshold = {val:.3f} A')
    
#     plt.title(f"{TARGET_SPECIES} displacement histogram and tail thresholds")
#     plt.xlabel("|dr| over 50 frames (= 0.5 ps) (A)")
#     plt.ylabel("Counts")
#     plt.legend()
#     plt.grid(axis='y', alpha=0.3)
#     plt.xlim(left=0) 
    
#     plt.tight_layout()
#     plt.savefig(HISTOGRAM_FILE, dpi=300)
#     plt.close()
    
#     print(f"完了しました！ ヒストグラムを {HISTOGRAM_FILE} に保存しました。")

# if __name__ == "__main__":
#     main()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. 設定項目
# # ==========================================
# # displacement列が既に計算されているCSVファイルを指定してください
# INPUT_CSV = "train_1000ps_20260728_1_weighted.csv" 
# HISTOGRAM_FILE = "displacement_histogram_with_stats.png"
# TARGET_SPECIES = "He"

# # 添付画像の情報に合わせるための設定
# TOTAL_FRAMES = 100001
# LAG = 50
# HE_COUNT = 2

# def main():
#     print(f"CSVを読み込んでいます: {INPUT_CSV}")
#     df = pd.read_csv(INPUT_CSV)
    
#     if "displacement" not in df.columns:
#         raise ValueError("CSVに 'displacement' 列が見つかりません。")

#     print("ヒストグラムの画像を作成しています...")
    
#     # 統計情報の計算
#     n_samples = len(df)
#     mean_val = df["displacement"].mean()
#     std_val = df["displacement"].std()
    
#     # 描画したい上位パーセンタイルのリストと色
#     top_pcts = [1, 2, 5, 10, 20]
#     colors = ['#663399', '#4169E1', '#E75480', '#D2691E', '#CD5C5C']
    
#     plt.figure(figsize=(10, 6))
    
#     # ヒストグラムの描画
#     plt.hist(df["displacement"], bins=50, color='#5F9EA0', edgecolor='white')
    
#     # 各パーセンタイルの閾値を計算して縦線を引く
#     for pct, color in zip(top_pcts, colors):
#         val = df["displacement"].quantile(1.0 - pct/100.0)
#         plt.axvline(val, color=color, linestyle='--', linewidth=2, 
#                     label=f'top {pct}% threshold = {val:.3f} A')
    
#     # ==========================================
#     # テキストボックス（統計情報）の追加
#     # ==========================================
#     # 表示するテキストの作成
#     stats_text = (
#         f"frames={TOTAL_FRAMES}, lag={LAG}\n"
#         f"{TARGET_SPECIES} count={HE_COUNT}\n"
#         f"n(samples)={n_samples} (expected {n_samples})\n"
#         f"mean={mean_val:.4f} Å\n"
#         f"std={std_val:.4f} Å"
#     )
    
#     # グラフの左上（x=0.02, y=0.95）に枠付きテキストを配置
#     # transform=plt.gca().transAxes を指定すると、グラフ内の相対座標（0.0〜1.0）で位置を決められます
#     plt.text(0.69, 0.75, stats_text, 
#              transform=plt.gca().transAxes, 
#              fontsize=10, 
#              verticalalignment='top', 
#              bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
    
#     # ==========================================
#     # グラフの装飾と保存
#     # ==========================================
#     plt.title(f"{TARGET_SPECIES} displacement histogram and tail thresholds")
#     plt.xlabel(f"|dr| over {LAG} frames (= {LAG/100:.1f} ps) (A)")
#     plt.ylabel("Counts")
    
#     # 凡例をグラフの右上枠外に置くか、右上に配置
#     plt.legend(loc='upper right')
    
#     plt.grid(axis='y', alpha=0.3)
#     plt.xlim(left=0) 
    
#     plt.tight_layout()
#     plt.savefig(HISTOGRAM_FILE, dpi=300)
#     plt.close()
    
#     print(f"完了しました！ 統計情報付きのヒストグラムを {HISTOGRAM_FILE} に保存しました。")

# if __name__ == "__main__":
#     main()