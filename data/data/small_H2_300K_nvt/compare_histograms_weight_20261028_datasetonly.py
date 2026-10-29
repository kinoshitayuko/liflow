import os
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# フォントサイズの一括設定エリア
# =========================================================
BASE_FONT_SIZE = 18

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE + 1,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 2,
        "ytick.labelsize": BASE_FONT_SIZE - 2,
        "legend.fontsize": BASE_FONT_SIZE - 2,
        "figure.titlesize": BASE_FONT_SIZE + 3,
    }
)


# =========================================================
# 設定エリア
# =========================================================
TRUE_XYZ_FILE = "output.xyz"  # TrainのXYZファイル
species_target = "He"        # 対象原子種

# 読み込んだXYZの何フレーム差で変位を計算するか
# XYZの保存間隔 × LAG_TRUE が0.5 psになるように設定
LAG_TRUE = 50

# ヒストグラムの範囲とbin数
MAX_DISPLACEMENT = 15.0  # Å
NUM_BINS = 50

# 重み設定
# 0.0 : 重み付けなし
# 0.5 : 平方根逆頻度
# 1.0 : 逆頻度
ALPHA = 1.0
EPSILON = 1e-8

# 最後のコピペ用配列だけに適用するクリッピング範囲
MIN_WEIGHT = 0.1
MAX_WEIGHT = 10.0

# 出力ファイル
# グラフ・NPZ・CSVにはクリッピング前の重みを使用
OUTPUT_PNG = "dataset_histogram_weights.png"
OUTPUT_NPZ = "dataset_distance_weights.npz"
OUTPUT_CSV = "dataset_distance_weights.csv"


# =========================================================
# 設定の検証
# =========================================================
def validate_settings():
    if not isinstance(LAG_TRUE, int) or LAG_TRUE < 1:
        raise ValueError("LAG_TRUEは1以上の整数にしてください。")

    if not isinstance(NUM_BINS, int) or NUM_BINS < 1:
        raise ValueError("NUM_BINSは1以上の整数にしてください。")

    if not np.isfinite(MAX_DISPLACEMENT) or MAX_DISPLACEMENT <= 0:
        raise ValueError("MAX_DISPLACEMENTは正の有限値にしてください。")

    if not np.isfinite(ALPHA) or ALPHA < 0:
        raise ValueError("ALPHAは0以上の有限値にしてください。")

    if not np.isfinite(EPSILON) or EPSILON <= 0:
        raise ValueError("EPSILONは正の有限値にしてください。")

    if (
        not np.isfinite(MIN_WEIGHT)
        or not np.isfinite(MAX_WEIGHT)
        or MIN_WEIGHT <= 0
        or MAX_WEIGHT < MIN_WEIGHT
    ):
        raise ValueError(
            "重みの範囲は 0 < MIN_WEIGHT <= MAX_WEIGHT にしてください。"
        )


# =========================================================
# XYZから対象原子の変位の大きさを計算する
# =========================================================
def get_displacements(filepath, target_species, lag=1):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"XYZファイルが見つかりません: {filepath}"
        )

    frames = []
    reference_species = None
    frame_index = 0

    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()

            if not line:
                break

            if not line.strip():
                continue

            frame_index += 1

            try:
                n_atoms = int(line.strip())
            except ValueError as exc:
                raise ValueError(
                    f"フレーム{frame_index}: 原子数を読み取れません。"
                ) from exc

            if n_atoms <= 0:
                raise ValueError(
                    f"フレーム{frame_index}: "
                    "原子数は正である必要があります。"
                )

            # XYZのコメント行
            if f.readline() == "":
                raise ValueError(
                    f"フレーム{frame_index}: コメント行がありません。"
                )

            frame_species = []
            target_coords = []

            for atom_index in range(n_atoms):
                parts = f.readline().split()

                if len(parts) < 4:
                    raise ValueError(
                        f"フレーム{frame_index}, 原子{atom_index + 1}: "
                        "原子種とxyz座標を読み取れません。"
                    )

                species = parts[0]
                frame_species.append(species)

                if species == target_species:
                    try:
                        xyz = [
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        ]
                    except ValueError as exc:
                        raise ValueError(
                            f"フレーム{frame_index}, 原子{atom_index + 1}: "
                            "座標を数値に変換できません。"
                        ) from exc

                    target_coords.append(xyz)

            if reference_species is None:
                reference_species = frame_species
            elif frame_species != reference_species:
                raise ValueError(
                    f"フレーム{frame_index}: "
                    "原子数または原子種の並びが"
                    "最初のフレームと異なります。"
                )

            if not target_coords:
                raise ValueError(
                    f"フレーム{frame_index}: "
                    f"対象原子種 '{target_species}' がありません。"
                )

            frames.append(target_coords)

    if len(frames) < lag + 1:
        raise ValueError(
            f"lag={lag}の変位計算には{lag + 1}フレーム以上必要です。"
            f"現在のフレーム数: {len(frames)}"
        )

    frames = np.asarray(frames, dtype=np.float64)

    if not np.all(np.isfinite(frames)):
        raise ValueError(
            "対象原子の座標にNaNまたはInfが含まれています。"
        )

    # 周期境界を考慮して正解変位を計算できる座標を使用
    # この関数自体ではunwrap処理を行わない
    diffs = frames[lag:] - frames[:-lag]

    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(diffs, axis=-1).reshape(-1)

    if not np.all(np.isfinite(distances)):
        raise ValueError(
            "計算した距離にNaNまたはInfが含まれています。"
        )

    print(f"入力ファイル       : {filepath}")
    print(f"対象原子種         : {target_species}")
    print(f"フレーム数         : {frames.shape[0]}")
    print(f"対象原子数/frame   : {frames.shape[1]}")
    print(f"lag                : {lag}")
    print(f"変位の総件数       : {distances.size}")
    print(f"最小変位           : {distances.min():.6f} Å")
    print(f"最大変位           : {distances.max():.6f} Å")

    return distances


# =========================================================
# ヒストグラムとbinごとの重みを計算する
# ここではクリッピングしない
# =========================================================
def calculate_bin_weights(distances, bin_edges, alpha, epsilon):
    if distances.size == 0:
        raise ValueError("変位データが空です。")

    # 全Trainデータを重み計算に含めるため、範囲外はエラーにする
    outside = (
        (distances < bin_edges[0])
        | (distances > bin_edges[-1])
    )

    if np.any(outside):
        raise ValueError(
            f"ヒストグラム範囲外の変位が"
            f"{np.count_nonzero(outside)}件あります。\n"
            f"最大変位は{distances.max():.6f} Åです。\n"
            "MAX_DISPLACEMENTを大きくしてください。"
        )

    counts, _ = np.histogram(distances, bins=bin_edges)
    total_count = distances.size

    if counts.sum() != total_count:
        raise RuntimeError(
            "ヒストグラムの合計件数が総件数と一致しません。"
        )

    # p_b = n_b / N
    probabilities = counts.astype(np.float64) / total_count

    # 描画用の確率密度
    bin_widths = np.diff(bin_edges)
    density = probabilities / bin_widths

    occupied = counts > 0

    # 正規化前の重み
    # tilde_w_b = 1 / (p_b + epsilon)^alpha
    with np.errstate(
        over="ignore",
        divide="ignore",
        invalid="ignore",
    ):
        raw_weights = (probabilities + epsilon) ** (-alpha)

    if not np.all(np.isfinite(raw_weights)):
        raise ValueError(
            "重み計算でNaNまたはInfが発生しました。"
            "ALPHAとEPSILONを見直してください。"
        )

    # Train全体における平均重み
    # Z = sum_b p_b * tilde_w_b
    normalizer = np.sum(
        probabilities[occupied] * raw_weights[occupied]
    )

    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("重みの正規化係数が不正です。")

    # 平均1に正規化した重み（クリッピング前）
    weights = raw_weights / normalizer

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "正規化後の重みにNaNまたはInfがあります。"
        )

    mean_weight = np.sum(
        probabilities[occupied] * weights[occupied]
    )

    if not np.isclose(mean_weight, 1.0):
        raise RuntimeError(
            f"平均重みが1になっていません: {mean_weight}"
        )

    return (
        counts,
        probabilities,
        density,
        raw_weights,
        weights,
        occupied,
        normalizer,
        mean_weight,
    )


# =========================================================
# Datasetヒストグラムとクリッピング前の重みを描画する
# =========================================================
def plot_histogram_and_weights(
    bin_edges,
    density,
    weights,
    occupied,
    alpha,
    output_png,
):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1]},
    )

    # 上段：Datasetのみのヒストグラム
    ax1.bar(
        bin_centers,
        density,
        width=bin_widths,
        alpha=0.45,
        label="Dataset",
        color="blue",
        linewidth=1.0,
    )

    ax1.set_ylabel("Probability Density")
    ax1.set_title("Dataset Displacement Histogram", pad=10)
    ax1.set_xlabel(r"Displacement Magnitude $|\Delta r|$ (Å)")
    ax1.set_xlim(bin_edges[0], bin_edges[-1])
    ax1.tick_params(labelbottom=True)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 下段：クリッピング前の正規化済み重み
    # 空binはNaNとして非表示にする
    plotted_weights = np.where(occupied, weights, np.nan)

    ax2.stairs(
        plotted_weights,
        bin_edges,
        baseline=None,
        color="green",
        linewidth=1.5,
        label=rf"Bin Weight ($\alpha={alpha:g}$)",
    )

    ax2.plot(
        bin_centers[occupied],
        weights[occupied],
        linestyle="none",
        marker="o",
        markersize=4.5,
        color="green",
    )

    ax2.axhline(
        y=1.0,
        color="black",
        linestyle="--",
        alpha=0.7,
        label="Weight = 1",
    )

    ax2.set_yscale("log")
    ax2.set_xlabel(r"Displacement Magnitude $|\Delta r|$ (Å)")
    ax2.set_ylabel(r"Weight $w_b$")
    ax2.set_title(
        rf"Normalized Frequency-Based Weights ($\alpha={alpha:g}$)",
        pad=10,
    )
    ax2.set_xlim(bin_edges[0], bin_edges[-1])
    ax2.legend(loc="best", frameon=True)
    ax2.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5,
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)

    print(f"\nグラフを保存しました: {output_png}")
    print("※ グラフはクリッピング前の重みです。")


# =========================================================
# クリッピング前の重みテーブルを保存する
# =========================================================
def save_weight_table(
    bin_edges,
    counts,
    probabilities,
    density,
    raw_weights,
    weights,
    occupied,
):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # クリッピング前の正規化済み重みを保存
    np.savez(
        OUTPUT_NPZ,
        bin_edges=bin_edges,
        bin_weights=weights,
        bin_counts=counts,
        bin_probabilities=probabilities,
        occupied_bins=occupied,
        alpha=ALPHA,
        epsilon=EPSILON,
        lag=LAG_TRUE,
        species=species_target,
        source_file=TRUE_XYZ_FILE,
    )

    table = np.column_stack(
        [
            bin_edges[:-1],
            bin_edges[1:],
            bin_centers,
            counts,
            probabilities,
            density,
            raw_weights,
            weights,
            occupied.astype(int),
        ]
    )

    np.savetxt(
        OUTPUT_CSV,
        table,
        delimiter=",",
        header=(
            "bin_left_A,bin_right_A,bin_center_A,"
            "count,probability,probability_density,"
            "raw_weight,normalized_weight,occupied"
        ),
        comments="",
        fmt=[
            "%.10g",
            "%.10g",
            "%.10g",
            "%d",
            "%.12g",
            "%.12g",
            "%.12g",
            "%.12g",
            "%d",
        ],
    )

    print(f"重みテーブルを保存しました: {OUTPUT_NPZ}")
    print(f"確認用CSVを保存しました   : {OUTPUT_CSV}")
    print("※ NPZ・CSVもクリッピング前の重みです。")


# =========================================================
# コピペ用配列を表示する
# =========================================================
def print_numpy_array(name, array):
    values = ", ".join(f"{value:.12g}" for value in array)

    print(f"{name} = np.array([")
    print(f"    {values}")
    print("], dtype=np.float64)")


# =========================================================
# メイン処理
# =========================================================
if __name__ == "__main__":
    validate_settings()

    # 1. Trainの正解変位から距離を取得
    true_data = get_displacements(
        TRUE_XYZ_FILE,
        species_target,
        lag=LAG_TRUE,
    )

    # 2. bin境界を作成
    bins = np.linspace(
        0.0,
        MAX_DISPLACEMENT,
        NUM_BINS + 1,
    )

    # 3. ヒストグラムと正規化済み重みを計算
    (
        counts,
        probabilities,
        density,
        raw_weights,
        weights,
        occupied,
        normalizer,
        mean_weight,
    ) = calculate_bin_weights(
        true_data,
        bins,
        alpha=ALPHA,
        epsilon=EPSILON,
    )

    print("\n--- 重み計算の確認：クリッピング前 ---")
    print(f"ALPHA                   : {ALPHA}")
    print(f"EPSILON                 : {EPSILON}")
    print(f"総件数 N                : {counts.sum()}")
    print(f"全bin数                 : {NUM_BINS}")
    print(f"データのあるbin数       : {np.count_nonzero(occupied)}")
    print(f"空bin数                 : {np.count_nonzero(~occupied)}")
    print(f"正規化前の平均重み Z    : {normalizer:.12g}")
    print(f"正規化後の平均重み      : {mean_weight:.12f}")
    print(f"使用binの最小重み       : {weights[occupied].min():.12g}")
    print(f"使用binの最大重み       : {weights[occupied].max():.12g}")

    # 4. クリッピング前の重みを描画
    plot_histogram_and_weights(
        bins,
        density,
        weights,
        occupied,
        ALPHA,
        OUTPUT_PNG,
    )

    # 5. クリッピング前の重みを保存
    save_weight_table(
        bins,
        counts,
        probabilities,
        density,
        raw_weights,
        weights,
        occupied,
    )

    # 6. 最後のコピペ用配列だけクリッピング
    # 元のweightsは変更しない
    # クリッピング後の再正規化は行わない
    weights_for_copy = np.clip(
        weights,
        MIN_WEIGHT,
        MAX_WEIGHT,
    )

    mean_weight_clipped = np.sum(
        probabilities[occupied] * weights_for_copy[occupied]
    )

    # 実データがあるbinに対するクリッピング件数
    clipped_low = occupied & (weights < MIN_WEIGHT)
    clipped_high = occupied & (weights > MAX_WEIGHT)

    print("\n--- コピペ用重みの確認：クリッピング後 ---")
    print(f"クリッピング範囲        : [{MIN_WEIGHT}, {MAX_WEIGHT}]")
    print(f"Train平均重み           : {mean_weight_clipped:.12f}")
    print(
        f"使用binの最小重み       : "
        f"{weights_for_copy[occupied].min():.12g}"
    )
    print(
        f"使用binの最大重み       : "
        f"{weights_for_copy[occupied].max():.12g}"
    )
    print(f"下限でクリップしたbin数 : {np.count_nonzero(clipped_low)}")
    print(f"上限でクリップしたbin数 : {np.count_nonzero(clipped_high)}")
    print(f"下限でクリップした件数  : {counts[clipped_low].sum()}")
    print(f"上限でクリップした件数  : {counts[clipped_high].sum()}")
    print("※ クリッピング後の再正規化は行っていません。")

    # 7. クリッピング済みのコピペ用配列を出力
    print("\n--- 以下を学習コードにコピペしてください ---")
    print_numpy_array("bin_edges_np", bins)
    print()
    print_numpy_array("bin_weights_np", weights_for_copy)

    print(
        "\n※ 上のbin_weights_npだけがクリッピング済みです。"
        "\n※ 空binの重みもクリッピングして配列に含めています。"
        "\n※ グラフ・NPZ・CSVはクリッピング前のままです。"
    )