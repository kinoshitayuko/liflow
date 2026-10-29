import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from scipy.stats import maxwell


# =========================================================
# フォントサイズの一括設定
# =========================================================
BASE_FONT_SIZE = 16

plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE + 2,
        "axes.labelsize": BASE_FONT_SIZE + 1,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
        "figure.titlesize": BASE_FONT_SIZE + 4,
    }
)


# =========================================================
# ファイル・解析設定
# =========================================================
TRUE_XYZ_FILE = "output.xyz"

GEN_XYZ_FILE = (
    "../../../scripts/traj_output_TiO2SiO2_H2/"
    "traj_300K_42_20261019_propose_1.xyz"
    # "traj_300K_42_20261014_finetune_1.xyz"
)

OUTPUT_PNG = "histogram_1stepdisp_20261026_density_mixture_maxwell.png"

SPECIES_TARGET = "He"

# 両方が同じ実時間幅になるように指定してください。
# 例：Datasetの保存間隔が0.01 psなら、50フレームで0.5 ps。
#     Inferenceの保存間隔が0.5 psなら、1フレームで0.5 ps。
LAG_TRUE = 50
LAG_GEN = 1

MAX_DISPLACEMENT = 15.0
NUM_BINS = 50

TAIL_THRESHOLD = 4.0  # Å

FIGURE_TITLE = "Displacement magnitude distribution (0.5 ps)"

# 1次元の変位成分にフィットした値。再フィットせず固定します。
MIXTURE_WEIGHTS = np.array([0.7023, 0.2977], dtype=float)
MIXTURE_SIGMAS = np.array([0.9306, 3.0208], dtype=float)  # Å

N_CURVE_POINTS = 2000


# =========================================================
# XYZファイルから変位の大きさとメタデータを取得
#
# 前提：
# ・各フレームで原子順序が固定されていること
# ・先頭4列が species, x, y, z であること
# ・変位を直接引き算できるunwrapped座標であること
# =========================================================
def get_displacements_with_info(filepath, target_species, lag=1):
    if not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError("lagは1以上の整数にしてください。")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"ファイルが見つかりません: {filepath}"
        )

    frames = []
    reference_species = None
    target_mask = None
    frame_index = 0

    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()

            if not line:
                break

            line = line.strip()

            if not line:
                continue

            try:
                n_atoms = int(line)
            except ValueError as exc:
                raise ValueError(
                    f"{filepath}: フレーム{frame_index}の"
                    f"原子数を読み取れません: {line!r}"
                ) from exc

            if n_atoms <= 0:
                raise ValueError(
                    f"{filepath}: 原子数が不正です: {n_atoms}"
                )

            comment = f.readline()

            if comment == "":
                raise ValueError(
                    f"{filepath}: フレーム{frame_index}の"
                    "コメント行がありません。"
                )

            species_list = []
            coords = []

            for atom_index in range(n_atoms):
                atom_line = f.readline()

                if atom_line == "":
                    raise ValueError(
                        f"{filepath}: フレーム{frame_index}の"
                        f"原子{atom_index}でファイルが終了しました。"
                    )

                parts = atom_line.split()

                if len(parts) < 4:
                    raise ValueError(
                        f"{filepath}: フレーム{frame_index}の"
                        f"原子{atom_index}の列数が不足しています。"
                    )

                species_list.append(parts[0])

                try:
                    xyz = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
                except ValueError as exc:
                    raise ValueError(
                        f"{filepath}: フレーム{frame_index}の"
                        f"原子{atom_index}の座標を読み取れません。"
                    ) from exc

                coords.append(xyz)

            coords = np.asarray(coords, dtype=float)

            if not np.all(np.isfinite(coords)):
                raise ValueError(
                    f"{filepath}: フレーム{frame_index}に"
                    "NaNまたは無限大の座標があります。"
                )

            if reference_species is None:
                reference_species = species_list
                target_mask = np.array(
                    [s == target_species for s in species_list],
                    dtype=bool,
                )

                if not np.any(target_mask):
                    raise ValueError(
                        f"{filepath}: 対象元素"
                        f"'{target_species}'が見つかりません。"
                    )

            elif species_list != reference_species:
                raise ValueError(
                    f"{filepath}: フレーム{frame_index}で"
                    "原子数または元素の並びが変化しています。"
                )

            frames.append(coords[target_mask])
            frame_index += 1

    if len(frames) < lag + 1:
        raise ValueError(
            f"{filepath}: lag={lag}の計算には"
            f"{lag + 1}フレーム以上必要です。"
            f"現在は{len(frames)}フレームです。"
        )

    frames = np.asarray(frames, dtype=float)

    n_frames = frames.shape[0]
    n_target = frames.shape[1]

    diffs = frames[lag:] - frames[:-lag]
    mags = np.linalg.norm(diffs, axis=-1).ravel()

    if not np.all(np.isfinite(mags)):
        raise ValueError(
            f"{filepath}: 計算した変位にNaNまたは無限大があります。"
        )

    info = {
        "n_frames": n_frames,
        "n_target": n_target,
        "expected_samples": (n_frames - lag) * n_target,
        "lag": lag,
        "n_samples": len(mags),
    }

    print(
        f"[{filepath}]\n"
        f"  species={target_species}, "
        f"frames={n_frames}, atoms={n_target}, lag={lag}\n"
        f"  samples={len(mags)}\n"
        f"  mean={np.mean(mags):.4f} Å, "
        f"std={np.std(mags):.4f} Å"
    )

    return mags, info


# =========================================================
# 混合分布パラメータの検証
# =========================================================
def validate_mixture_parameters(weights, sigmas):
    weights = np.asarray(weights, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    if weights.ndim != 1 or sigmas.ndim != 1:
        raise ValueError("weightsとsigmasは1次元配列にしてください。")

    if len(weights) != len(sigmas) or len(weights) == 0:
        raise ValueError("weightsとsigmasの長さが一致していません。")

    if not np.all(np.isfinite(weights)):
        raise ValueError("weightsに不正な値があります。")

    if not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmasに不正な値があります。")

    if np.any(weights < 0):
        raise ValueError("混合重みは0以上にしてください。")

    if np.any(sigmas <= 0):
        raise ValueError("標準偏差は正の値にしてください。")

    if not np.isclose(np.sum(weights), 1.0, rtol=0, atol=1e-8):
        raise ValueError(
            f"混合重みの合計が1ではありません: {np.sum(weights)}"
        )

    return weights, sigmas


# =========================================================
# 混合マクスウェル分布の確率密度
#
# 原子ごとに混合成分を1つ選び、そのsigmaをxyzで共有する
# 等方的な3次元混合ガウスモデルに対応します。
# =========================================================
def mixture_maxwell_pdf(r, weights, sigmas):
    r = np.asarray(r, dtype=float)
    pdf = np.zeros_like(r, dtype=float)

    for weight, sigma in zip(weights, sigmas):
        pdf += weight * maxwell.pdf(
            r,
            loc=0.0,
            scale=sigma,
        )

    return pdf


# =========================================================
# 混合マクスウェル分布の閾値超過確率
# =========================================================
def mixture_maxwell_tail_probability(threshold, weights, sigmas):
    return float(
        sum(
            weight * maxwell.sf(threshold, loc=0.0, scale=sigma)
            for weight, sigma in zip(weights, sigmas)
        )
    )


# =========================================================
# ヒストグラムの確率密度
#
# 分母には表示範囲外を含む全データ数を使用します。
# 表示範囲が有限でも、範囲内だけで再正規化しません。
# =========================================================
def histogram_density(data, bins):
    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError("ヒストグラムを作るデータが空です。")

    counts, edges = np.histogram(
        data,
        bins=bins,
        density=False,
    )

    widths = np.diff(edges)
    density = counts / (data.size * widths)

    return density, edges


# =========================================================
# メイン処理
# =========================================================
def main():
    weights, sigmas = validate_mixture_parameters(
        MIXTURE_WEIGHTS,
        MIXTURE_SIGMAS,
    )

    if MAX_DISPLACEMENT <= 0:
        raise ValueError("MAX_DISPLACEMENTは正の値にしてください。")

    if NUM_BINS < 1:
        raise ValueError("NUM_BINSは1以上にしてください。")

    if TAIL_THRESHOLD < 0:
        raise ValueError("TAIL_THRESHOLDは0以上にしてください。")

    true_data, true_info = get_displacements_with_info(
        TRUE_XYZ_FILE,
        SPECIES_TARGET,
        lag=LAG_TRUE,
    )

    gen_data, gen_info = get_displacements_with_info(
        GEN_XYZ_FILE,
        SPECIES_TARGET,
        lag=LAG_GEN,
    )

    # 件数を揃えるためのサンプリングは行わず、全件を使用します。
    mu_true = np.mean(true_data)
    std_true = np.std(true_data)

    mu_gen = np.mean(gen_data)
    std_gen = np.std(gen_data)

    tail_true = np.mean(true_data > TAIL_THRESHOLD)
    tail_gen = np.mean(gen_data > TAIL_THRESHOLD)

    tail_mixture = mixture_maxwell_tail_probability(
        TAIL_THRESHOLD,
        weights,
        sigmas,
    )

    outside_true = np.mean(true_data > MAX_DISPLACEMENT)
    outside_gen = np.mean(gen_data > MAX_DISPLACEMENT)

    outside_mixture = mixture_maxwell_tail_probability(
        MAX_DISPLACEMENT,
        weights,
        sigmas,
    )

    print("\n=== Distribution comparison ===")

    print(
        f"Dataset   : N={len(true_data)}, "
        f"mean={mu_true:.4f} Å, std={std_true:.4f} Å, "
        f"P(r>{TAIL_THRESHOLD:.1f} Å)={tail_true:.6f}"
    )

    print(
        f"Inference : N={len(gen_data)}, "
        f"mean={mu_gen:.4f} Å, std={std_gen:.4f} Å, "
        f"P(r>{TAIL_THRESHOLD:.1f} Å)={tail_gen:.6f}"
    )

    print(
        f"Mixture   : "
        f"P(r>{TAIL_THRESHOLD:.1f} Å)={tail_mixture:.6f}"
    )

    print("\n=== Fixed mixture parameters ===")

    for i, (weight, sigma) in enumerate(zip(weights, sigmas), start=1):
        print(
            f"Component {i}: "
            f"weight={weight:.4f}, sigma={sigma:.4f} Å"
        )

    print(
        f"\n=== Probability outside plotting range "
        f"(r>{MAX_DISPLACEMENT:.1f} Å) ==="
    )
    print(f"Dataset   : {outside_true:.8f}")
    print(f"Inference : {outside_gen:.8f}")
    print(f"Mixture   : {outside_mixture:.8f}")

    bins = np.linspace(
        0.0,
        MAX_DISPLACEMENT,
        NUM_BINS + 1,
    )

    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_widths = np.diff(bins)

    density_true, _ = histogram_density(true_data, bins)
    density_gen, _ = histogram_density(gen_data, bins)

    r_curve = np.linspace(
        0.0,
        MAX_DISPLACEMENT,
        N_CURVE_POINTS,
    )

    pdf_mixture = mixture_maxwell_pdf(
        r_curve,
        weights,
        sigmas,
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.bar(
        bin_centers,
        density_true,
        width=bin_widths,
        alpha=0.45,
        label="Dataset",
        color="blue",
        linewidth=1.0,
        zorder=2,
    )

    ax.bar(
        bin_centers,
        density_gen,
        width=bin_widths,
        alpha=0.45,
        label="Inference",
        color="red",
        linewidth=1.0,
        zorder=3,
    )

    ax.plot(
        r_curve,
        pdf_mixture,
        color="green",
        linewidth=2.5,
        label="Mixture Maxwell (from fitted prior)",
        zorder=4,
    )

    ax.set_xlabel(
        r"Displacement magnitude $|\Delta \mathbf{r}|$ (Å)"
    )
    ax.set_ylabel(
        r"Probability density (Å$^{-1}$)"
    )
    ax.set_title(
        FIGURE_TITLE,
        pad=10,
    )

    ax.set_xlim(0, MAX_DISPLACEMENT)
    ax.set_ylim(bottom=0)

    ax.set_axisbelow(True)
    ax.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        prop={"size": BASE_FONT_SIZE - 2},
    )

    stats_text = (
        f"P(r > {TAIL_THRESHOLD:g} Å)\n"
        f"Dataset   = {tail_true:.3f}\n"
        f"Inference = {tail_gen:.3f}\n"
        f"Mixture   = {tail_mixture:.3f}"
    )

    anchored_text = AnchoredText(
        stats_text,
        loc="upper right",
        bbox_to_anchor=(0.987, 0.70),
        bbox_transform=ax.transAxes,
        prop={"size": BASE_FONT_SIZE - 1},
        frameon=True,
        pad=0.5,
        borderpad=0.0,
    )

    anchored_text.patch.set_alpha(0.85)
    anchored_text.patch.set_facecolor("white")

    ax.add_artist(anchored_text)

    fig.tight_layout()

    output_dir = os.path.dirname(OUTPUT_PNG)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nグラフを {OUTPUT_PNG} に保存しました。")


if __name__ == "__main__":
    main()