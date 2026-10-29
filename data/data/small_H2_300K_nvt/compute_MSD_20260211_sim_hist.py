# # -*- coding: utf-8 -*-
# import re
# from collections import deque
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.offsetbox import AnchoredText


# # -------- user parameters --------
# input_path = "output.xyz"
# species_target = "He"

# lag_frames = 100      # ←「??step後」をここで指定（フレーム単位）
# bins = 80

# # 表示用（1フレームが何psか）
# time_step_ps = 0.01
# dump_stride = 1
# dt_frame_ps = time_step_ps * dump_stride
# # --------------------------------


# def iter_xyz_frames(path):
#     """Plain XYZ reader: natoms / comment / natoms lines (sp x y z ...)."""
#     with open(path, "r") as f:
#         while True:
#             line = f.readline()
#             if not line:
#                 return
#             line = line.strip()
#             if not line:
#                 continue
#             if not re.fullmatch(r"\d+", line):
#                 continue

#             natoms = int(line)
#             comment = f.readline()
#             if not comment:
#                 return
#             comment = comment.strip()

#             species = []
#             pos = np.empty((natoms, 3), dtype=float)
#             for i in range(natoms):
#                 parts = f.readline().split()
#                 if len(parts) < 4:
#                     raise ValueError("Bad atom line: need 'sp x y z ...'")
#                 species.append(parts[0])
#                 pos[i, 0] = float(parts[1])
#                 pos[i, 1] = float(parts[2])
#                 pos[i, 2] = float(parts[3])

#             yield {"species": species, "pos": pos, "comment": comment}


# def compute_species_displacement_magnitudes_unwrap(path, species_target, lag_frames):
#     """
#     unwrap座標前提:
#       1) 1フレーム目で species_target のインデックスを決める
#       2) 全時間窓 t→t+lag で |r(t+lag)-r(t)| を全原子分集める
#     返り値: disp(|Δr|) 1D array, n_frames, n_target
#     """
#     it = iter_xyz_frames(path)
#     fr0 = next(it)  # 1フレーム目
#     idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
#     if len(idx) == 0:
#         raise ValueError(f"No atoms found for species_target={species_target}")

#     buf = deque(maxlen=lag_frames + 1)

#     # frame0を入れる
#     buf.append(fr0["pos"][idx])
#     n_frames = 1

#     disp = []

#     # 残りフレームをストリーミング処理
#     for fr in it:
#         # 念のため、同じインデックスが同じ元素かチェック（順序が固定前提）
#         for k in idx:
#             if fr["species"][k] != species_target:
#                 raise ValueError("Atom ordering/species inconsistent across frames.")

#         buf.append(fr["pos"][idx])
#         n_frames += 1

#         if len(buf) == lag_frames + 1:
#             dr = buf[-1] - buf[0]                  # unwrap差分
#             mag = np.linalg.norm(dr, axis=1)       # (n_target,)
#             disp.extend(mag.tolist())

#     return np.array(disp, dtype=float), n_frames, len(idx)


# def main():
#     disp_A, n_frames, n_target = compute_species_displacement_magnitudes_unwrap(
#         input_path, species_target, lag_frames
#     )

#     lag_ps = lag_frames * dt_frame_ps
#     n_samples_expected = n_target * max(n_frames - lag_frames, 0)

#     base = Path(input_path).with_suffix("")
#     out_png = Path(f"{base}.{species_target}.disp_hist_lag{lag_frames}f.png")
#     out_csv = Path(f"{base}.{species_target}.disp_lag{lag_frames}f.csv")

#     # # CSV保存（1列：|Δr|）
#     # np.savetxt(out_csv, disp_A, delimiter=",", header="disp_A", comments="")
#     # print(f"Saved: {out_csv}")

#     # ヒストグラム
#     plt.figure(figsize=(6.8, 4.6))
#     plt.hist(disp_A, bins=bins, color="#1f77b4", alpha=0.90, edgecolor="white")
#     plt.xlabel(f"|Δr| over {lag_frames} frames (= {lag_ps:g} ps)  (Å)")
#     plt.ylabel("Counts")
#     plt.title(f"{species_target}: displacement magnitude histogram (unwrap)")

#     mu = float(np.mean(disp_A)) if disp_A.size else float("nan")
#     sd = float(np.std(disp_A, ddof=1)) if disp_A.size >= 2 else float("nan")
#     at = AnchoredText(
#         f"frames={n_frames}, lag={lag_frames}\n"
#         f"{species_target} count={n_target}\n"
#         f"n(samples)={disp_A.size} (expected {n_samples_expected})\n"
#         f"mean={mu:.4f} Å\nstd={sd:.4f} Å",
#         prop=dict(size=9), frameon=True, loc="upper right"
#     )
#     at.patch.set_alpha(0.85)
#     plt.gca().add_artist(at)

#     plt.grid(True, ls="--", alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=180)
#     plt.close()
#     print(f"Saved: {out_png}")


# if __name__ == "__main__":
#     main()
    
    
    
    








# -*- coding: utf-8 -*-
import re
from collections import deque
from pathlib import Path

import numpy as np

# --- headless save (avoid Qt/xcb) ---
import matplotlib
matplotlib.use("Agg")  # save without GUI [web:126]
import matplotlib.pyplot as plt

# -------- user parameters --------
input_path = "output.xyz"
species_target = "He"

lag_frames = 100          # ★ 1 step 変位
max_vectors_plot = 199802  # 図が重いときは減らす
seed = 0

show_axes = True
draw_sphere = True      # ガイド球（半透明）
sphere_alpha = 0.12

# 出力
out_png = None          # Noneなら自動命名
out_html = None         # PlotlyがあればHTMLも出す（回転したいなら推奨）
# --------------------------------


def iter_xyz_frames(path):
    """Plain XYZ reader: natoms / comment / natoms lines (sp x y z ...)."""
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            if not re.fullmatch(r"\d+", line):
                continue

            natoms = int(line)
            comment = f.readline()
            if not comment:
                return
            comment = comment.strip()

            species = []
            pos = np.empty((natoms, 3), dtype=float)
            for i in range(natoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError("Bad atom line: need 'sp x y z ...'")
                species.append(parts[0])
                pos[i, 0] = float(parts[1])
                pos[i, 1] = float(parts[2])
                pos[i, 2] = float(parts[3])

            yield {"species": species, "pos": pos, "comment": comment}


def displacement_vectors_reservoir_unwrap(path, species_target, lag_frames, max_keep=1000, seed=0):
    """
    unwrap座標前提:
      1) 1フレーム目で species_target のインデックスを決める
      2) 全時間窓 t→t+lag で dr = r(t+lag)-r(t) を集める
    返り値:
      sample_vecs: (M,3)  描画用にリザーバサンプリングした変位ベクトル
      mean_vec: (3,)      全サンプルの平均（オンライン計算）
      n_total: int        全サンプル数（= n_target*(n_frames-lag)）
      n_frames: int
      n_target: int
    """
    rng = np.random.default_rng(seed)

    it = iter_xyz_frames(path)
    fr0 = next(it)
    idx = [i for i, sp in enumerate(fr0["species"]) if sp == species_target]
    if len(idx) == 0:
        raise ValueError(f"No atoms found for species_target={species_target}")

    buf = deque(maxlen=lag_frames + 1)
    buf.append(fr0["pos"][idx])
    n_frames = 1

    # reservoir + online mean
    keep = []
    n_total = 0
    mean_vec = np.zeros(3, dtype=float)

    for fr in it:
        for k in idx:
            if fr["species"][k] != species_target:
                raise ValueError("Atom ordering/species inconsistent across frames.")

        buf.append(fr["pos"][idx])
        n_frames += 1

        if len(buf) == lag_frames + 1:
            dr = buf[-1] - buf[0]  # (n_target,3)
            for v in dr:
                n_total += 1
                mean_vec += (v - mean_vec) / n_total

                if len(keep) < max_keep:
                    keep.append(v.copy())
                else:
                    j = rng.integers(0, n_total)
                    if j < max_keep:
                        keep[j] = v.copy()

    sample_vecs = np.array(keep, dtype=float) if keep else np.zeros((0, 3), dtype=float)
    return sample_vecs, mean_vec, n_total, n_frames, len(idx)


def plot_matplotlib_quiver(sample_vecs, mean_vec, title, out_png, show_axes=True, draw_sphere=True):
    # スケール決め（表示範囲）
    if sample_vecs.size:
        norms = np.linalg.norm(sample_vecs, axis=1)
        R = float(np.quantile(norms, 0.999)) if norms.size else 1.0
    else:
        R = 1.0
    R = max(R, float(np.linalg.norm(mean_vec)), 1e-6) * 1.15

    fig = plt.figure(figsize=(6.8, 5.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # 原点
    ax.scatter([0], [0], [0], s=40)

    # ガイド球（半透明）
    if draw_sphere:
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        x = R*np.outer(np.cos(u), np.sin(v))
        y = R*np.outer(np.sin(u), np.sin(v))
        z = R*np.outer(np.ones_like(u), np.cos(v))
        #ax.plot_surface(x, y, z, alpha=sphere_alpha, linewidth=0, antialiased=False)

    # サンプル矢印（たくさん）
    if sample_vecs.size:
        X0 = np.zeros(sample_vecs.shape[0])
        Y0 = np.zeros(sample_vecs.shape[0])
        Z0 = np.zeros(sample_vecs.shape[0])
        ax.quiver(
            X0, Y0, Z0,
            sample_vecs[:, 0], sample_vecs[:, 1], sample_vecs[:, 2],
            arrow_length_ratio=0.08, linewidth=0.6, alpha=0.25
        )  # Axes3D.quiver [web:53]

    # 平均矢印（太く）
    ax.quiver(0, 0, 0, mean_vec[0], mean_vec[1], mean_vec[2],
              arrow_length_ratio=0.12, linewidth=2.5, color="crimson")

    # 軸矢印
    if show_axes:
        ax.quiver(0, 0, 0, R, 0, 0, arrow_length_ratio=0.06, color="k", linewidth=1.0)
        ax.quiver(0, 0, 0, 0, R, 0, arrow_length_ratio=0.06, color="k", linewidth=1.0)
        ax.quiver(0, 0, 0, 0, 0, R, arrow_length_ratio=0.06, color="k", linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)

    # はみ出し対策：tight bbox + pad [web:81]
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_plotly_html(sample_vecs, mean_vec, title, out_html):
    """
    “保存した図を回転”したい場合は Plotly のHTMLが確実。
    pip install plotly して使う。
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print(f"[skip] plotly not available: {e}")
        return

    # 原点→端点の線分として描く（矢印っぽく）
    xs, ys, zs = [], [], []
    for v in sample_vecs:
        xs += [0, v[0], None]
        ys += [0, v[1], None]
        zs += [0, v[2], None]

    fig = go.Figure()
    if sample_vecs.size:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=2, color="rgba(31,119,180,0.25)"),
            name="samples"
        ))
    fig.add_trace(go.Scatter3d(
        x=[0, mean_vec[0]], y=[0, mean_vec[1]], z=[0, mean_vec[2]],
        mode="lines+markers",
        line=dict(width=6, color="crimson"),
        marker=dict(size=4, color="crimson"),
        name="mean"
    ))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="cube"),
        showlegend=True
    )
    fig.write_html(out_html)
    print(f"Saved (interactive): {out_html}")


def main():
    sample_vecs, mean_vec, n_total, n_frames, n_target = displacement_vectors_reservoir_unwrap(
        input_path, species_target, lag_frames,
        max_keep=max_vectors_plot, seed=seed
    )

    base = Path(input_path).with_suffix("")
    png_path = out_png or f"{base}.{species_target}.dr3d_lag{lag_frames}f.png"
    html_path = out_html or f"{base}.{species_target}.dr3d_lag{lag_frames}f.html"

    title = (f"{species_target} displacement vectors (unwrap)  lag={lag_frames} frames\n"
             f"frames={n_frames}, {species_target} atoms={n_target}, total samples={n_total}, plotted={len(sample_vecs)}")

    plot_matplotlib_quiver(sample_vecs, mean_vec, title, png_path,
                           show_axes=show_axes, draw_sphere=draw_sphere)
    print(f"Saved: {png_path}")

    # “保存した図を回転”したいならHTMLも出す（Plotlyがある場合）
    plot_plotly_html(sample_vecs, mean_vec, title, html_path)


if __name__ == "__main__":
    main()

