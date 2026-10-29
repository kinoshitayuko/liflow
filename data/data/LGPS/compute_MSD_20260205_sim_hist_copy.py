# -*- coding: utf-8 -*-
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- user parameters --------
input_path = "traj_300K_42_20251015_0.5_3_fixed.xyz"  # アンラップ座標（xu,yu,zu）を含む想定
species_target = "He"
time_step_ps = 1.0
dump_stride = 1
dt_frame_ps = time_step_ps * dump_stride
interval_frames = 1         # 1 ps刻み
bins = 80
# ---------------------------------

def parse_lattice_lengths(lattice_str):
    nums = [float(x) for x in lattice_str.replace('"','').split()
            if re.match(r"[-+0-9.eE]+", x)]
    if len(nums) >= 9:
        return np.array([nums[0], nums[4], nums[8]], dtype=float)
    raise ValueError("Lattice line not in expected orthorhombic format")

def read_extended_xyz(path):
    frames = []
    with open(path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.isdigit():
            natoms = int(line)
            i += 1
            if i >= len(lines): break
            header = lines[i].strip()
            m_lat = re.search(r'Lattice=("[^"]+")', header)
            if not m_lat:
                raise ValueError(f"No Lattice= found near line {i}: {header}")
            box = parse_lattice_lengths(m_lat.group(1))
            species, pos = [], []
            for _ in range(natoms):
                i += 1
                sp, x, y, z = lines[i].split()[:4]
                species.append(sp)
                # ここは xu,yu,zu を読む前提（アンラップ）
                pos.append([float(x), float(y), float(z)])
            frames.append({"species": species, "pos": np.array(pos, float), "box": box})
        i += 1
    return frames

def target_indices(species, target):
    return [idx for idx, sp in enumerate(species) if sp == target]

def compute_pair_mean_interval_displacement_unwrap(frames, interval=1):
    """各1 ps区間でHe2原子の変位の平均（アンラップ差分）を算出し、時系列とともに返す。"""
    he_idx = target_indices(frames[0]["species"], species_target)
    assert len(he_idx) == 2, f"{species_target} が2個ではありません: {len(he_idx)}"
    # 順序の整合性チェック
    for fr in frames:
        assert [fr["species"][k] for k in he_idx] == [frames[0]["species"][k] for k in he_idx], \
            "Target atom ordering inconsistent across frames"

    times_ps, dxm, dym, dzm = [], [], [], []
    for m in range(interval, len(frames), interval):
        r1 = frames[m - interval]["pos"][he_idx]  # unwrap
        r2 = frames[m]["pos"][he_idx]             # unwrap
        d = r2 - r1                               # 単純差（unwrap）
        dm = d.mean(axis=0)                       # 2原子平均
        dxm.append(dm[0]); dym.append(dm[1]); dzm.append(dm[2])
        times_ps.append(m * dt_frame_ps)
    return np.array(times_ps), np.array(dxm), np.array(dym), np.array(dzm)

def main():
    frames = read_extended_xyz(input_path)
    print(f"Loaded {len(frames)} frames from {input_path}")
    base = Path(input_path).with_suffix("")

    t_ps, dxm, dym, dzm = compute_pair_mean_interval_displacement_unwrap(
        frames, interval=interval_frames
    )

    # 1) 時系列（2原子平均Δ）
    def plot_series(t, v, label, color, outname):
        plt.figure(figsize=(7.5,4.3))
        plt.plot(t, v, lw=1.6, color=color)
        plt.axhline(0.0, color='k', lw=1.0, alpha=0.6)
        plt.xlabel("Time (ps)")
        plt.ylabel(f"mean {label} per 1 ps (Å)")
        plt.title(f"{species_target} pair-mean {label} time series (unwrapped)")
        plt.grid(True, ls="--", alpha=0.35)
        plt.tight_layout(); plt.savefig(outname, dpi=180); plt.close()
        print(f"Saved: {outname}")

    plot_series(t_ps, dxm, "Δx", "#1f77b4", Path(str(base)+".He_pairmean_dx_series_unwrap.png"))
    plot_series(t_ps, dym, "Δy", "#2ca02c", Path(str(base)+".He_pairmean_dy_series_unwrap.png"))
    plot_series(t_ps, dzm, "Δz", "#d62728", Path(str(base)+".He_pairmean_dz_series_unwrap.png"))

    # 2) ヒストグラム（2原子平均Δ）
    def plot_hist(data, color, label, outfile):
        plt.figure(figsize=(6.8,4.6))
        plt.hist(data, bins=bins, color=color, alpha=0.88, edgecolor="white")
        plt.xlabel(f"mean {label} per 1 ps (Å)")
        plt.ylabel("Counts")
        plt.title(f"{species_target} pair-mean {label} distribution (1 ps, unwrapped)")
        mu, sd, n = float(np.mean(data)), float(np.std(data, ddof=1)), data.size
        from matplotlib.offsetbox import AnchoredText
        at = AnchoredText(f"n={n}\nmean={mu:.3f} Å\nstd={sd:.3f} Å",
                          prop=dict(size=10), frameon=True, loc="upper right")
        at.patch.set_alpha(0.85); plt.gca().add_artist(at)
        plt.grid(True, ls="--", alpha=0.3)
        plt.tight_layout(); plt.savefig(outfile, dpi=180); plt.close()
        print(f"Saved: {outfile}")

    plot_hist(dxm, "#1f77b4", "Δx", Path(str(base)+".He_pairmean_dx_hist_unwrap.png"))
    plot_hist(dym, "#2ca02c", "Δy", Path(str(base)+".He_pairmean_dy_hist_unwrap.png"))
    plot_hist(dzm, "#d62728", "Δz", Path(str(base)+".He_pairmean_dz_hist_unwrap.png"))

if __name__ == "__main__":
    main()
