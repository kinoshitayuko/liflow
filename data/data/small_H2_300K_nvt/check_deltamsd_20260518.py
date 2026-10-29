import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# ============================================================
# Settings
# ============================================================
XYZ_FILE   = "output.xyz"
DT_FS      = 10.0       # timestep of dataset (fs)
WINDOW_PS  = 0.5        # inference step width (ps)
THRESH_PCT = 80         # percentile threshold (80 = top 20%)
ELEMENT    = "Ti"       # e.g. "Li" or None for all atoms

# ============================================================
# Load trajectory
# ============================================================
print("Loading trajectory...")
traj = read(XYZ_FILE, index=":")
n_frames = len(traj)
print(f"  {n_frames} frames, {len(traj[0])} atoms")

symbols = np.array(traj[0].get_chemical_symbols())
if ELEMENT:
    atom_idx = np.where(symbols == ELEMENT)[0]
    print(f"  Filtering element: {ELEMENT} ({len(atom_idx)} atoms)")
else:
    atom_idx = np.arange(len(traj[0]))

# ============================================================
# Windowed ΔMSD
# ============================================================
window_steps = int(WINDOW_PS * 1000 / DT_FS)  # 0.5 ps = 50 steps
stride       = window_steps // 2               # 50% overlap

t_mids, delta_msds = [], []

for s in range(0, n_frames - window_steps, stride):
    e = s + window_steps
    pos_s = traj[s].get_positions()[atom_idx]
    pos_e = traj[e].get_positions()[atom_idx]
    disp2 = np.sum((pos_e - pos_s) ** 2, axis=1)
    delta_msds.append(disp2.mean())
    t_mids.append((s + window_steps / 2) * DT_FS / 1000)  # ps

t_mids     = np.array(t_mids)
delta_msds = np.array(delta_msds)

# ============================================================
# Cumulative MSD (reference: frame 0)
# ============================================================
ds_stride = max(1, n_frames // 500)
cum_t, cum_msd = [], []
ref_pos = traj[0].get_positions()[atom_idx]
for i in range(0, n_frames, ds_stride):
    pos   = traj[i].get_positions()[atom_idx]
    disp2 = np.sum((pos - ref_pos) ** 2, axis=1)
    cum_msd.append(disp2.mean())
    cum_t.append(i * DT_FS / 1000)

cum_t   = np.array(cum_t)
cum_msd = np.array(cum_msd)

# ============================================================
# Threshold & high-activity windows
# ============================================================
threshold  = np.percentile(delta_msds, THRESH_PCT)
high_mask  = delta_msds >= threshold
# Top 5% for overlay on cumulative MSD (avoid clutter)
top5_mask  = delta_msds >= np.percentile(delta_msds, 95)

el_label = ELEMENT if ELEMENT else "ALL"
print(f"\nThreshold (top {100 - THRESH_PCT}%): {threshold:.4f} A^2")
print(f"High-activity windows: {high_mask.sum()} / {len(delta_msds)}")

print(f"\n--- High-activity windows (top {100 - THRESH_PCT}%) ---")
print(f"{'rank':>4}  {'t_start(ps)':>11}  {'t_end(ps)':>9}  {'dMSD(A^2)':>10}")
for rank, idx in enumerate(np.argsort(delta_msds)[::-1][:30], 1):
    t_s = t_mids[idx] - WINDOW_PS / 2
    t_e = t_mids[idx] + WINDOW_PS / 2
    print(f"{rank:>4}  {t_s:>11.4f}  {t_e:>9.4f}  {delta_msds[idx]:>10.4f}")

# ============================================================
# Plot (English labels only — no CJK font needed)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
fig.suptitle(
    f"delta-MSD Analysis  |  window={WINDOW_PS} ps  |  element={el_label}",
    fontsize=13, fontweight="bold"
)

# --- (1) Windowed delta-MSD time series ---
ax = axes[0]
bar_colors = np.where(high_mask, "#e05260", "#4f98a3")
ax.bar(t_mids, delta_msds,
       width=stride * DT_FS / 1000 * 0.85,
       color=bar_colors, alpha=0.85, linewidth=0)
ax.axhline(threshold, color="#e05260", lw=1.5, ls="--",
           label=f"threshold {threshold:.3f} A^2 (top {100 - THRESH_PCT}%)")
ax.set_xlabel("time (ps)")
ax.set_ylabel("delta-MSD (A^2)")
ax.set_title(f"0.5 ps windowed delta-MSD  [red = high-activity, top {100 - THRESH_PCT}%]")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, lw=0.8)

# --- (2) Cumulative MSD with top-5% overlay ---
ax = axes[1]
ax.plot(cum_t, cum_msd, color="#5591c7", lw=1.8, zorder=3)
ax.fill_between(cum_t, cum_msd, alpha=0.12, color="#5591c7")
# Overlay only top-5% windows to avoid clutter
for idx in np.where(top5_mask)[0]:
    t_s = t_mids[idx] - WINDOW_PS / 2
    t_e = t_mids[idx] + WINDOW_PS / 2
    ax.axvspan(t_s, t_e, color="#e05260", alpha=0.25, zorder=2)
ax.set_xlabel("time (ps)")
ax.set_ylabel("MSD (A^2)")
ax.set_title("Cumulative MSD  [red shading = top 5% delta-MSD windows]")
ax.grid(alpha=0.3, lw=0.8)

# --- (3) delta-MSD histogram ---
ax = axes[2]
n, bins, patches = ax.hist(delta_msds, bins=50,
                           color="#4f98a3", alpha=0.7, edgecolor="none")
thresh_bin = np.searchsorted(bins, threshold)
for patch, left in zip(patches, bins[:-1]):
    if left >= threshold:
        patch.set_facecolor("#e05260")
        patch.set_alpha(0.8)
ax.axvline(threshold, color="#e05260", lw=2, ls="--",
           label=f"threshold {threshold:.3f} A^2")
ax.set_xlabel("delta-MSD (A^2)")
ax.set_ylabel("count")
ax.set_title("delta-MSD distribution  [red = priority training region]")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, lw=0.8)

plt.savefig("delta_msd_analysis_20260608_Ti.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: delta_msd_analysis_20260608_Ti.png")