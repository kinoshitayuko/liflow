import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# ============================================================
# Settings
# ============================================================
XYZ_FILE   = "positions_650K.xyz"
DT_FS      = 15.0       # timestep of dataset (fs)
WINDOW_PS  = 1.005        # inference step width (ps)
THRESH_PCT = 80         # percentile threshold (80 = top 20%)
ELEMENT    = "Li"        # e.g. "Li" or None for all atoms

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
window_steps = int(WINDOW_PS * 1000 / DT_FS)
#stride = window_steps // 2
stride = 1

t_mids, delta_msds = [], []

for s in range(0, n_frames - window_steps, stride):
    e = s + window_steps
    pos_s = traj[s].get_positions()[atom_idx]
    pos_e = traj[e].get_positions()[atom_idx]
    disp2 = np.sum((pos_e - pos_s) ** 2, axis=1)
    delta_msds.append(disp2.mean())
    t_mids.append((s + window_steps / 2) * DT_FS / 1000)

t_mids = np.array(t_mids)
delta_msds = np.array(delta_msds)

# ============================================================
# Threshold & high-activity windows
# ============================================================
threshold = np.percentile(delta_msds, THRESH_PCT)
high_mask = delta_msds >= threshold

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
# Plot: delta-MSD only
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
fig.suptitle(
    f"windowed MSD Analysis | window={WINDOW_PS} ps | element={el_label}",
    fontsize=13, fontweight="bold"
)

bar_colors = np.where(high_mask, "#e05260", "#4f98a3")
ax.bar(
    t_mids,
    delta_msds,
    width=stride * DT_FS / 1000 * 0.85,
    color=bar_colors,
    alpha=0.85,
    linewidth=0
)

ax.axhline(
    threshold,
    color="#e05260",
    lw=1.5,
    ls="--",
    label=f"threshold {threshold:.3f} A^2 (top {100 - THRESH_PCT}%)"
)

ax.set_xlabel("time (ps)")
ax.set_ylabel("MSD over 1.005 ps window (A^2)")
ax.set_title(f"Mean squared displacement in each 1.005 ps window (A^2) [red = high-activity, top {100 - THRESH_PCT}%]")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, lw=0.8)

plt.savefig("delta_msd_analysis_20260702_Li.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: delta_msd_analysis_20260702_Li.png")