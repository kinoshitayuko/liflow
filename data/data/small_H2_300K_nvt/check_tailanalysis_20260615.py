import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read

# ============================================================
# Settings
# ============================================================
XYZ_FILE   = "output.xyz"
DT_FS      = 10.0       # timestep of dataset (fs)
WINDOW_PS  = 0.5        # inference step width (ps)
ELEMENT    = "He"       # e.g. "Li" or None for all atoms
TOP_Q_LIST = [1, 2, 5, 10, 20]   # top q% tail candidates
NBINS      = 70
PREFIX     = "rare_event_contrib_20260615_He"

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
# Collect all 0.5 ps displacements for target atoms
# ============================================================
window_steps = int(WINDOW_PS * 1000 / DT_FS)
valid_starts = n_frames - window_steps

print(f"\nCollecting |dr| over {window_steps} frames (= {WINDOW_PS} ps) ...")
drs = []
rows = []
for s in range(valid_starts):
    e = s + window_steps
    pos_s = traj[s].get_positions()[atom_idx]
    pos_e = traj[e].get_positions()[atom_idx]
    disp = pos_e - pos_s
    dr = np.linalg.norm(disp, axis=1)
    for local_i, atom_global_idx in enumerate(atom_idx):
        rows.append({
            "start_step": s,
            "end_step": e,
            "start_ps": s * DT_FS / 1000,
            "end_ps": e * DT_FS / 1000,
            "atom_index": int(atom_global_idx),
            "dr_A": float(dr[local_i]),
            "dr2_A2": float(dr[local_i] ** 2),
        })
        drs.append(dr[local_i])

df = pd.DataFrame(rows)
drs = np.array(drs)
dr2 = drs ** 2
n_samples = len(df)

print(f"  n(samples) = {n_samples}")
print(f"  mean |dr| = {drs.mean():.4f} A")
print(f"  mean |dr|^2 = {dr2.mean():.4f} A^2")

# ============================================================
# Tail contribution analysis
# ============================================================
order = np.argsort(dr2)[::-1]
sorted_dr2 = dr2[order]
cumsum_dr2 = np.cumsum(sorted_dr2)
total_dr2 = sorted_dr2.sum()

summary_rows = []
for q in TOP_Q_LIST:
    k = max(1, int(np.ceil(n_samples * q / 100.0)))
    tail_idx = order[:k]
    threshold_dr = drs[tail_idx].min()
    threshold_dr2 = dr2[tail_idx].min()
    contrib = sorted_dr2[:k].sum() / total_dr2
    summary_rows.append({
        "top_percent": q,
        "n_tail_samples": k,
        "tail_fraction": k / n_samples,
        "threshold_dr_A": threshold_dr,
        "threshold_dr2_A2": threshold_dr2,
        "msd_contribution_fraction": contrib,
        "suggested_training_presence_min": min(1.0, contrib * 0.5),
        "suggested_training_presence_mid": min(1.0, contrib * 0.75),
        "suggested_training_presence_max": min(1.0, contrib * 1.0),
        "oversample_ratio_min": (min(1.0, contrib * 0.5) / (k / n_samples)),
        "oversample_ratio_mid": (min(1.0, contrib * 0.75) / (k / n_samples)),
        "oversample_ratio_max": (min(1.0, contrib * 1.0) / (k / n_samples)),
    })

summary = pd.DataFrame(summary_rows)

# ============================================================
# Window-level mean delta-MSD (same style as your current script)
# ============================================================
stride = window_steps // 2
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

# # ============================================================
# # Save CSV files
# # ============================================================
# summary_csv = f"{PREFIX}_summary.csv"
# samples_csv = f"{PREFIX}_all_samples.csv"
# windows_csv = f"{PREFIX}_windowed_delta_msd.csv"
# summary.to_csv(summary_csv, index=False)
# df.to_csv(samples_csv, index=False)
# pd.DataFrame({"t_mid_ps": t_mids, "delta_msd_A2": delta_msds}).to_csv(windows_csv, index=False)

# ============================================================
# Print concise summary
# ============================================================
print("\n=== Tail contribution summary ===")
print(summary[[
    "top_percent",
    "n_tail_samples",
    "threshold_dr_A",
    "msd_contribution_fraction",
    "oversample_ratio_mid",
]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# Plot 1: displacement histogram + threshold lines
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
ax.hist(drs, bins=NBINS, color="#4f98a3", edgecolor="white", alpha=0.90)
for q, color in zip(TOP_Q_LIST, ["#7b2cbf", "#3a86ff", "#ff006e", "#fb5607", "#e63946"]):
    thr = summary.loc[summary["top_percent"] == q, "threshold_dr_A"].iloc[0]
    ax.axvline(thr, color=color, lw=2, ls="--", label=f"top {q}% threshold = {thr:.3f} A")
ax.set_xlabel(f"|dr| over {window_steps} frames (= {WINDOW_PS} ps) (A)")
ax.set_ylabel("Counts")
ax.set_title(f"{ELEMENT if ELEMENT else 'ALL'} displacement histogram and tail thresholds")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)
fig.savefig(f"{PREFIX}_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Plot 2: cumulative MSD contribution curve
# ============================================================
frac = np.arange(1, n_samples + 1) / n_samples * 100.0
cum_frac = cumsum_dr2 / total_dr2 * 100.0
fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
ax.plot(frac, cum_frac, color="#d62828", lw=2.5)
for q in TOP_Q_LIST:
    row = summary[summary["top_percent"] == q].iloc[0]
    y = row["msd_contribution_fraction"] * 100.0
    ax.scatter([q], [y], s=55, color="#1d3557", zorder=3)
    ax.text(q, y + 2.0, f"top {q}% -> {y:.1f}%", fontsize=9, ha="center")
ax.set_xlim(0, max(TOP_Q_LIST) * 1.2)
ax.set_ylim(0, min(100, cum_frac[np.searchsorted(frac, max(TOP_Q_LIST) * 1.2)] + 10))
ax.set_xlabel("Top-q% largest |dr|^2 samples")
ax.set_ylabel("Cumulative contribution to total sum(|dr|^2) (%)")
ax.set_title(f"Tail contribution to MSD-like quantity over {WINDOW_PS} ps samples")
ax.grid(alpha=0.3)
fig.savefig(f"{PREFIX}_cumulative_contribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Plot 3: your original windowed delta-MSD bar chart
# ============================================================
threshold_20 = np.percentile(delta_msds, 80)
high_mask = delta_msds >= threshold_20
fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
bar_colors = np.where(high_mask, "#e05260", "#4f98a3")
ax.bar(
    t_mids,
    delta_msds,
    width=stride * DT_FS / 1000 * 0.85,
    color=bar_colors,
    alpha=0.85,
    linewidth=0,
)
ax.axhline(
    threshold_20,
    color="#e05260",
    lw=1.5,
    ls="--",
    label=f"threshold {threshold_20:.3f} A^2 (top 20%)",
)
ax.set_xlabel("time (ps)")
ax.set_ylabel(f"MSD over {WINDOW_PS} ps window (A^2)")
ax.set_title(f"Mean squared displacement in each {WINDOW_PS} ps window (A^2)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, lw=0.8)
fig.savefig(f"{PREFIX}_windowed_delta_msd.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# print(f"\nSaved: {summary_csv}")
# print(f"Saved: {samples_csv}")
# print(f"Saved: {windows_csv}")
print(f"Saved: {PREFIX}_histogram.png")
print(f"Saved: {PREFIX}_cumulative_contribution.png")
print(f"Saved: {PREFIX}_windowed_delta_msd.png")