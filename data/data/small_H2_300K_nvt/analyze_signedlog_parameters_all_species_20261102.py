# -*- coding: utf-8 -*-
"""Analyze all species in unwrapped XYZ trajectories for signed-log parameters.

All available lag pairs are used; no train/validation split is performed.
Requires numpy, scipy, matplotlib. XYZ columns must start with species x y z.
"""
import csv
import json
import re
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

# ---------------- User settings ----------------
INPUT_PATHS = ["output.xyz"]
LAG_FRAMES = 50
TIME_STEP_PS = 0.01
DUMP_STRIDE = 1
STRENGTHS = [0.1]
BINS = 80
# OUTPUT_DIR = Path("signedlog_parameter_analysis")
OUTPUT_DIR = Path(".")
OUT_STEM = "positions_300K"

# Fill with the EXACT symbol -> internal ID mapping used by the model.
# Leave empty to analyze by species without guessing model IDs.
# Example syntax only: ELEMENT_ID_MAP = {"symbol": internal_id}
# ELEMENT_ID_MAP = {}
ELEMENT_ID_MAP = {
    "H": 0,
    "He": 1,
    "O": 2,
    "Si": 3,
    "Ti": 4,
}
STD_DDOF = 1
STATS_BIAS = True
SMALL_STD_WARNING_A = 1e-8  # Diagnostic threshold, not a numerical floor.
INVERSE_RTOL = 1e-10
INVERSE_ATOL = 1e-12
FIGURE_SIZE = (15, 6)
FIGURE_DPI = 300
SAVE_PDF = False
SAVE_DIAGNOSTIC_CSV = True
# ------------------------------------------------


def iter_xyz_frames(path):
    with open(path, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            if not re.fullmatch(r"\d+", line):
                raise ValueError(f"{path}: expected atom count, got {line!r}")
            n = int(line)
            if n < 1 or not f.readline():
                raise ValueError(f"{path}: invalid atom count or missing comment")
            symbols, positions = [], np.empty((n, 3), dtype=np.float64)
            for i in range(n):
                fields = f.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"{path}: invalid atom line {i + 1}")
                symbols.append(fields[0])
                positions[i] = [float(x) for x in fields[1:4]]
            if not np.isfinite(positions).all():
                raise ValueError(f"{path}: nonfinite coordinates")
            yield tuple(symbols), positions


def collect_displacements():
    chunks, expected, sources = {}, {}, []
    for path in INPUT_PATHS:
        frames = iter_xyz_frames(path)
        first = next(frames, None)
        if first is None:
            raise ValueError(f"{path}: empty trajectory")
        reference, pos = first
        masks = {s: np.array([x == s for x in reference])
                 for s in sorted(set(reference))}
        for s in masks:
            chunks.setdefault(s, [])
            expected.setdefault(s, 0)
        # Each chunk contains all atoms and all three components.
        # Split by species only after concatenation to avoid one array per frame/species.
        local_chunks = []
        buf = deque([pos], maxlen=LAG_FRAMES + 1)
        nframes = 1
        for symbols, pos in frames:
            if symbols != reference:
                raise ValueError(f"{path}: atom count/order/species changed")
            nframes += 1
            buf.append(pos)
            if len(buf) == LAG_FRAMES + 1:
                local_chunks.append(buf[-1] - buf[0])
        npairs = max(nframes - LAG_FRAMES, 0)
        if local_chunks:
            vectors = np.stack(local_chunks)
            del local_chunks
            for s, mask in masks.items():
                chunks[s].append(vectors[:, mask, :].reshape(-1, 3))
            del vectors
        counts = {s: int(mask.sum()) for s, mask in masks.items()}
        for s, n in counts.items():
            expected[s] += n * npairs
        sources.append({"path": str(path), "frames": nframes,
                        "atoms_per_frame": len(reference),
                        "atoms_per_species": counts, "lag_pairs": npairs})
        print(f"[{path}] frames={nframes:,}, lag_pairs={npairs:,}, species={counts}")
    data = {}
    for s, parts in chunks.items():
        values = np.concatenate(parts) if parts else np.empty((0, 3))
        if len(values) != expected[s] or len(values) < 2:
            raise ValueError(f"{s}: unexpected or insufficient sample count")
        if not np.isfinite(values).all():
            raise ValueError(f"{s}: nonfinite displacement")
        data[s] = values
    return data, sources


def statistics(x):
    result = {"n_samples": int(x.size), "mean": float(np.mean(x)),
              "std": float(np.std(x, ddof=STD_DDOF)),
              "skew": float(skew(x, bias=STATS_BIAS)),
              "kurt": float(kurtosis(x, fisher=False, bias=STATS_BIAS))}
    if not all(np.isfinite(v) for v in result.values()) or result["std"] <= 0:
        raise ValueError("Undefined statistics or zero standard deviation")
    if result["kurt"] <= 0:
        raise ValueError("Kurtosis must be positive")
    return result


def transform(x, a):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return np.sign(x) * np.log1p(a * np.abs(x))


def inverse(z, a):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return np.sign(z) * np.expm1(np.abs(z)) / a


def tag(value):
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def plot_species(symbol, eid, before, after, stats_before, stats_after, strength, base):
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE,
                             sharex="row", sharey="row", squeeze=False)
    id_label = str(eid) if eid is not None else "unmapped"
    for row, (values, stats, stage, color) in enumerate([
        (before, stats_before, "Before transformation", "#1f77b4"),
        (after, stats_after, f"After signed-log (lambda={strength:g})", "#ff7f0e")]):
        limit = float(np.max(np.abs(values)))
        edges = np.linspace(-limit, limit, BINS + 1)
        peak = 0.0
        for c, component in enumerate("xyz"):
            ax, st = axes[row, c], stats[c]
            unit = " Å" if row == 0 else ""
            label = (f"mean = {st['mean']:.4g}{unit}\nstd = {st['std']:.4g}{unit}\n"
                     f"skew = {st['skew']:.4f}\nkurt = {st['kurt']:.4f}")
            heights, _, _ = ax.hist(values[:, c], bins=edges, density=True,
                color=color, alpha=0.85, edgecolor="white", linewidth=0.35, label=label)
            peak = max(peak, float(heights.max()))
            ax.set_title(f"{stage}: {component}", fontsize=11)
            ax.set_xlabel(rf"$\Delta {component}$ (Å)" if row == 0
                          else rf"Transformed $\Delta {component}$ (dimensionless)")
            ax.set_ylabel("Probability density")
            ax.set_xlim(-1.02 * limit, 1.02 * limit)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend(loc="upper right", fontsize=9)
        for ax in axes[row]:
            ax.set_ylim(0, peak * 1.10)
    fig.suptitle(f"{symbol} | model element ID: {id_label}", fontsize=13)
    fig.tight_layout(pad=0.6, h_pad=1.0, w_pad=0.8)
    fig.savefig(f"{base}.png", dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.03)
    if SAVE_PDF:
        fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {base}.png")


def validate_settings():
    if not INPUT_PATHS:
        raise ValueError("INPUT_PATHS is empty")
    for path in INPUT_PATHS:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not isinstance(LAG_FRAMES, int) or LAG_FRAMES < 1:
        raise ValueError("LAG_FRAMES must be a positive integer")
    if not isinstance(BINS, int) or BINS < 1 or STD_DDOF not in (0, 1):
        raise ValueError("Invalid BINS or STD_DDOF")
    if not np.isfinite(TIME_STEP_PS * DUMP_STRIDE) or TIME_STEP_PS * DUMP_STRIDE <= 0:
        raise ValueError("Frame interval must be positive and finite")
    if not STRENGTHS or any(not np.isfinite(s) or s <= 0 for s in STRENGTHS):
        raise ValueError("STRENGTHS must contain positive finite values")
    ids = list(ELEMENT_ID_MAP.values())
    if any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Mapping requires unique nonnegative integer IDs")


def main():
    validate_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    data, sources = collect_displacements()
    if ELEMENT_ID_MAP:
        missing = sorted(set(data) - set(ELEMENT_ID_MAP))
        if missing:
            raise ValueError(f"Species missing from ELEMENT_ID_MAP: {missing}")
    ready = bool(ELEMENT_ID_MAP)
    if not ready:
        print("[WARNING] IDs are unset: species analysis only. Set ELEMENT_ID_MAP before model use.")
    species_order = sorted(data, key=lambda s: (ELEMENT_ID_MAP.get(s, 0), s))
    before_stats = {}
    for symbol in species_order:
        before_stats[symbol] = []
        for c, component in enumerate("xyz"):
            try:
                st = statistics(data[symbol][:, c])
            except ValueError as exc:
                raise ValueError(f"{symbol}/{component}: {exc}") from exc
            before_stats[symbol].append(st)
            if st["std"] < SMALL_STD_WARNING_A:
                print(f"[WARNING] small std: {symbol}/{component} = {st['std']:.8g} Å")
    for strength in dict.fromkeys(float(s) for s in STRENGTHS):
        parameter_rows, diagnostic_rows, entries = [], [], []
        for symbol in species_order:
            eid = ELEMENT_ID_MAP.get(symbol)
            values = data[symbol]
            sb = before_stats[symbol]
            sigma = np.array([s["std"] for s in sb])
            kval = np.array([s["kurt"] for s in sb])
            a = strength * kval / sigma
            if not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(f"{symbol}: invalid transform coefficients")
            after = transform(values, a)
            restored = inverse(after, a)
            if not np.isfinite(after).all() or not np.isfinite(restored).all():
                raise ValueError(f"{symbol}: nonfinite transformed/restored values")
            if not np.allclose(restored, values, rtol=INVERSE_RTOL, atol=INVERSE_ATOL):
                raise ValueError(f"{symbol}: inverse consistency check failed")
            errors = np.max(np.abs(restored - values), axis=0)
            sa = [statistics(after[:, c]) for c in range(3)]
            for c, component in enumerate("xyz"):
                common = {"element_id": eid, "species": symbol, "component": component,
                          "lag_frames": LAG_FRAMES,
                          "lag_ps": LAG_FRAMES * TIME_STEP_PS * DUMP_STRIDE,
                          "std_ddof": STD_DDOF, "stats_bias": STATS_BIAS,
                          "kurtosis_definition": "Pearson (normal=3)",
                          "data_scope": "all_input_frames_no_split"}
                parameter_rows.append({**common, "n_samples": len(values),
                    "sigma_A": sigma[c], "K": kval[c], "lambda": strength,
                    "effective_k": strength * kval[c], "a_per_A": a[c],
                    "inverse_max_abs_error_A": errors[c]})
                for stage, st in [("before", sb[c]), ("after", sa[c])]:
                    diagnostic_rows.append({**common, "stage": stage, **st,
                        "mean_std_unit": "angstrom" if stage == "before" else "dimensionless",
                        "transform_sigma_A": sigma[c], "transform_K": kval[c],
                        "lambda": strength, "a_per_A": a[c],
                        "inverse_max_abs_error_A": errors[c]})
            entries.append({"element_id": eid, "species": symbol,
                "n_samples_per_component": len(values), "sigma_A": sigma.tolist(),
                "K": kval.tolist(), "a_per_A": a.tolist()})
            print(f"[{symbol}, id={eid}] N={len(values):,}, sigma={sigma}, K={kval}, "
                  f"max_inverse_error={errors.max():.3e} Å")
            base = OUTPUT_DIR / (f"{OUT_STEM}_{symbol}_id{eid if eid is not None else 'unmapped'}"
                f"_xyz_signedlog_lambda{tag(strength)}_lag{LAG_FRAMES}f")
            plot_species(symbol, eid, values, after, sb, sa, strength, base)
            del after, restored
        base = OUTPUT_DIR / f"{OUT_STEM}_all_species_lambda{tag(strength)}_lag{LAG_FRAMES}f"
        write_csv(f"{base}_parameters.csv", parameter_rows)
        if SAVE_DIAGNOSTIC_CSV:
            write_csv(f"{base}_statistics.csv", diagnostic_rows)
        payload = {"schema_version": 1, "ready_for_model_use": ready,
            "formula": "sign(d)*log1p(lambda*K*abs(d)/sigma)",
            "inverse_formula": "sign(z)*expm1(abs(z))/a", "component_order": list("xyz"),
            "lambda": strength, "lag_frames": LAG_FRAMES,
            "frame_interval_ps": TIME_STEP_PS * DUMP_STRIDE,
            "lag_ps": LAG_FRAMES * TIME_STEP_PS * DUMP_STRIDE,
            "std_ddof": STD_DDOF, "stats_bias": STATS_BIAS,
            "kurtosis_definition": "Pearson (normal=3)",
            "data_scope": "all_input_frames_no_split", "coordinates": "already_unwrapped",
            "mean_centering": False, "sources": sources, "elements": entries}
        with open(f"{base}_parameters.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"Saved: {base}_parameters.json")


if __name__ == "__main__":
    main()