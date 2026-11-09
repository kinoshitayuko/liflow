"""Evaluate oxygen velocity errors on ground-truth flow inputs; no rollout/training."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from liflow.model.modules import FlowModule
from liflow.data.dataset import TimeDelayedPairDataset
from liflow.data.modules import _temporal_train_valid_split
from liflow.utils.prior import get_prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--times", type=float, nargs="+", default=[0., .5, .8, .95])
    parser.add_argument("--output-dir", default="oxygen_teacher_diagnostics")
    args = parser.parse_args()
    if args.samples < 1 or any(not 0 <= t <= 1 for t in args.times):
        raise ValueError("samples must be positive; times must be in [0,1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    model = FlowModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    model = model.to(args.device).eval()
    cfg = model.cfg
    if not getattr(model, "signed_log_enabled", False):
        raise ValueError("This diagnostic expects a signed-log checkpoint")
    if str(cfg.model.prediction_mode) != "velocity":
        raise ValueError("This diagnostic expects velocity mode")
    if not hasattr(model, "_prepare_flow"):
        raise ValueError("FlowModule._prepare_flow is required to reproduce training inputs")
    root = Path(args.data_dir).resolve()
    frame = pd.concat([pd.read_csv(root / str(f)) for f in cfg.data.index_files],
                      ignore_index=True)
    if cfg.data.get("train_valid_split", True):
        _, frame = _temporal_train_valid_split(frame, cfg.data.get("valid_fraction", .2))
        split = "validation"
    else:
        split = "all_rows_no_validation_split"
    if len(frame) == 0:
        raise ValueError("No evaluation rows")
    prior_cfg = cfg.propagate_prior
    params = dict(prior_cfg.params)
    params["seed"] = args.seed
    prior = get_prior(prior_cfg.class_name, **params)
    dataset = TimeDelayedPairDataset(
        data_path=root, df=frame, time_delay_steps=cfg.data.time_delay_steps,
        prior=prior, cutoff=cfg.data.cutoff, pbc=cfg.data.get("pbc", True),
        neighbor_list_both_ends=False, in_memory=cfg.data.get("in_memory", False),
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(dataset), size=args.samples)
    atomic_table = np.load(root / "atomic_numbers.npy", allow_pickle=True).item()
    rows = []
    with torch.no_grad():
        for sample, row_index in enumerate(indices):
            entry = frame.iloc[int(row_index)]
            atomic_numbers = np.asarray(atomic_table[entry["name"]])
            oxygen = np.flatnonzero(atomic_numbers == 8)
            if not len(oxygen):
                raise ValueError(f"No oxygen in structure {entry['name']}")
            batch = Batch.from_data_list([dataset.get(int(row_index))]).to(args.device)
            ids = torch.as_tensor(oxygen, device=args.device)
            for time in args.times:
                t = torch.tensor([time], dtype=torch.float32, device=args.device)
                flow, target, disp = model._prepare_flow(batch, t)
                prediction = model(flow, t_per_graph=t)
                if not (torch.isfinite(prediction).all() and torch.isfinite(target).all()):
                    raise FloatingPointError(f"Nonfinite prediction/target: sample={sample}, t={time}")
                p = prediction[ids].cpu().numpy()
                y = target[ids].cpu().numpy()
                d = disp[ids].cpu().numpy()
                x = (flow.positions_2 - flow.positions_1)[ids].cpu().numpy()
                pr = batch.prior[ids].cpu().numpy()
                element = batch.elements[ids].cpu().numpy()
                for k, atom in enumerate(oxygen):
                    record = dict(sample=sample, csv_row_index=int(row_index),
                        structure=str(entry["name"]), temperature=float(entry["temp"]),
                        time=float(time), atom_index=int(atom), element_id=int(element[k]))
                    for axis, component in enumerate("xyz"):
                        for label, array in (("prediction", p), ("target", y),
                            ("true_disp_A", d), ("input_disp_A", x), ("prior_A", pr)):
                            record[f"{label}_{component}"] = float(array[k, axis])
                    record["sq_error"] = float(np.sum((p[k].astype(float)-y[k])**2))
                    record["zero_baseline_sq_error"] = float(np.sum(y[k].astype(float)**2))
                    rows.append(record)
            if (sample + 1) % 16 == 0:
                print(f"Evaluated {sample + 1}/{args.samples} samples", flush=True)
    values = pd.DataFrame(rows)
    values.to_csv(out / "oxygen_per_atom.csv", index=False)
    summaries = []
    for time, group in values.groupby("time", sort=True):
        error = group.sq_error.mean()
        baseline = group.zero_baseline_sq_error.mean()
        record = dict(time=float(time), atom_evaluations=len(group),
            mean_sq_error=float(error), zero_prediction_baseline=float(baseline),
            error_to_zero_baseline=float(error/baseline) if baseline > 0 else None)
        for axis in "xyz":
            for label in ("prediction", "target", "true_disp_A", "input_disp_A"):
                v = group[f"{label}_{axis}"].to_numpy()
                record[f"{label}_{axis}_rms"] = float(np.sqrt(np.mean(v*v)))
                record[f"{label}_{axis}_max_abs"] = float(np.max(np.abs(v)))
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "oxygen_summary.csv", index=False)
    values[values.atom_index == 73].to_csv(out / "atom73.csv", index=False)
    metadata = dict(checkpoint=str(Path(args.ckpt).resolve()), data_dir=str(root),
        samples=args.samples, seed=args.seed, split=split, times=args.times,
        note="Random frames within selected CSV windows; intervals may overlap training. No rollout.")
    with (out / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(summary[["time", "mean_sq_error", "zero_prediction_baseline",
                   "error_to_zero_baseline"]].to_string(index=False))
    print(f"Saved diagnostics to {out.resolve()}")


if __name__ == "__main__":
    main()
