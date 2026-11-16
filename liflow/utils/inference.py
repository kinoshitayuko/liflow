# """Flow simulation with signed-log integration, retries and failure diagnostics."""
#csvは出力しない。
# import json
# import tempfile
# from pathlib import Path

# import ase
# import numpy as np
# import torch
# from torch_geometric.data import Data
# from tqdm import tqdm

# from liflow.model.modules import FlowModule
# from liflow.utils.geometry import get_neighbor_list_batch
# from liflow.utils.prior import (
#     AdaptiveMaxwellBoltzmannPrior, MaxwellBoltzmannPrior,
#     NormalPrior, Prior, UniformScaleNormalPrior,
# )


# class NonFiniteFlowError(RuntimeError):
#     """Numerical failure eligible for retry; configuration errors are not caught."""


# class FlowDiagnostics:
#     """Store scalar history plus latest tensor per name, without retaining GPU tensors."""
#     def __init__(self):
#         self.directory = Path(tempfile.mkdtemp(prefix="flow_debug_", dir="."))
#         self.step, self.attempt = -1, 0
#         self.arrays, self.trace = {}, []
#         print(f"[FlowDiagnostics] output={self.directory.resolve()}")

#     def put(self, name, value):
#         if isinstance(value, torch.Tensor):
#             value = value.detach().cpu().numpy()
#         self.arrays[name] = np.array(value, copy=True)

#     def record(self, tag, name, value, i=-1, t=None):
#         key = f"{tag}_{name}"
#         self.put(key, value)
#         array = self.arrays[key]
#         absolute = np.abs(array)
#         valid = np.isfinite(absolute)
#         maximum, index = None, None
#         if valid.any():
#             maximum = float(absolute[valid].max())
#             index = [int(x) for x in np.unravel_index(
#                 np.where(valid, absolute, -1).argmax(), array.shape)]
#         time = float(t.item()) if isinstance(t, torch.Tensor) else t
#         self.trace.append(dict(step=self.step, attempt=self.attempt, tag=tag,
#             name=name, flow_i=int(i), t=time, max_abs=maximum,
#             index=index, nonfinite=int((~valid).sum())))

#     def begin(self, step, attempt, positions):
#         self.step, self.attempt = step, attempt
#         self.arrays, self.trace = {}, []
#         self.put("positions_start", positions)

#     def save(self, error=None):
#         stem = (self.directory / "last_success" if error is None else
#                 self.directory / f"failure_step{self.step:06d}_attempt{self.attempt:02d}")
#         np.savez_compressed(str(stem) + ".npz", **self.arrays)
#         with open(str(stem) + ".json", "w", encoding="utf-8") as f:
#             json.dump(dict(step=self.step, attempt=self.attempt,
#                 error=None if error is None else str(error), trace=self.trace),
#                 f, indent=2, allow_nan=False)
#         if error is not None:
#             print(f"[FlowDiagnostics] saved: {stem}")
#             for row in self.trace[-16:]:
#                 print(f"  [trace] {row}")


# class FlowSimulator:
#     def __init__(
#         self, propagate_model: FlowModule, propagate_prior: Prior,
#         atomic_numbers: np.ndarray, element_idx: np.ndarray,
#         lattice: np.ndarray, temp: float,
#         correct_model: FlowModule | None = None,
#         correct_prior: Prior | None = None, pbc: bool = True,
#         scale_Li_index: int | None = None,
#         scale_frame_index: int | None = None,
#         fix_frame_species: list[int] | None = None,
#     ):
#         self.propagate_model = propagate_model
#         self.propagate_prior = propagate_prior
#         self.atomic_numbers = np.asarray(atomic_numbers, dtype=int)
#         lookup = np.asarray(element_idx)
#         if (lookup.ndim != 1 or self.atomic_numbers.ndim != 1
#                 or self.atomic_numbers.size == 0
#                 or np.any(self.atomic_numbers < 0)
#                 or np.any(self.atomic_numbers >= len(lookup))):
#             raise ValueError("Invalid atomic numbers or element lookup table")
#         self.elements = lookup[self.atomic_numbers]
#         if not np.issubdtype(self.elements.dtype, np.integer) or np.any(self.elements < 0):
#             raise ValueError("Element lookup must contain supported integer IDs")
#         self.masses = ase.data.atomic_masses[self.atomic_numbers]
#         if not np.isfinite(self.masses).all() or np.any(self.masses <= 0):
#             raise ValueError("Invalid masses")
#         self.lattice = np.asarray(lattice)
#         self.temp = temp
#         self.correct_model, self.correct_prior = correct_model, correct_prior
#         self.pbc = pbc
#         self.device = propagate_model.device
#         self.scale_Li_index, self.scale_frame_index = scale_Li_index, scale_frame_index
#         # API compatibility only: the supplied implementation did not apply this mask.
#         self.frame_mask = (np.isin(self.atomic_numbers, fix_frame_species)
#                            if fix_frame_species is not None else None)
#         if (correct_model is None) != (correct_prior is None):
#             raise ValueError("Corrector model and prior must be supplied together")
#         self._validate_model(propagate_model, "prop")
#         if correct_model is not None:
#             if torch.device(correct_model.device) != torch.device(self.device):
#                 raise ValueError("Propagator and corrector must use the same device")
#             self._validate_model(correct_model, "corr")
#         self.diagnostics = FlowDiagnostics()
#         for label, model in (("Propagator", propagate_model), ("Corrector", correct_model)):
#             if model is not None:
#                 space = "transformed velocity" if self._signed(model) else "physical-space flow"
#                 print(f"[FlowSimulator] {label}: signed_log={self._signed(model)}, "
#                       f"prediction_mode={self._mode(model)}, space={space}")

#     @staticmethod
#     def _signed(model):
#         return bool(getattr(model, "signed_log_enabled", False))

#     @staticmethod
#     def _mode(model):
#         return str(model.cfg.model.get("prediction_mode", "velocity"))

#     @staticmethod
#     def _finite(value, context):
#         if not torch.isfinite(value).all().item():
#             raise NonFiniteFlowError(f"Nonfinite values in {context}")

#     def _validate_model(self, model, tag):
#         mode = self._mode(model)
#         if mode not in ("velocity", "data"):
#             raise ValueError(f"Unsupported prediction mode: {mode}")
#         if self._signed(model):
#             if tag == "corr" or mode != "velocity":
#                 raise ValueError("Signed-log supports propagator velocity mode only")
#             transform = getattr(model, "displacement_transform", None)
#             if transform is None:
#                 raise ValueError("Signed-log checkpoint has no displacement_transform")
#             probe = torch.zeros((len(self.elements), 3), device=model.device)
#             ids = torch.as_tensor(self.elements, dtype=torch.long, device=model.device)
#             with torch.no_grad():
#                 z = transform.transform(probe, ids)
#                 self._finite(z, "transform validation")
#                 self._finite(transform.inverse(z, ids), "inverse validation")

#     def initialize_batch(self, positions: np.ndarray, prior: Prior) -> Data:
#         positions = np.asarray(positions)
#         if positions.shape != (len(self.elements), 3):
#             raise ValueError("Positions must have shape (n_atoms, 3)")
#         if not np.isfinite(positions).all():
#             raise NonFiniteFlowError("Nonfinite input positions")
#         edge_index, shifts = get_neighbor_list_batch(
#             positions_batch=[positions], lattice=self.lattice,
#             cutoff=self.propagate_model.cfg.data.cutoff, periodic=self.pbc,
#         )
#         if isinstance(prior, NormalPrior):
#             sampled = prior.sample(shape=positions.shape)
#         elif isinstance(prior, MaxwellBoltzmannPrior):
#             sampled = prior.sample(temperature=self.temp, masses=self.masses,
#                                    shape=positions.shape)
#         elif isinstance(prior, AdaptiveMaxwellBoltzmannPrior):
#             sampled = prior.sample(
#                 temperature=self.temp, atomic_numbers=self.atomic_numbers,
#                 masses=self.masses, scale_Li_index=self.scale_Li_index,
#                 scale_frame_index=self.scale_frame_index, shape=positions.shape,
#             )
#         elif isinstance(prior, UniformScaleNormalPrior):
#             sampled = prior.sample(shape=positions.shape)
#         else:
#             raise NotImplementedError(f"Unsupported prior: {type(prior).__name__}")
#         self.diagnostics.put("latest_sampled_prior", sampled)
#         data = Data(
#             num_nodes=len(self.elements),
#             positions_1=torch.tensor(positions, dtype=torch.float),
#             prior=torch.tensor(sampled, dtype=torch.float),
#             positions_2=torch.tensor(positions, dtype=torch.float),
#             edge_index=torch.tensor(edge_index, dtype=torch.long),
#             shifts=torch.tensor(shifts, dtype=torch.float),
#             elements=torch.tensor(self.elements, dtype=torch.long),
#             temp=torch.full(self.elements.shape, self.temp).float(),
#         )
#         data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
#         data = data.to(self.device, non_blocking=True)
#         self._finite(data.prior, "sampled prior")
#         return data

#     @staticmethod
#     def _min_pair_distance_torch(pos: torch.Tensor) -> float:
#         # Nonperiodic diagnostic only; not used to accept/reject configurations.
#         d = torch.cdist(pos, pos)
#         d.fill_diagonal_(float("inf"))
#         return d.min().item()

#     @staticmethod
#     @torch.no_grad()
#     def propagate_flow(data: Data, model: FlowModule, flow_steps: int = 10,
#                        solver: str = "euler", tag: str = "flow",
#                        diagnostics=None) -> torch.Tensor:
#         if flow_steps < 1:
#             raise ValueError("flow_steps must be >= 1")
#         if solver not in ("euler", "heun"):
#             raise ValueError(f"Unsupported solver: {solver}")
#         mode = FlowSimulator._mode(model)
#         if mode not in ("velocity", "data"):
#             raise ValueError(f"Unsupported prediction mode: {mode}")
#         finite = FlowSimulator._finite
#         def record(name, value, i=-1, t=None):
#             if diagnostics is not None:
#                 diagnostics.record(tag, name, value, i, t)
#         record("prior", data["prior"])
#         record("reference", data["positions_1"])
#         if diagnostics is not None:
#             for key in ("elements", "edge_index", "shifts"):
#                 diagnostics.put(f"{tag}_{key}", data[key])
#         t_list = torch.linspace(0.0, 1.0, flow_steps + 1,
#                                 device=model.device, dtype=torch.float)
#         dt = t_list[1] - t_list[0]
#         if FlowSimulator._signed(model):
#             if tag == "corr" or mode != "velocity":
#                 raise ValueError("Signed-log supports propagator velocity mode only")
#             transform = getattr(model, "displacement_transform", None)
#             if transform is None:
#                 raise ValueError("Missing signed-log transform")
#             reference, elements = data["positions_1"].clone(), data["elements"]
#             try:
#                 z = transform.transform(data["prior"], elements)
#             except FloatingPointError as exc:
#                 raise NonFiniteFlowError(f"{tag}: prior transform failed: {exc}") from exc
#             record("initial_z", z)
#             finite(z, f"{tag}: initial z")
#             if diagnostics is not None:
#                 diagnostics.put(f"{tag}_a_per_A", transform.a_per_A[elements.long()])

#             def physical(state, i, t, stage):
#                 record(f"{stage}_z", state, i, t)
#                 finite(state, f"{tag}: {stage} z, i={i}")
#                 try:
#                     disp = transform.inverse(state, elements)
#                 except FloatingPointError as exc:
#                     index = int(state.abs().reshape(-1).argmax().item())
#                     atom, axis = divmod(index, 3)
#                     element = int(elements[atom].item())
#                     value = float(state[atom, axis].item())
#                     a = float(transform.a_per_A[element, axis].item())
#                     # Largest |z| is a diagnostic candidate, not proven overflow origin.
#                     message = (f"{tag}: inverse failed, flow_i={i}, t={float(t):.6f}, "
#                         f"stage={stage}, max_abs_z={abs(value):.6e}, "
#                         f"atom_index={atom}, element_id={element}, "
#                         f"component={('x', 'y', 'z')[axis]}, z={value:.6e}, "
#                         f"a_per_A={a:.6e}; {exc}")
#                     raise NonFiniteFlowError(message) from exc
#                 record(f"{stage}_displacement", disp, i, t)
#                 finite(disp, f"{tag}: inverse displacement")
#                 pos = reference + disp
#                 record(f"{stage}_positions", pos, i, t)
#                 finite(pos, f"{tag}: physical coordinates")
#                 return pos

#             for i, t in enumerate(t_list[:-1]):
#                 data["positions_2"] = physical(z, i, t, "input")
#                 velocity = model(data, t_per_graph=t.unsqueeze(0))
#                 record("velocity", velocity, i, t)
#                 finite(velocity, f"{tag}: velocity step {i}")
#                 proposal = z + dt * velocity
#                 record("proposal_z", proposal, i, t_list[i + 1])
#                 finite(proposal, f"{tag}: proposed z step {i}")
#                 if solver == "euler":
#                     z = proposal
#                 else:
#                     next_data = data.clone()
#                     next_data["positions_2"] = physical(proposal, i, t_list[i + 1], "heun_predictor")
#                     next_velocity = model(next_data, t_per_graph=t_list[i + 1].unsqueeze(0))
#                     record("heun_velocity", next_velocity, i, t_list[i + 1])
#                     finite(next_velocity, f"{tag}: Heun velocity step {i}")
#                     z = z + 0.5 * (velocity + next_velocity) * dt
#                     record("heun_updated_z", z, i, t_list[i + 1])
#                     finite(z, f"{tag}: Heun z step {i}")
#             data["positions_2"] = physical(z, flow_steps, t_list[-1], "final")
#             return data["positions_2"]

#         for i, t in enumerate(t_list[:-1]):
#             record("input_positions", data["positions_2"], i, t)
#             prediction = model(data, t_per_graph=t.unsqueeze(0))
#             record("prediction", prediction, i, t)
#             finite(prediction, f"{tag}: prediction step {i}")
#             velocity = (prediction if mode == "velocity" else
#                 (data["positions_1"] + prediction - data["positions_2"]) / (1 - t))
#             record("velocity", velocity, i, t)
#             finite(velocity, f"{tag}: physical velocity step {i}")
#             if solver == "euler":
#                 data["positions_2"] += velocity * dt
#             else:
#                 next_t = (t + dt).unsqueeze(0)
#                 next_data = data.clone()
#                 next_data["positions_2"] += velocity * dt
#                 record("heun_proposal_positions", next_data["positions_2"], i, next_t[0])
#                 finite(next_data["positions_2"], f"{tag}: Heun proposal step {i}")
#                 next_prediction = model(next_data, t_per_graph=next_t)
#                 record("heun_prediction", next_prediction, i, next_t[0])
#                 finite(next_prediction, f"{tag}: Heun prediction step {i}")
#                 if mode == "velocity":
#                     next_velocity = next_prediction
#                 else:
#                     if i == flow_steps - 1:
#                         result = next_data["positions_1"] + next_prediction
#                         record("final_data_positions", result, i, next_t[0])
#                         finite(result, f"{tag}: final data prediction")
#                         return result
#                     next_velocity = (next_data["positions_1"] + next_prediction
#                                      - next_data["positions_2"]) / (1 - next_t)
#                 finite(next_velocity, f"{tag}: Heun physical velocity step {i}")
#                 data["positions_2"] += 0.5 * (velocity + next_velocity) * dt
#             record("updated_positions", data["positions_2"], i, t_list[i + 1])
#             finite(data["positions_2"], f"{tag}: updated coordinates step {i}")
#         return data["positions_2"]

#     @torch.no_grad()
#     def step(self, positions: np.ndarray, flow_steps: int = 10,
#              solver: str = "euler", use_corrector: bool = True) -> np.ndarray:
#         data = self.initialize_batch(positions, self.propagate_prior)
#         self.diagnostics.put("prop_sampled_prior", data["prior"])
#         if getattr(self.propagate_model, "use_scale", False):
#             scale = self.propagate_model.scale_model(data)
#             data["prior"] = data["prior"] * scale[:, None]
#             self._finite(data["prior"], "scaled prior")
#         if not self._signed(self.propagate_model):
#             data["positions_2"] = data["positions_2"] + data["prior"]
#         new_positions = self.propagate_flow(data, self.propagate_model,
#             flow_steps, solver, tag="prop", diagnostics=self.diagnostics).detach().cpu().numpy()
#         self.diagnostics.record("prop", "output_positions", new_positions)
#         self.diagnostics.record("prop", "output_displacement", new_positions - positions)
#         if use_corrector:
#             if self.correct_model is None or self.correct_prior is None:
#                 raise ValueError("Corrector requested but not configured")
#             prop_positions = new_positions.copy()
#             data = self.initialize_batch(new_positions, self.correct_prior)
#             self.diagnostics.put("corr_sampled_prior", data["prior"])
#             new_positions = self.propagate_flow(data, self.correct_model,
#                 flow_steps, solver, tag="corr", diagnostics=self.diagnostics).detach().cpu().numpy()
#             self.diagnostics.record("corr", "output_positions", new_positions)
#             self.diagnostics.record("corr", "correction", new_positions - prop_positions)
#         return new_positions

#     @torch.no_grad()
#     def run(self, positions: np.ndarray, steps: int = 25,
#             flow_steps: int = 10, solver: str = "euler", verbose: bool = False,
#             fix_com: bool = True, num_retries: int = 2,
#             corrector_every: int = 1) -> np.ndarray:
#         if steps < 0 or num_retries < 0:
#             raise ValueError("steps and num_retries must be nonnegative")
#         if flow_steps < 1 or corrector_every < 1:
#             raise ValueError("flow_steps and corrector_every must be >= 1")
#         if solver not in ("euler", "heun"):
#             raise ValueError(f"Unsupported solver: {solver}")
#         positions = np.asarray(positions).copy()
#         if positions.shape != (len(self.elements), 3) or not np.isfinite(positions).all():
#             raise ValueError("Invalid initial positions")
#         print(f"[FlowSimulator] solver={solver}, flow_steps={flow_steps}")
#         traj = [positions.copy()]
#         iterator = tqdm(range(steps)) if verbose else range(steps)
#         for i_step in iterator:
#             previous = positions.copy()
#             success = False
#             for i_retry in range(num_retries + 1):
#                 use_corrector = ((i_step + 1) % corrector_every == 0
#                                  and self.correct_model is not None)
#                 self.diagnostics.begin(i_step, i_retry + 1, previous)
#                 for name, value in (("atomic_numbers", self.atomic_numbers),
#                     ("elements", self.elements), ("lattice", self.lattice),
#                     ("masses", self.masses), ("pbc", self.pbc), ("temperature", self.temp),
#                     ("solver", solver), ("flow_steps", flow_steps),
#                     ("use_corrector", use_corrector), ("fix_com", fix_com)):
#                     self.diagnostics.put(name, value)
#                 try:
#                     candidate = self.step(previous, flow_steps, solver, use_corrector)
#                     self.diagnostics.put("positions_before_com", candidate)
#                     if fix_com:
#                         diff = candidate - previous
#                         com_diff = np.sum(diff * self.masses[:, None], axis=0) / np.sum(self.masses)
#                         self.diagnostics.put("com_diff", com_diff)
#                         candidate -= com_diff
#                     self.diagnostics.put("positions_after_com", candidate)
#                     if not np.isfinite(candidate).all():
#                         raise NonFiniteFlowError("Nonfinite final coordinates")
#                     if np.abs(candidate - previous).max() > 1e3:
#                         raise NonFiniteFlowError("Displacement component exceeded 1000 A")
#                 except NonFiniteFlowError as exc:
#                     self.diagnostics.save(exc)
#                     print(f"[FlowSimulator] step={i_step}, attempt={i_retry + 1}/"
#                           f"{num_retries + 1}: {exc}")
#                     continue
#                 self.diagnostics.save()
#                 positions = candidate
#                 success = True
#                 break
#             if not success:
#                 print(f"[FlowSimulator] Stopped early: completed {len(traj) - 1}/{steps} steps")
#                 break
#             traj.append(positions.copy())
#         return traj




"""Flow simulation with signed-log integration, retries and CSV diagnostics."""
#csvは出力する。

import csv
import json
import tempfile
from pathlib import Path

import ase
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from liflow.model.modules import FlowModule
from liflow.utils.geometry import get_neighbor_list_batch
from liflow.utils.prior import (
    AdaptiveMaxwellBoltzmannPrior,
    MaxwellBoltzmannPrior,
    NormalPrior,
    Prior,
    UniformScaleNormalPrior,
)

from ase.geometry import wrap_positions


class NonFiniteFlowError(RuntimeError):
    """Numerical failure eligible for retry."""


class FlowDiagnostics:
    """Store scalar history and latest arrays without retaining GPU tensors."""

    def __init__(self):
        self.directory = Path(
            tempfile.mkdtemp(prefix="flow_debug_", dir=".")
        )
        self.step = -1
        self.attempt = 0
        self.arrays = {}
        self.trace = []

        print(
            f"[FlowDiagnostics] output={self.directory.resolve()}"
        )

    def put(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        self.arrays[name] = np.array(value, copy=True)

    def record(self, tag, name, value, i=-1, t=None):
        key = f"{tag}_{name}"
        self.put(key, value)

        array = self.arrays[key]
        absolute = np.abs(array)
        valid = np.isfinite(absolute)

        maximum = None
        index = None

        if valid.any():
            maximum = float(absolute[valid].max())
            flat_index = np.where(valid, absolute, -1).argmax()
            index = [
                int(x)
                for x in np.unravel_index(flat_index, array.shape)
            ]

        time = float(t.item()) if isinstance(t, torch.Tensor) else t

        self.trace.append(
            dict(
                step=self.step,
                attempt=self.attempt,
                tag=tag,
                name=name,
                flow_i=int(i),
                t=time,
                max_abs=maximum,
                index=index,
                nonfinite=int((~valid).sum()),
            )
        )

    def begin(self, step, attempt, positions):
        self.step = step
        self.attempt = attempt
        self.arrays = {}
        self.trace = []

        self.put("positions_start", positions)

    def save(self, error=None):
        if error is None:
            stem = self.directory / "last_success"
        else:
            stem = self.directory / (
                f"failure_step{self.step:06d}"
                f"_attempt{self.attempt:02d}"
            )

        np.savez_compressed(
            str(stem) + ".npz",
            **self.arrays,
        )

        with open(
            str(stem) + ".json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                dict(
                    step=self.step,
                    attempt=self.attempt,
                    error=None if error is None else str(error),
                    trace=self.trace,
                ),
                f,
                indent=2,
                allow_nan=False,
            )

        if error is not None:
            print(f"[FlowDiagnostics] saved: {stem}")

            for row in self.trace[-16:]:
                print(f"  [trace] {row}")


class FlowSimulator:
    def __init__(
        self,
        propagate_model: FlowModule,
        propagate_prior: Prior,
        atomic_numbers: np.ndarray,
        element_idx: np.ndarray,
        lattice: np.ndarray,
        temp: float,
        correct_model: FlowModule | None = None,
        correct_prior: Prior | None = None,
        pbc: bool = True,
        scale_Li_index: int | None = None,
        scale_frame_index: int | None = None,
        fix_frame_species: list[int] | None = None,
    ):
        self.propagate_model = propagate_model
        self.propagate_prior = propagate_prior

        self.atomic_numbers = np.asarray(
            atomic_numbers,
            dtype=int,
        )
        lookup = np.asarray(element_idx)

        if (
            lookup.ndim != 1
            or self.atomic_numbers.ndim != 1
            or self.atomic_numbers.size == 0
            or np.any(self.atomic_numbers < 0)
            or np.any(self.atomic_numbers >= len(lookup))
        ):
            raise ValueError(
                "Invalid atomic numbers or element lookup table"
            )

        self.elements = lookup[self.atomic_numbers]

        if (
            not np.issubdtype(self.elements.dtype, np.integer)
            or np.any(self.elements < 0)
        ):
            raise ValueError(
                "Element lookup must contain supported integer IDs"
            )

        self.masses = ase.data.atomic_masses[self.atomic_numbers]

        if (
            not np.isfinite(self.masses).all()
            or np.any(self.masses <= 0)
        ):
            raise ValueError("Invalid masses")

        self.lattice = np.asarray(lattice)
        self.temp = temp
        self.correct_model = correct_model
        self.correct_prior = correct_prior
        self.pbc = pbc
        self.device = propagate_model.device

        self.scale_Li_index = scale_Li_index
        self.scale_frame_index = scale_frame_index

        # API compatibility: this mask is not applied in the original code.
        self.frame_mask = (
            np.isin(self.atomic_numbers, fix_frame_species)
            if fix_frame_species is not None
            else None
        )

        if (correct_model is None) != (correct_prior is None):
            raise ValueError(
                "Corrector model and prior must be supplied together"
            )

        self._validate_model(propagate_model, "prop")

        if correct_model is not None:
            if (
                torch.device(correct_model.device)
                != torch.device(self.device)
            ):
                raise ValueError(
                    "Propagator and corrector must use the same device"
                )

            self._validate_model(correct_model, "corr")

        self.diagnostics = FlowDiagnostics()

        self.csv_atom_indices = np.flatnonzero(
            self.atomic_numbers == ase.data.atomic_numbers["He"]
        )
        self.displacement_csv_path = (
            self.diagnostics.directory / "he_displacements.csv"
        )
        self._csv_run_id = -1

        if self.csv_atom_indices.size:
            print(
                "[FlowSimulator] CSV target=He-labelled H, "
                f"n_atoms={self.csv_atom_indices.size}, "
                "atom_index=0-based, displacement_unit=Angstrom"
            )
            print(
                "[FlowSimulator] displacement CSV="
                f"{self.displacement_csv_path.resolve()}"
            )
        else:
            print(
                "[FlowSimulator] WARNING: no He-labelled atoms found; "
                "displacement CSV output is disabled."
            )

        for label, model in (
            ("Propagator", propagate_model),
            ("Corrector", correct_model),
        ):
            if model is not None:
                space = (
                    "transformed velocity"
                    if self._signed(model)
                    else "physical-space flow"
                )

                print(
                    f"[FlowSimulator] {label}: "
                    f"signed_log={self._signed(model)}, "
                    f"prediction_mode={self._mode(model)}, "
                    f"space={space}"
                )

    @staticmethod
    def _signed(model):
        return bool(
            getattr(model, "signed_log_enabled", False)
        )

    @staticmethod
    def _mode(model):
        return str(
            model.cfg.model.get("prediction_mode", "velocity")
        )

    @staticmethod
    def _finite(value, context):
        if not torch.isfinite(value).all().item():
            raise NonFiniteFlowError(
                f"Nonfinite values in {context}"
            )

    def _validate_model(self, model, tag):
        mode = self._mode(model)

        if mode not in ("velocity", "data"):
            raise ValueError(
                f"Unsupported prediction mode: {mode}"
            )

        if self._signed(model):
            if tag == "corr" or mode != "velocity":
                raise ValueError(
                    "Signed-log supports propagator velocity mode only"
                )

            transform = getattr(
                model,
                "displacement_transform",
                None,
            )

            if transform is None:
                raise ValueError(
                    "Signed-log checkpoint has no displacement_transform"
                )

            probe = torch.zeros(
                (len(self.elements), 3),
                device=model.device,
            )
            ids = torch.as_tensor(
                self.elements,
                dtype=torch.long,
                device=model.device,
            )

            with torch.no_grad():
                z = transform.transform(probe, ids)
                self._finite(z, "transform validation")
                self._finite(
                    transform.inverse(z, ids),
                    "inverse validation",
                )

    def initialize_batch(
        self,
        positions: np.ndarray,
        prior: Prior,
    ) -> Data:
        positions = np.asarray(positions)

        if positions.shape != (len(self.elements), 3):
            raise ValueError(
                "Positions must have shape (n_atoms, 3)"
            )

        if not np.isfinite(positions).all():
            raise NonFiniteFlowError(
                "Nonfinite input positions"
            )

        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=[positions],
            lattice=self.lattice,
            cutoff=self.propagate_model.cfg.data.cutoff,
            periodic=self.pbc,
        )

        if isinstance(prior, NormalPrior):
            sampled = prior.sample(
                shape=positions.shape
            )

        elif isinstance(prior, MaxwellBoltzmannPrior):
            sampled = prior.sample(
                temperature=self.temp,
                masses=self.masses,
                shape=positions.shape,
            )

        elif isinstance(prior, AdaptiveMaxwellBoltzmannPrior):
            sampled = prior.sample(
                temperature=self.temp,
                atomic_numbers=self.atomic_numbers,
                masses=self.masses,
                scale_Li_index=self.scale_Li_index,
                scale_frame_index=self.scale_frame_index,
                shape=positions.shape,
            )

        elif isinstance(prior, UniformScaleNormalPrior):
            sampled = prior.sample(
                shape=positions.shape
            )

        else:
            raise NotImplementedError(
                f"Unsupported prior: {type(prior).__name__}"
            )

        self.diagnostics.put(
            "latest_sampled_prior",
            sampled,
        )

        data = Data(
            num_nodes=len(self.elements),
            positions_1=torch.tensor(
                positions,
                dtype=torch.float,
            ),
            prior=torch.tensor(
                sampled,
                dtype=torch.float,
            ),
            positions_2=torch.tensor(
                positions,
                dtype=torch.float,
            ),
            edge_index=torch.tensor(
                edge_index,
                dtype=torch.long,
            ),
            shifts=torch.tensor(
                shifts,
                dtype=torch.float,
            ),
            elements=torch.tensor(
                self.elements,
                dtype=torch.long,
            ),
            temp=torch.full(
                self.elements.shape,
                self.temp,
            ).float(),
        )

        data.batch = torch.zeros(
            data.num_nodes,
            dtype=torch.long,
        )
        data = data.to(
            self.device,
            non_blocking=True,
        )

        self._finite(data.prior, "sampled prior")

        return data

    @staticmethod
    def _min_pair_distance_torch(
        pos: torch.Tensor,
    ) -> float:
        # Nonperiodic diagnostic; not used for acceptance.
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(float("inf"))
        return d.min().item()

    @staticmethod
    @torch.no_grad()
    def propagate_flow(
        data: Data,
        model: FlowModule,
        flow_steps: int = 10,
        solver: str = "euler",
        tag: str = "flow",
        diagnostics=None,
    ) -> torch.Tensor:
        if flow_steps < 1:
            raise ValueError(
                "flow_steps must be >= 1"
            )

        if solver not in ("euler", "heun"):
            raise ValueError(
                f"Unsupported solver: {solver}"
            )

        mode = FlowSimulator._mode(model)

        if mode not in ("velocity", "data"):
            raise ValueError(
                f"Unsupported prediction mode: {mode}"
            )

        finite = FlowSimulator._finite

        def record(name, value, i=-1, t=None):
            if diagnostics is not None:
                diagnostics.record(
                    tag, name, value, i, t
                )

        record("prior", data["prior"])
        record("reference", data["positions_1"])

        if diagnostics is not None:
            for key in (
                "elements",
                "edge_index",
                "shifts",
            ):
                diagnostics.put(
                    f"{tag}_{key}",
                    data[key],
                )

        t_list = torch.linspace(
            0.0,
            1.0,
            flow_steps + 1,
            device=model.device,
            dtype=torch.float,
        )
        dt = t_list[1] - t_list[0]

        if FlowSimulator._signed(model):
            if tag == "corr" or mode != "velocity":
                raise ValueError(
                    "Signed-log supports propagator velocity mode only"
                )

            transform = getattr(
                model,
                "displacement_transform",
                None,
            )

            if transform is None:
                raise ValueError(
                    "Missing signed-log transform"
                )

            reference = data["positions_1"].clone()
            elements = data["elements"]

            try:
                z = transform.transform(
                    data["prior"],
                    elements,
                )
            except FloatingPointError as exc:
                raise NonFiniteFlowError(
                    f"{tag}: prior transform failed: {exc}"
                ) from exc

            record("initial_z", z)
            finite(z, f"{tag}: initial z")

            if diagnostics is not None:
                diagnostics.put(
                    f"{tag}_a_per_A",
                    transform.a_per_A[elements.long()],
                )

            def physical(state, i, t, stage):
                record(
                    f"{stage}_z",
                    state,
                    i,
                    t,
                )
                finite(
                    state,
                    f"{tag}: {stage} z, i={i}",
                )

                try:
                    disp = transform.inverse(
                        state,
                        elements,
                    )
                except FloatingPointError as exc:
                    index = int(
                        state.abs().reshape(-1).argmax().item()
                    )
                    atom, axis = divmod(index, 3)
                    element = int(
                        elements[atom].item()
                    )
                    value = float(
                        state[atom, axis].item()
                    )
                    a = float(
                        transform.a_per_A[element, axis].item()
                    )

                    # Largest |z| is a candidate, not a proven origin.
                    message = (
                        f"{tag}: inverse failed, "
                        f"flow_i={i}, t={float(t):.6f}, "
                        f"stage={stage}, "
                        f"max_abs_z={abs(value):.6e}, "
                        f"atom_index={atom}, "
                        f"element_id={element}, "
                        f"component={('x', 'y', 'z')[axis]}, "
                        f"z={value:.6e}, "
                        f"a_per_A={a:.6e}; {exc}"
                    )

                    raise NonFiniteFlowError(
                        message
                    ) from exc

                record(
                    f"{stage}_displacement",
                    disp,
                    i,
                    t,
                )
                finite(
                    disp,
                    f"{tag}: inverse displacement",
                )

                pos = reference + disp

                record(
                    f"{stage}_positions",
                    pos,
                    i,
                    t,
                )
                finite(
                    pos,
                    f"{tag}: physical coordinates",
                )

                return pos

            for i, t in enumerate(t_list[:-1]):
                data["positions_2"] = physical(
                    z,
                    i,
                    t,
                    "input",
                )

                velocity = model(
                    data,
                    t_per_graph=t.unsqueeze(0),
                )

                record(
                    "velocity",
                    velocity,
                    i,
                    t,
                )
                finite(
                    velocity,
                    f"{tag}: velocity step {i}",
                )

                proposal = z + dt * velocity

                record(
                    "proposal_z",
                    proposal,
                    i,
                    t_list[i + 1],
                )
                finite(
                    proposal,
                    f"{tag}: proposed z step {i}",
                )

                if solver == "euler":
                    z = proposal

                else:
                    next_data = data.clone()
                    next_data["positions_2"] = physical(
                        proposal,
                        i,
                        t_list[i + 1],
                        "heun_predictor",
                    )

                    next_velocity = model(
                        next_data,
                        t_per_graph=t_list[i + 1].unsqueeze(0),
                    )

                    record(
                        "heun_velocity",
                        next_velocity,
                        i,
                        t_list[i + 1],
                    )
                    finite(
                        next_velocity,
                        f"{tag}: Heun velocity step {i}",
                    )

                    z = (
                        z
                        + 0.5
                        * (velocity + next_velocity)
                        * dt
                    )

                    record(
                        "heun_updated_z",
                        z,
                        i,
                        t_list[i + 1],
                    )
                    finite(
                        z,
                        f"{tag}: Heun z step {i}",
                    )

            data["positions_2"] = physical(
                z,
                flow_steps,
                t_list[-1],
                "final",
            )

            return data["positions_2"]

        for i, t in enumerate(t_list[:-1]):
            record(
                "input_positions",
                data["positions_2"],
                i,
                t,
            )

            prediction = model(
                data,
                t_per_graph=t.unsqueeze(0),
            )

            record(
                "prediction",
                prediction,
                i,
                t,
            )
            finite(
                prediction,
                f"{tag}: prediction step {i}",
            )

            velocity = (
                prediction
                if mode == "velocity"
                else (
                    data["positions_1"]
                    + prediction
                    - data["positions_2"]
                ) / (1 - t)
            )

            record(
                "velocity",
                velocity,
                i,
                t,
            )
            finite(
                velocity,
                f"{tag}: physical velocity step {i}",
            )

            if solver == "euler":
                data["positions_2"] += velocity * dt

            else:
                next_t = (t + dt).unsqueeze(0)
                next_data = data.clone()
                next_data["positions_2"] += velocity * dt

                record(
                    "heun_proposal_positions",
                    next_data["positions_2"],
                    i,
                    next_t[0],
                )
                finite(
                    next_data["positions_2"],
                    f"{tag}: Heun proposal step {i}",
                )

                next_prediction = model(
                    next_data,
                    t_per_graph=next_t,
                )

                record(
                    "heun_prediction",
                    next_prediction,
                    i,
                    next_t[0],
                )
                finite(
                    next_prediction,
                    f"{tag}: Heun prediction step {i}",
                )

                if mode == "velocity":
                    next_velocity = next_prediction

                else:
                    if i == flow_steps - 1:
                        result = (
                            next_data["positions_1"]
                            + next_prediction
                        )

                        record(
                            "final_data_positions",
                            result,
                            i,
                            next_t[0],
                        )
                        finite(
                            result,
                            f"{tag}: final data prediction",
                        )

                        return result

                    next_velocity = (
                        next_data["positions_1"]
                        + next_prediction
                        - next_data["positions_2"]
                    ) / (1 - next_t)

                finite(
                    next_velocity,
                    f"{tag}: Heun physical velocity step {i}",
                )

                data["positions_2"] += (
                    0.5
                    * (velocity + next_velocity)
                    * dt
                )

            record(
                "updated_positions",
                data["positions_2"],
                i,
                t_list[i + 1],
            )
            finite(
                data["positions_2"],
                f"{tag}: updated coordinates step {i}",
            )

        return data["positions_2"]

    def _write_he_displacements_csv(
        self,
        previous: np.ndarray,
        final_positions: np.ndarray,
        use_corrector: bool,
    ):
        """Write He-labelled atom samples for an accepted run() step."""
        if self.csv_atom_indices.size == 0:
            return

        arrays = self.diagnostics.arrays
        previous = np.asarray(
            previous,
            dtype=np.float64,
        )

        stages = {
            "prior_raw": np.asarray(
                arrays["prop_sampled_prior"],
                dtype=np.float64,
            ),
            "prior_used": np.asarray(
                arrays["prop_prior"],
                dtype=np.float64,
            ),
            "before_corrector": (
                np.asarray(
                    arrays["prop_output_positions"],
                    dtype=np.float64,
                ) - previous
            ),
            "after_corrector": (
                np.asarray(
                    arrays["positions_before_com"],
                    dtype=np.float64,
                ) - previous
            ),
            "final": (
                np.asarray(
                    final_positions,
                    dtype=np.float64,
                ) - previous
            ),
        }
        
        # 20261116追加
        # Signed-log: final displacement before/after inverse transform.
        if self._signed(self.propagate_model):
            stages.update({
                "transformed_final": np.asarray(
                    arrays["prop_final_z"],
                    dtype=np.float64,
                ),
                "inverse_final": np.asarray(
                    arrays["prop_final_displacement"],
                    dtype=np.float64,
                ),
            })
        

        expected_shape = (
            len(self.atomic_numbers),
            3,
        )

        for name, values in stages.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name}: expected {expected_shape}, "
                    f"got {values.shape}"
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Nonfinite values in CSV stage: {name}"
                )

        path = self.displacement_csv_path
        write_header = (
            not path.exists()
            or path.stat().st_size == 0
        )

        header = [
            "run_id",
            "step",
            "attempt",
            "atom_index",
            "corrector_applied",
        ]

        for name in stages:
            header.extend(
                f"{name}_d{axis}"
                for axis in ("x", "y", "z")
            )

        with path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(header)

            for atom_index in self.csv_atom_indices:
                row = [
                    self._csv_run_id,
                    self.diagnostics.step,
                    self.diagnostics.attempt,
                    int(atom_index),
                    int(use_corrector),
                ]

                for values in stages.values():
                    row.extend(
                        float(value)
                        for value in values[atom_index]
                    )

                writer.writerow(row)

    @torch.no_grad()
    def step(
        self,
        positions: np.ndarray,
        flow_steps: int = 10,
        solver: str = "euler",
        use_corrector: bool = True,
    ) -> np.ndarray:
        # data = self.initialize_batch(
        #     positions,
        #     self.propagate_prior,
        # )
        
        # 20261112: 累積座標はunwrappedで保持し、
        # Propagatorへの開始入力だけをwrapする。
        # Keep the cumulative coordinates unwrapped.
        positions = np.asarray(positions)

        # Wrap only the starting structure supplied to the Propagator.
        positions_wrapped = wrap_positions(
            positions,
            cell=self.lattice,
            pbc=self.pbc,
            eps=0.0,
        )
        coordinate_shift = positions - positions_wrapped

        data = self.initialize_batch(
            positions_wrapped,
            self.propagate_prior,
        )
        
        

        self.diagnostics.put(
            "prop_sampled_prior",
            data["prior"],
        )

        if getattr(
            self.propagate_model,
            "use_scale",
            False,
        ):
            scale = self.propagate_model.scale_model(data)
            data["prior"] = data["prior"] * scale[:, None]

            self._finite(
                data["prior"],
                "scaled prior",
            )

        if not self._signed(self.propagate_model):
            data["positions_2"] = (
                data["positions_2"]
                + data["prior"]
            )

        new_positions = self.propagate_flow(
            data,
            self.propagate_model,
            flow_steps,
            solver,
            tag="prop",
            diagnostics=self.diagnostics,
        ).detach().cpu().numpy()
        
        # 20261112: 入力時に引いた座標シフトを戻し、
        # 出力を元のunwrapped座標表現に戻す。
        # Restore the original unwrapped coordinate representation.
        new_positions = new_positions + coordinate_shift

        self.diagnostics.record(
            "prop",
            "output_positions",
            new_positions,
        )
        self.diagnostics.record(
            "prop",
            "output_displacement",
            new_positions - positions,
        )

        if use_corrector:
            if (
                self.correct_model is None
                or self.correct_prior is None
            ):
                raise ValueError(
                    "Corrector requested but not configured"
                )

            prop_positions = new_positions.copy()

            data = self.initialize_batch(
                new_positions,
                self.correct_prior,
            )

            self.diagnostics.put(
                "corr_sampled_prior",
                data["prior"],
            )

            new_positions = self.propagate_flow(
                data,
                self.correct_model,
                flow_steps,
                solver,
                tag="corr",
                diagnostics=self.diagnostics,
            ).detach().cpu().numpy()

            self.diagnostics.record(
                "corr",
                "output_positions",
                new_positions,
            )
            self.diagnostics.record(
                "corr",
                "correction",
                new_positions - prop_positions,
            )

        return new_positions

    @torch.no_grad()
    def run(
        self,
        positions: np.ndarray,
        steps: int = 25,
        flow_steps: int = 10,
        solver: str = "euler",
        verbose: bool = False,
        fix_com: bool = True,
        num_retries: int = 2,
        corrector_every: int = 1,
    ) -> np.ndarray:
        if steps < 0 or num_retries < 0:
            raise ValueError(
                "steps and num_retries must be nonnegative"
            )

        if flow_steps < 1 or corrector_every < 1:
            raise ValueError(
                "flow_steps and corrector_every must be >= 1"
            )

        if solver not in ("euler", "heun"):
            raise ValueError(
                f"Unsupported solver: {solver}"
            )

        positions = np.asarray(positions).copy()

        if (
            positions.shape != (len(self.elements), 3)
            or not np.isfinite(positions).all()
        ):
            raise ValueError(
                "Invalid initial positions"
            )

        self._csv_run_id += 1

        print(
            f"[FlowSimulator] solver={solver}, "
            f"flow_steps={flow_steps}, "
            f"csv_run_id={self._csv_run_id}"
        )

        traj = [positions.copy()]
        iterator = (
            tqdm(range(steps))
            if verbose
            else range(steps)
        )

        for i_step in iterator:
            previous = positions.copy()
            success = False

            for i_retry in range(num_retries + 1):
                use_corrector = (
                    (i_step + 1) % corrector_every == 0
                    and self.correct_model is not None
                )

                self.diagnostics.begin(
                    i_step,
                    i_retry + 1,
                    previous,
                )

                for name, value in (
                    ("atomic_numbers", self.atomic_numbers),
                    ("elements", self.elements),
                    ("lattice", self.lattice),
                    ("masses", self.masses),
                    ("pbc", self.pbc),
                    ("temperature", self.temp),
                    ("solver", solver),
                    ("flow_steps", flow_steps),
                    ("use_corrector", use_corrector),
                    ("fix_com", fix_com),
                    ("csv_run_id", self._csv_run_id),
                ):
                    self.diagnostics.put(
                        name,
                        value,
                    )

                try:
                    candidate = self.step(
                        previous,
                        flow_steps,
                        solver,
                        use_corrector,
                    )

                    self.diagnostics.put(
                        "positions_before_com",
                        candidate,
                    )

                    if fix_com:
                        diff = candidate - previous
                        com_diff = (
                            np.sum(
                                diff * self.masses[:, None],
                                axis=0,
                            )
                            / np.sum(self.masses)
                        )

                        self.diagnostics.put(
                            "com_diff",
                            com_diff,
                        )

                        candidate -= com_diff

                    self.diagnostics.put(
                        "positions_after_com",
                        candidate,
                    )

                    if not np.isfinite(candidate).all():
                        raise NonFiniteFlowError(
                            "Nonfinite final coordinates"
                        )

                    if (
                        np.abs(candidate - previous).max()
                        > 1e3
                    ):
                        raise NonFiniteFlowError(
                            "Displacement component exceeded 1000 A"
                        )

                except NonFiniteFlowError as exc:
                    self.diagnostics.save(exc)

                    print(
                        f"[FlowSimulator] step={i_step}, "
                        f"attempt={i_retry + 1}/"
                        f"{num_retries + 1}: {exc}"
                    )

                    continue

                self.diagnostics.save()

                self._write_he_displacements_csv(
                    previous=previous,
                    final_positions=candidate,
                    use_corrector=use_corrector,
                )

                positions = candidate
                success = True
                break

            if not success:
                print(
                    "[FlowSimulator] Stopped early: "
                    f"completed {len(traj) - 1}/{steps} steps"
                )
                break

            traj.append(positions.copy())

        return traj