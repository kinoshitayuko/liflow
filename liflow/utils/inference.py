"""Flow simulation with signed-log integration, retries and failure diagnostics."""
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
    AdaptiveMaxwellBoltzmannPrior, MaxwellBoltzmannPrior,
    NormalPrior, Prior, UniformScaleNormalPrior,
)


class NonFiniteFlowError(RuntimeError):
    """Numerical failure eligible for retry; configuration errors are not caught."""


class FlowDiagnostics:
    """Store scalar history plus latest tensor per name, without retaining GPU tensors."""
    def __init__(self):
        self.directory = Path(tempfile.mkdtemp(prefix="flow_debug_", dir="."))
        self.step, self.attempt = -1, 0
        self.arrays, self.trace = {}, []
        print(f"[FlowDiagnostics] output={self.directory.resolve()}")

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
        maximum, index = None, None
        if valid.any():
            maximum = float(absolute[valid].max())
            index = [int(x) for x in np.unravel_index(
                np.where(valid, absolute, -1).argmax(), array.shape)]
        time = float(t.item()) if isinstance(t, torch.Tensor) else t
        self.trace.append(dict(step=self.step, attempt=self.attempt, tag=tag,
            name=name, flow_i=int(i), t=time, max_abs=maximum,
            index=index, nonfinite=int((~valid).sum())))

    def begin(self, step, attempt, positions):
        self.step, self.attempt = step, attempt
        self.arrays, self.trace = {}, []
        self.put("positions_start", positions)

    def save(self, error=None):
        stem = (self.directory / "last_success" if error is None else
                self.directory / f"failure_step{self.step:06d}_attempt{self.attempt:02d}")
        np.savez_compressed(str(stem) + ".npz", **self.arrays)
        with open(str(stem) + ".json", "w", encoding="utf-8") as f:
            json.dump(dict(step=self.step, attempt=self.attempt,
                error=None if error is None else str(error), trace=self.trace),
                f, indent=2, allow_nan=False)
        if error is not None:
            print(f"[FlowDiagnostics] saved: {stem}")
            for row in self.trace[-16:]:
                print(f"  [trace] {row}")


class FlowSimulator:
    def __init__(
        self, propagate_model: FlowModule, propagate_prior: Prior,
        atomic_numbers: np.ndarray, element_idx: np.ndarray,
        lattice: np.ndarray, temp: float,
        correct_model: FlowModule | None = None,
        correct_prior: Prior | None = None, pbc: bool = True,
        scale_Li_index: int | None = None,
        scale_frame_index: int | None = None,
        fix_frame_species: list[int] | None = None,
    ):
        self.propagate_model = propagate_model
        self.propagate_prior = propagate_prior
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        lookup = np.asarray(element_idx)
        if (lookup.ndim != 1 or self.atomic_numbers.ndim != 1
                or self.atomic_numbers.size == 0
                or np.any(self.atomic_numbers < 0)
                or np.any(self.atomic_numbers >= len(lookup))):
            raise ValueError("Invalid atomic numbers or element lookup table")
        self.elements = lookup[self.atomic_numbers]
        if not np.issubdtype(self.elements.dtype, np.integer) or np.any(self.elements < 0):
            raise ValueError("Element lookup must contain supported integer IDs")
        self.masses = ase.data.atomic_masses[self.atomic_numbers]
        if not np.isfinite(self.masses).all() or np.any(self.masses <= 0):
            raise ValueError("Invalid masses")
        self.lattice = np.asarray(lattice)
        self.temp = temp
        self.correct_model, self.correct_prior = correct_model, correct_prior
        self.pbc = pbc
        self.device = propagate_model.device
        self.scale_Li_index, self.scale_frame_index = scale_Li_index, scale_frame_index
        # API compatibility only: the supplied implementation did not apply this mask.
        self.frame_mask = (np.isin(self.atomic_numbers, fix_frame_species)
                           if fix_frame_species is not None else None)
        if (correct_model is None) != (correct_prior is None):
            raise ValueError("Corrector model and prior must be supplied together")
        self._validate_model(propagate_model, "prop")
        if correct_model is not None:
            if torch.device(correct_model.device) != torch.device(self.device):
                raise ValueError("Propagator and corrector must use the same device")
            self._validate_model(correct_model, "corr")
        self.diagnostics = FlowDiagnostics()
        for label, model in (("Propagator", propagate_model), ("Corrector", correct_model)):
            if model is not None:
                space = "transformed velocity" if self._signed(model) else "physical-space flow"
                print(f"[FlowSimulator] {label}: signed_log={self._signed(model)}, "
                      f"prediction_mode={self._mode(model)}, space={space}")

    @staticmethod
    def _signed(model):
        return bool(getattr(model, "signed_log_enabled", False))

    @staticmethod
    def _mode(model):
        return str(model.cfg.model.get("prediction_mode", "velocity"))

    @staticmethod
    def _finite(value, context):
        if not torch.isfinite(value).all().item():
            raise NonFiniteFlowError(f"Nonfinite values in {context}")

    def _validate_model(self, model, tag):
        mode = self._mode(model)
        if mode not in ("velocity", "data"):
            raise ValueError(f"Unsupported prediction mode: {mode}")
        if self._signed(model):
            if tag == "corr" or mode != "velocity":
                raise ValueError("Signed-log supports propagator velocity mode only")
            transform = getattr(model, "displacement_transform", None)
            if transform is None:
                raise ValueError("Signed-log checkpoint has no displacement_transform")
            probe = torch.zeros((len(self.elements), 3), device=model.device)
            ids = torch.as_tensor(self.elements, dtype=torch.long, device=model.device)
            with torch.no_grad():
                z = transform.transform(probe, ids)
                self._finite(z, "transform validation")
                self._finite(transform.inverse(z, ids), "inverse validation")

    def initialize_batch(self, positions: np.ndarray, prior: Prior) -> Data:
        positions = np.asarray(positions)
        if positions.shape != (len(self.elements), 3):
            raise ValueError("Positions must have shape (n_atoms, 3)")
        if not np.isfinite(positions).all():
            raise NonFiniteFlowError("Nonfinite input positions")
        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=[positions], lattice=self.lattice,
            cutoff=self.propagate_model.cfg.data.cutoff, periodic=self.pbc,
        )
        if isinstance(prior, NormalPrior):
            sampled = prior.sample(shape=positions.shape)
        elif isinstance(prior, MaxwellBoltzmannPrior):
            sampled = prior.sample(temperature=self.temp, masses=self.masses,
                                   shape=positions.shape)
        elif isinstance(prior, AdaptiveMaxwellBoltzmannPrior):
            sampled = prior.sample(
                temperature=self.temp, atomic_numbers=self.atomic_numbers,
                masses=self.masses, scale_Li_index=self.scale_Li_index,
                scale_frame_index=self.scale_frame_index, shape=positions.shape,
            )
        elif isinstance(prior, UniformScaleNormalPrior):
            sampled = prior.sample(shape=positions.shape)
        else:
            raise NotImplementedError(f"Unsupported prior: {type(prior).__name__}")
        self.diagnostics.put("latest_sampled_prior", sampled)
        data = Data(
            num_nodes=len(self.elements),
            positions_1=torch.tensor(positions, dtype=torch.float),
            prior=torch.tensor(sampled, dtype=torch.float),
            positions_2=torch.tensor(positions, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            shifts=torch.tensor(shifts, dtype=torch.float),
            elements=torch.tensor(self.elements, dtype=torch.long),
            temp=torch.full(self.elements.shape, self.temp).float(),
        )
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        data = data.to(self.device, non_blocking=True)
        self._finite(data.prior, "sampled prior")
        return data

    @staticmethod
    def _min_pair_distance_torch(pos: torch.Tensor) -> float:
        # Nonperiodic diagnostic only; not used to accept/reject configurations.
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(float("inf"))
        return d.min().item()

    @staticmethod
    @torch.no_grad()
    def propagate_flow(data: Data, model: FlowModule, flow_steps: int = 10,
                       solver: str = "euler", tag: str = "flow",
                       diagnostics=None) -> torch.Tensor:
        if flow_steps < 1:
            raise ValueError("flow_steps must be >= 1")
        if solver not in ("euler", "heun"):
            raise ValueError(f"Unsupported solver: {solver}")
        mode = FlowSimulator._mode(model)
        if mode not in ("velocity", "data"):
            raise ValueError(f"Unsupported prediction mode: {mode}")
        finite = FlowSimulator._finite
        def record(name, value, i=-1, t=None):
            if diagnostics is not None:
                diagnostics.record(tag, name, value, i, t)
        record("prior", data["prior"])
        record("reference", data["positions_1"])
        if diagnostics is not None:
            for key in ("elements", "edge_index", "shifts"):
                diagnostics.put(f"{tag}_{key}", data[key])
        t_list = torch.linspace(0.0, 1.0, flow_steps + 1,
                                device=model.device, dtype=torch.float)
        dt = t_list[1] - t_list[0]
        if FlowSimulator._signed(model):
            if tag == "corr" or mode != "velocity":
                raise ValueError("Signed-log supports propagator velocity mode only")
            transform = getattr(model, "displacement_transform", None)
            if transform is None:
                raise ValueError("Missing signed-log transform")
            reference, elements = data["positions_1"].clone(), data["elements"]
            try:
                z = transform.transform(data["prior"], elements)
            except FloatingPointError as exc:
                raise NonFiniteFlowError(f"{tag}: prior transform failed: {exc}") from exc
            record("initial_z", z)
            finite(z, f"{tag}: initial z")
            if diagnostics is not None:
                diagnostics.put(f"{tag}_a_per_A", transform.a_per_A[elements.long()])

            def physical(state, i, t, stage):
                record(f"{stage}_z", state, i, t)
                finite(state, f"{tag}: {stage} z, i={i}")
                try:
                    disp = transform.inverse(state, elements)
                except FloatingPointError as exc:
                    index = int(state.abs().reshape(-1).argmax().item())
                    atom, axis = divmod(index, 3)
                    element = int(elements[atom].item())
                    value = float(state[atom, axis].item())
                    a = float(transform.a_per_A[element, axis].item())
                    # Largest |z| is a diagnostic candidate, not proven overflow origin.
                    message = (f"{tag}: inverse failed, flow_i={i}, t={float(t):.6f}, "
                        f"stage={stage}, max_abs_z={abs(value):.6e}, "
                        f"atom_index={atom}, element_id={element}, "
                        f"component={('x', 'y', 'z')[axis]}, z={value:.6e}, "
                        f"a_per_A={a:.6e}; {exc}")
                    raise NonFiniteFlowError(message) from exc
                record(f"{stage}_displacement", disp, i, t)
                finite(disp, f"{tag}: inverse displacement")
                pos = reference + disp
                record(f"{stage}_positions", pos, i, t)
                finite(pos, f"{tag}: physical coordinates")
                return pos

            for i, t in enumerate(t_list[:-1]):
                data["positions_2"] = physical(z, i, t, "input")
                velocity = model(data, t_per_graph=t.unsqueeze(0))
                record("velocity", velocity, i, t)
                finite(velocity, f"{tag}: velocity step {i}")
                proposal = z + dt * velocity
                record("proposal_z", proposal, i, t_list[i + 1])
                finite(proposal, f"{tag}: proposed z step {i}")
                if solver == "euler":
                    z = proposal
                else:
                    next_data = data.clone()
                    next_data["positions_2"] = physical(proposal, i, t_list[i + 1], "heun_predictor")
                    next_velocity = model(next_data, t_per_graph=t_list[i + 1].unsqueeze(0))
                    record("heun_velocity", next_velocity, i, t_list[i + 1])
                    finite(next_velocity, f"{tag}: Heun velocity step {i}")
                    z = z + 0.5 * (velocity + next_velocity) * dt
                    record("heun_updated_z", z, i, t_list[i + 1])
                    finite(z, f"{tag}: Heun z step {i}")
            data["positions_2"] = physical(z, flow_steps, t_list[-1], "final")
            return data["positions_2"]

        for i, t in enumerate(t_list[:-1]):
            record("input_positions", data["positions_2"], i, t)
            prediction = model(data, t_per_graph=t.unsqueeze(0))
            record("prediction", prediction, i, t)
            finite(prediction, f"{tag}: prediction step {i}")
            velocity = (prediction if mode == "velocity" else
                (data["positions_1"] + prediction - data["positions_2"]) / (1 - t))
            record("velocity", velocity, i, t)
            finite(velocity, f"{tag}: physical velocity step {i}")
            if solver == "euler":
                data["positions_2"] += velocity * dt
            else:
                next_t = (t + dt).unsqueeze(0)
                next_data = data.clone()
                next_data["positions_2"] += velocity * dt
                record("heun_proposal_positions", next_data["positions_2"], i, next_t[0])
                finite(next_data["positions_2"], f"{tag}: Heun proposal step {i}")
                next_prediction = model(next_data, t_per_graph=next_t)
                record("heun_prediction", next_prediction, i, next_t[0])
                finite(next_prediction, f"{tag}: Heun prediction step {i}")
                if mode == "velocity":
                    next_velocity = next_prediction
                else:
                    if i == flow_steps - 1:
                        result = next_data["positions_1"] + next_prediction
                        record("final_data_positions", result, i, next_t[0])
                        finite(result, f"{tag}: final data prediction")
                        return result
                    next_velocity = (next_data["positions_1"] + next_prediction
                                     - next_data["positions_2"]) / (1 - next_t)
                finite(next_velocity, f"{tag}: Heun physical velocity step {i}")
                data["positions_2"] += 0.5 * (velocity + next_velocity) * dt
            record("updated_positions", data["positions_2"], i, t_list[i + 1])
            finite(data["positions_2"], f"{tag}: updated coordinates step {i}")
        return data["positions_2"]

    @torch.no_grad()
    def step(self, positions: np.ndarray, flow_steps: int = 10,
             solver: str = "euler", use_corrector: bool = True) -> np.ndarray:
        data = self.initialize_batch(positions, self.propagate_prior)
        self.diagnostics.put("prop_sampled_prior", data["prior"])
        if getattr(self.propagate_model, "use_scale", False):
            scale = self.propagate_model.scale_model(data)
            data["prior"] = data["prior"] * scale[:, None]
            self._finite(data["prior"], "scaled prior")
        if not self._signed(self.propagate_model):
            data["positions_2"] = data["positions_2"] + data["prior"]
        new_positions = self.propagate_flow(data, self.propagate_model,
            flow_steps, solver, tag="prop", diagnostics=self.diagnostics).detach().cpu().numpy()
        self.diagnostics.record("prop", "output_positions", new_positions)
        self.diagnostics.record("prop", "output_displacement", new_positions - positions)
        if use_corrector:
            if self.correct_model is None or self.correct_prior is None:
                raise ValueError("Corrector requested but not configured")
            prop_positions = new_positions.copy()
            data = self.initialize_batch(new_positions, self.correct_prior)
            self.diagnostics.put("corr_sampled_prior", data["prior"])
            new_positions = self.propagate_flow(data, self.correct_model,
                flow_steps, solver, tag="corr", diagnostics=self.diagnostics).detach().cpu().numpy()
            self.diagnostics.record("corr", "output_positions", new_positions)
            self.diagnostics.record("corr", "correction", new_positions - prop_positions)
        return new_positions

    @torch.no_grad()
    def run(self, positions: np.ndarray, steps: int = 25,
            flow_steps: int = 10, solver: str = "euler", verbose: bool = False,
            fix_com: bool = True, num_retries: int = 2,
            corrector_every: int = 1) -> np.ndarray:
        if steps < 0 or num_retries < 0:
            raise ValueError("steps and num_retries must be nonnegative")
        if flow_steps < 1 or corrector_every < 1:
            raise ValueError("flow_steps and corrector_every must be >= 1")
        if solver not in ("euler", "heun"):
            raise ValueError(f"Unsupported solver: {solver}")
        positions = np.asarray(positions).copy()
        if positions.shape != (len(self.elements), 3) or not np.isfinite(positions).all():
            raise ValueError("Invalid initial positions")
        print(f"[FlowSimulator] solver={solver}, flow_steps={flow_steps}")
        traj = [positions.copy()]
        iterator = tqdm(range(steps)) if verbose else range(steps)
        for i_step in iterator:
            previous = positions.copy()
            success = False
            for i_retry in range(num_retries + 1):
                use_corrector = ((i_step + 1) % corrector_every == 0
                                 and self.correct_model is not None)
                self.diagnostics.begin(i_step, i_retry + 1, previous)
                for name, value in (("atomic_numbers", self.atomic_numbers),
                    ("elements", self.elements), ("lattice", self.lattice),
                    ("masses", self.masses), ("pbc", self.pbc), ("temperature", self.temp),
                    ("solver", solver), ("flow_steps", flow_steps),
                    ("use_corrector", use_corrector), ("fix_com", fix_com)):
                    self.diagnostics.put(name, value)
                try:
                    candidate = self.step(previous, flow_steps, solver, use_corrector)
                    self.diagnostics.put("positions_before_com", candidate)
                    if fix_com:
                        diff = candidate - previous
                        com_diff = np.sum(diff * self.masses[:, None], axis=0) / np.sum(self.masses)
                        self.diagnostics.put("com_diff", com_diff)
                        candidate -= com_diff
                    self.diagnostics.put("positions_after_com", candidate)
                    if not np.isfinite(candidate).all():
                        raise NonFiniteFlowError("Nonfinite final coordinates")
                    if np.abs(candidate - previous).max() > 1e3:
                        raise NonFiniteFlowError("Displacement component exceeded 1000 A")
                except NonFiniteFlowError as exc:
                    self.diagnostics.save(exc)
                    print(f"[FlowSimulator] step={i_step}, attempt={i_retry + 1}/"
                          f"{num_retries + 1}: {exc}")
                    continue
                self.diagnostics.save()
                positions = candidate
                success = True
                break
            if not success:
                print(f"[FlowSimulator] Stopped early: completed {len(traj) - 1}/{steps} steps")
                break
            traj.append(positions.copy())
        return traj





# """Flow simulation with optional signed-log propagator integration."""
# #20261104 作成
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
#     """A numerical failure that can be retried with a fresh prior."""


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
#         element_idx = np.asarray(element_idx)
#         if (self.atomic_numbers.ndim != 1 or self.atomic_numbers.size == 0
#                 or np.any(self.atomic_numbers < 0)
#                 or np.any(self.atomic_numbers >= len(element_idx))):
#             raise ValueError("Invalid atomic numbers or element lookup table")
#         self.elements = element_idx[self.atomic_numbers]
#         if np.any(self.elements < 0):
#             raise ValueError("Element lookup contains unsupported IDs")
#         self.masses = ase.data.atomic_masses[self.atomic_numbers]
#         self.lattice = lattice
#         self.temp = temp
#         self.correct_model = correct_model
#         self.correct_prior = correct_prior
#         self.pbc = pbc
#         self.device = propagate_model.device
#         self.scale_Li_index = scale_Li_index
#         self.scale_frame_index = scale_frame_index
#         # Kept for API compatibility; no coordinate constraint is applied.
#         self.frame_mask = (np.isin(self.atomic_numbers, fix_frame_species)
#                            if fix_frame_species is not None else None)
#         if (correct_model is None) != (correct_prior is None):
#             raise ValueError("Corrector model and prior must be supplied together")
#         self._validate_model(propagate_model, "prop")
#         if correct_model is not None:
#             if torch.device(correct_model.device) != torch.device(self.device):
#                 raise ValueError("Propagator and corrector must use the same device")
#             self._validate_model(correct_model, "corr")
#         for label, model in (("Propagator", propagate_model),
#                              ("Corrector", correct_model)):
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

#     def _validate_model(self, model, tag):
#         mode = self._mode(model)
#         if mode not in ("velocity", "data"):
#             raise ValueError(f"Unsupported prediction mode: {mode}")
#         if self._signed(model):
#             if tag == "corr":
#                 raise ValueError("Signed-log correctors are not supported")
#             if mode != "velocity":
#                 raise ValueError("Signed-log integration requires velocity mode")
#             transform = getattr(model, "displacement_transform", None)
#             if transform is None:
#                 raise ValueError("Signed-log checkpoint has no displacement_transform")
#             # Validate every requested ID through the public transform API.
#             probe = torch.zeros((len(self.elements), 3), device=model.device)
#             ids = torch.as_tensor(self.elements, dtype=torch.long, device=model.device)
#             with torch.no_grad():
#                 z = transform.transform(probe, ids)
#                 self._finite(z, "transform validation")
#                 self._finite(transform.inverse(z, ids), "inverse validation")

#     @staticmethod
#     def _finite(value, context):
#         if not torch.isfinite(value).all().item():
#             raise NonFiniteFlowError(f"Nonfinite values in {context}")

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
#             # Also dispatches to GaussianMixtureAdaptivePrior.sample().
#             sampled = prior.sample(
#                 temperature=self.temp, atomic_numbers=self.atomic_numbers,
#                 masses=self.masses, scale_Li_index=self.scale_Li_index,
#                 scale_frame_index=self.scale_frame_index, shape=positions.shape,
#             )
#         elif isinstance(prior, UniformScaleNormalPrior):
#             sampled = prior.sample(shape=positions.shape)
#         else:
#             raise NotImplementedError(f"Unsupported prior: {type(prior).__name__}")
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
#         d = torch.cdist(pos, pos)
#         d.fill_diagonal_(float("inf"))
#         return d.min().item()

#     @staticmethod
#     @torch.no_grad()
#     def propagate_flow(data: Data, model: FlowModule, flow_steps: int = 10,
#                        solver: str = "euler", tag: str = "flow") -> torch.Tensor:
#         if flow_steps < 1:
#             raise ValueError("flow_steps must be >= 1")
#         if solver not in ("euler", "heun"):
#             raise ValueError(f"Unsupported solver: {solver}")
#         mode = FlowSimulator._mode(model)
#         if mode not in ("velocity", "data"):
#             raise ValueError(f"Unsupported prediction mode: {mode}")
#         finite = FlowSimulator._finite
#         t_list = torch.linspace(0.0, 1.0, flow_steps + 1,
#                                 device=model.device, dtype=torch.float)
#         dt = t_list[1] - t_list[0]

#         if FlowSimulator._signed(model):
#             if tag == "corr" or mode != "velocity":
#                 raise ValueError("Signed-log supports propagator velocity mode only")
#             transform = getattr(model, "displacement_transform", None)
#             if transform is None:
#                 raise ValueError("Missing signed-log transform")
#             reference = data["positions_1"].clone()
#             elements = data["elements"]
#             z = transform.transform(data["prior"], elements)
#             finite(z, f"{tag}: initial z")

#             # def physical(state):
#             #     disp = transform.inverse(state, elements)
#             #     finite(disp, f"{tag}: inverse displacement")
#             #     pos = reference + disp
#             #     finite(pos, f"{tag}: physical coordinates")
#             #     return pos
#             def physical(state):
#                 # Input-state failures are also handled by the retry loop.
#                 finite(state, f"{tag}: transformed state before inverse")

#                 try:
#                     disp = transform.inverse(state, elements)

#                 except FloatingPointError as exc:
#                     # Diagnostic information only; do not clip or modify state.
#                     abs_state = state.detach().abs()
#                     flat_index = int(abs_state.reshape(-1).argmax().item())
#                     atom_index, component_index = divmod(flat_index, 3)

#                     element_id = int(elements[atom_index].item())
#                     component = ("x", "y", "z")[component_index]
#                     z_value = float(
#                         state[atom_index, component_index].item()
#                     )

#                     # This identifies the largest |z|, not necessarily
#                     # the exact component that overflowed after division by a.
#                     detail = (
#                         f"{tag}: signed-log inverse failed; "
#                         f"max_abs_z={abs(z_value):.6e}, "
#                         f"atom_index={atom_index}, "
#                         f"element_id={element_id}, "
#                         f"component={component}, "
#                         f"z={z_value:.6e}"
#                     )

#                     coefficients = getattr(transform, "a_per_A", None)
#                     if coefficients is not None:
#                         a_value = float(
#                             coefficients[
#                                 element_id, component_index
#                             ].item()
#                         )
#                         detail += f", a_per_A={a_value:.6e}"

#                     raise NonFiniteFlowError(
#                         f"{detail}; original_error={exc}"
#                     ) from exc

#                 finite(disp, f"{tag}: inverse displacement")
#                 pos = reference + disp
#                 finite(pos, f"{tag}: physical coordinates")
#                 return pos
            
            

#             for i, t in enumerate(t_list[:-1]):
#                 data["positions_2"] = physical(z)
#                 velocity = model(data, t_per_graph=t.unsqueeze(0))
#                 finite(velocity, f"{tag}: velocity step {i}")
#                 proposal = z + dt * velocity
#                 finite(proposal, f"{tag}: proposed z step {i}")
#                 if solver == "euler":
#                     z = proposal
#                 else:
#                     next_data = data.clone()
#                     next_data["positions_2"] = physical(proposal)
#                     next_velocity = model(next_data, t_per_graph=t_list[i + 1].unsqueeze(0))
#                     finite(next_velocity, f"{tag}: Heun velocity step {i}")
#                     z = z + 0.5 * (velocity + next_velocity) * dt
#                     finite(z, f"{tag}: Heun z step {i}")
#             data["positions_2"] = physical(z)
#             return data["positions_2"]

#         # Original physical-space integration, including corrector initialization.
#         for i, t in enumerate(t_list[:-1]):
#             prediction = model(data, t_per_graph=t.unsqueeze(0))
#             finite(prediction, f"{tag}: prediction step {i}")
#             if mode == "velocity":
#                 velocity = prediction
#             else:
#                 velocity = (data["positions_1"] + prediction - data["positions_2"]) / (1 - t)
#             finite(velocity, f"{tag}: physical velocity step {i}")
#             if solver == "euler":
#                 data["positions_2"] += velocity * dt
#             else:
#                 next_t = (t + dt).unsqueeze(0)
#                 next_data = data.clone()
#                 next_data["positions_2"] += velocity * dt
#                 finite(next_data["positions_2"], f"{tag}: Heun proposal step {i}")
#                 next_prediction = model(next_data, t_per_graph=next_t)
#                 finite(next_prediction, f"{tag}: Heun prediction step {i}")
#                 if mode == "velocity":
#                     next_velocity = next_prediction
#                 else:
#                     if i == flow_steps - 1:
#                         result = next_data["positions_1"] + next_prediction
#                         finite(result, f"{tag}: final data prediction")
#                         return result
#                     next_velocity = (next_data["positions_1"] + next_prediction
#                                      - next_data["positions_2"]) / (1 - next_t)
#                 finite(next_velocity, f"{tag}: Heun physical velocity step {i}")
#                 data["positions_2"] += 0.5 * (velocity + next_velocity) * dt
#             finite(data["positions_2"], f"{tag}: updated coordinates step {i}")
#         return data["positions_2"]

#     @torch.no_grad()
#     def step(self, positions: np.ndarray, flow_steps: int = 10,
#              solver: str = "euler", use_corrector: bool = True) -> np.ndarray:
#         data = self.initialize_batch(positions, self.propagate_prior)
#         if getattr(self.propagate_model, "use_scale", False):
#             scale = self.propagate_model.scale_model(data)
#             data["prior"] = data["prior"] * scale[:, None]
#             self._finite(data["prior"], "scaled prior")
#         if not self._signed(self.propagate_model):
#             data["positions_2"] = data["positions_2"] + data["prior"]
#         new_positions = self.propagate_flow(
#             data, self.propagate_model, flow_steps, solver, tag="prop"
#         ).detach().cpu().numpy()
#         if use_corrector:
#             if self.correct_model is None or self.correct_prior is None:
#                 raise ValueError("Corrector requested but not configured")
#             data = self.initialize_batch(new_positions, self.correct_prior)
#             new_positions = self.propagate_flow(
#                 data, self.correct_model, flow_steps, solver, tag="corr"
#             ).detach().cpu().numpy()
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
#                 try:
#                     candidate = self.step(previous, flow_steps, solver, use_corrector)
#                     if fix_com:
#                         diff = candidate - previous
#                         com_diff = np.sum(diff * self.masses[:, None], axis=0) / np.sum(self.masses)
#                         candidate -= com_diff
#                     if not np.isfinite(candidate).all():
#                         raise NonFiniteFlowError("Nonfinite final coordinates")
#                     if np.abs(candidate - previous).max() > 1e3:
#                         raise NonFiniteFlowError("Displacement component exceeded 1000 A")
#                 except NonFiniteFlowError as exc:
#                     print(f"[FlowSimulator] step={i_step}, attempt={i_retry + 1}/"
#                           f"{num_retries + 1}: {exc}")
#                     continue
#                 positions = candidate
#                 success = True
#                 break
#             if not success:
#                 print(f"[FlowSimulator] Stopped early: completed {len(traj) - 1}/{steps} steps")
#                 break
#             traj.append(positions.copy())
#         return traj





# #csvファイルを作成しない推論ファイル。fix_frame_speciesを指定して、フレーム固定の推論を行うが、今は使ってない。
# #当分はこれでいい 20261104
# import ase
# import numpy as np
# import torch
# from torch_geometric.data import Data
# from tqdm import tqdm

# from liflow.model.modules import FlowModule
# from liflow.utils.geometry import get_neighbor_list_batch
# from liflow.utils.prior import (
#     AdaptiveMaxwellBoltzmannPrior,
#     MaxwellBoltzmannPrior,
#     NormalPrior,
#     Prior,
#     UniformScaleNormalPrior,   # 20260609追加
# )


# class FlowSimulator:
#     def __init__(
#         self,
#         propagate_model: FlowModule,
#         propagate_prior: Prior,
#         atomic_numbers: np.ndarray,
#         element_idx: np.ndarray,
#         lattice: np.ndarray,
#         temp: float,
#         correct_model: FlowModule | None = None,
#         correct_prior: Prior | None = None,
#         pbc: bool = True,
#         scale_Li_index: int | None = None,  # AdaptiveMaxwellBoltzmannPrior
#         scale_frame_index: int | None = None,  # AdaptiveMaxwellBoltzmannPrior
#         fix_frame_species: list[int] | None = None,  # ★ 20260526追加
#     ):
#         self.propagate_model = propagate_model
#         self.propagate_prior = propagate_prior
#         self.atomic_numbers = atomic_numbers
#         # elements: atomic numbers -> element indices in the model
#         self.elements = element_idx[atomic_numbers]
#         self.masses = ase.data.atomic_masses[atomic_numbers]
#         self.lattice = lattice
#         self.temp = temp
#         self.correct_model = correct_model
#         self.correct_prior = correct_prior
#         self.pbc = pbc
#         self.device = propagate_model.device
#         self.scale_Li_index = scale_Li_index
#         self.scale_frame_index = scale_frame_index
#         # ★ 20260526追加: フレーム固定マスクを事前計算
#         if fix_frame_species is not None:
#             self.frame_mask = np.isin(atomic_numbers, fix_frame_species)
#         else:
#             self.frame_mask = None

#     def initialize_batch(self, positions: np.ndarray, prior: Prior) -> Data:
#         # Get edge information
#         edge_index, shifts = get_neighbor_list_batch(
#             positions_batch=[positions],
#             lattice=self.lattice,
#             cutoff=self.propagate_model.cfg.data.cutoff,
#             periodic=self.pbc,
#         )

#         # Sample prior
#         if isinstance(prior, NormalPrior):
#             _prior = prior.sample(shape=positions.shape)
#         elif isinstance(prior, MaxwellBoltzmannPrior):
#             _prior = prior.sample(
#                 temperature=self.temp, masses=self.masses, shape=positions.shape
#             )
#         elif isinstance(prior, AdaptiveMaxwellBoltzmannPrior):
#             _prior = prior.sample(
#                 temperature=self.temp,
#                 atomic_numbers=self.atomic_numbers,
#                 masses=self.masses,
#                 scale_Li_index=self.scale_Li_index,
#                 scale_frame_index=self.scale_frame_index,
#                 shape=positions.shape,
#             )
#         elif isinstance(prior, UniformScaleNormalPrior):
#             _prior = prior.sample(shape=positions.shape)  # 20260609追加
#         else:
#             raise NotImplementedError

#         # Return data object
#         data = Data(
#             num_nodes=len(self.elements),
#             positions_1=torch.tensor(positions, dtype=torch.float),
#             prior=torch.tensor(_prior, dtype=torch.float),
#             positions_2=torch.tensor(positions, dtype=torch.float),
#             edge_index=torch.tensor(edge_index, dtype=torch.long),
#             shifts=torch.tensor(shifts, dtype=torch.float),
#             elements=torch.tensor(self.elements, dtype=torch.long),
#             temp=torch.full(self.elements.shape, self.temp).float(),
#         )
#         data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
#         return data.to(self.device, non_blocking=True)

#     #20260609追加
#     @staticmethod
#     def _min_pair_distance_torch(pos: torch.Tensor) -> float:
#         d = torch.cdist(pos, pos)
#         n = d.shape[0]
#         d[torch.arange(n), torch.arange(n)] = float("inf")
#         return d.min().item()
    
#     @staticmethod
#     @torch.no_grad()
#     def propagate_flow(
#         data: Data, model: FlowModule, flow_steps: int = 10, solver: str = "euler", tag: str = "flow",
#     ) -> torch.Tensor:
#         # Time grid
#         t_list = torch.linspace(
#             0.0, 1.0, flow_steps + 1, device=model.device, dtype=torch.float
#         )
#         step_dt = t_list[1] - t_list[0]
#         if hasattr(model.cfg.model, "prediction_mode"):
#             prediction_mode = model.cfg.model.prediction_mode
#         else:
#             prediction_mode = "velocity"
            
#         # 20260707追加: atom127の近傍原子を特定
#         # target_atom = 127
#         # neighbors_of_target = data['edge_index'][1][data['edge_index'][0] == target_atom]
        

#         # Propagate flow
#         for i, t in enumerate(t_list[:-1]):
#             if prediction_mode == "velocity":
#                 pred_dx_dt = model(data, t_per_graph=t.unsqueeze(0))
#             elif prediction_mode == "data":
#                 pred_disp = model(data, t_per_graph=t.unsqueeze(0))
#                 pred_dx_dt = (data["positions_1"] + pred_disp - data["positions_2"]) / (
#                     1 - t
#                 )
#             else:
#                 raise NotImplementedError
        
#             # # ここに追加 20260707
#             # if neighbors_of_target.numel() > 0:
#             #     dist_127 = torch.cdist(
#             #         data['positions_2'][target_atom:target_atom+1],
#             #         data['positions_2'][neighbors_of_target]
#             #     ).min().item()
#             # else:
#             #     dist_127 = float("nan")
                
#             # per_atom_absmax = pred_dx_dt.abs().max(dim=-1).values
#             # topk_vals, topk_atoms = per_atom_absmax.topk(min(10, per_atom_absmax.numel()))

#             # topk_pos = data['positions_2'][topk_atoms]  # [5, 3] 上位5原子の座標
#             # topk_pred = pred_dx_dt[topk_atoms]          # [5, 3] 上位5原子の予測ベクトル自体も見たい場合

#             # print(
#             #     f"[flow:before_update] i={i} t={t.item():.2f} "
#             #     f"pred_finite={torch.isfinite(pred_dx_dt).all().item()}"
#             # )
#             # for rank, (atom_idx, val, pos) in enumerate(zip(topk_atoms.tolist(), topk_vals.tolist(), topk_pos.tolist())):
#             #     print(
#             #         f"  #{rank+1} atom={atom_idx} pred_absmax={val:.3e} "
#             #         f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
#             #     )
            
#             #print(f"  [check] atom127 min_dist_to_neighbor={dist_127:.3f}")
            
            
            
#             #20261006 コメントアウト追加
#             # # ==========================================
#             # # ここに追加: Propagator の場合だけ出力する. 20260729追加
#             # # ==========================================
#             # if tag == "prop":
#             #     # 全原子の予測ベクトルの中で最大の絶対値などを取得
#             #     max_val = pred_dx_dt.max().item()
#             #     min_val = pred_dx_dt.min().item()
#             #     abs_max = pred_dx_dt.abs().max().item()
                
#             #     print(f"[{tag}] step_i={i}, t={t.item():.2f} | pred abs_max={abs_max:.4f} (min={min_val:.4f}, max={max_val:.4f})")
                
#             #     # もし異常に大きい値（例えば2.0以上）が出た場合は、どの原子が暴走しているか詳細を出す
#             #     if abs_max > 20.0:
#             #         per_atom_absmax = pred_dx_dt.abs().max(dim=-1).values
#             #         topk_vals, topk_atoms = per_atom_absmax.topk(min(3, per_atom_absmax.numel()))
                    
#             #         print(f"  --> [WARNING] 異常に大きい予測値！")
#             #         for rank, (atom_idx, val) in enumerate(zip(topk_atoms.tolist(), topk_vals.tolist())):
#             #             element_num = data["elements"][atom_idx].item()
#             #             print(f"      #{rank+1}: atom={atom_idx} (Element: {element_num}) val={val:.4f}")
#             # # ==========================================
            
            
            
           
            
#             if solver == "euler":
#                 data["positions_2"] += pred_dx_dt * step_dt  # Euler step
#             elif solver == "heun":
#                 next_t = (t + step_dt).unsqueeze(0)
#                 next_data = data.clone()
#                 next_data["positions_2"] += pred_dx_dt * step_dt
#                 if prediction_mode == "velocity":
#                     next_dx_dt = model(next_data, t_per_graph=next_t)
#                 elif prediction_mode == "data":
#                     next_disp = model(next_data, t_per_graph=next_t)
#                     if i == flow_steps - 1:
#                         return next_data["positions_1"] + next_disp
#                     next_dx_dt = (
#                         next_data["positions_1"] + next_disp - next_data["positions_2"]
#                     ) / (1 - next_t)
#                 data["positions_2"] += 0.5 * (pred_dx_dt + next_dx_dt) * step_dt
#             else:
#                 raise NotImplementedError

#         return data["positions_2"]

#     @torch.no_grad()
#     def step(
#         self,
#         positions: np.ndarray,
#         flow_steps: int = 10,
#         solver: str = "euler",
#         use_corrector: bool = True,
#     ) -> np.ndarray:
#         # Propagator
#         data = self.initialize_batch(positions, self.propagate_prior)
#         if (
#             hasattr(self.propagate_model, "use_scale")
#             and self.propagate_model.use_scale
#         ):
#             scale = self.propagate_model.scale_model(data)
#             data["prior"] = data["prior"] * scale[:, None]
#         data["positions_2"] = data["positions_2"] + data["prior"]
#         new_positions = self.propagate_flow(
#             data, self.propagate_model, flow_steps, solver, tag="prop"
#         )
#         new_positions = new_positions.detach().cpu().numpy()

#         # Corrector
#         if use_corrector:
#             data = self.initialize_batch(new_positions, self.correct_prior)
#             new_positions = self.propagate_flow(
#                 data, self.correct_model, flow_steps, solver, tag="corr"
#             )
#             new_positions = new_positions.detach().cpu().numpy()
#         return new_positions

#     @torch.no_grad()
#     def run(
#         self,
#         positions: np.ndarray,  # [n_atoms, 3]
#         steps: int = 25,
#         flow_steps: int = 10,
#         solver: str = "euler",
#         verbose: bool = False,
#         fix_com: bool = True,
#         num_retries: int = 2,
#         corrector_every: int = 1,
#     ) -> np.ndarray:
#         traj = [positions]
#         it = tqdm(range(steps)) if verbose else range(steps)
#         for i_step in it:
#             # Save previous center of mass
#             positions_prev = positions.copy()
#             # Propagate (and optionally correct)
#             success = False
#             # for _ in range(num_retries + 1):
#             #     use_corrector = ((i_step + 1) % corrector_every == 0) and (
#             for i_retry in range(num_retries + 1):
#                 use_corrector = ((i_step + 1) % corrector_every == 0) and (
#                     self.correct_model is not None
#                 )
#                 positions = self.step(positions_prev, flow_steps, solver, use_corrector)
#                 # Fix center of mass
#                 if fix_com:
#                     positions_diff = positions - positions_prev
#                     com_diff = np.sum(
#                         positions_diff * self.masses[:, None], axis=0
#                     ) / np.sum(self.masses)
#                     positions -= com_diff
#                 # Check for divergence
#                 if not (
#                     np.isnan(positions).any()
#                     or np.abs(positions - positions_prev).max() > 1e3
#                 ):
#                     success = True
#                     break
#             if not success:
#                 break
#             # Save trajectory
#             traj.append(positions)
#         return traj






# #20261029 追記(コメントアウト): FlowSimulatorのrun()にtrace_csv出力を追加.
# #通常の inference.py に、推論中のprior・各Flowステップの座標・予測ベクトル場・重心補正をCSVへ保存する機能を加えた診断用の FlowSimulator


# import csv
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


# class FlowSimulator:
#     TRACE_FIELDS = [
#         "outer_step", "retry_index", "phase", "flow_index", "flow_time",
#         "atom_index", "atomic_number", "mass_amu",
#         "reference_x", "reference_y", "reference_z",
#         "position_x", "position_y", "position_z",
#         "local_dx", "local_dy", "local_dz",
#         "net_dx", "net_dy", "net_dz",
#         "sampled_prior_dx", "sampled_prior_dy", "sampled_prior_dz",
#         "flow_prior_dx", "flow_prior_dy", "flow_prior_dz",
#         "velocity_x", "velocity_y", "velocity_z",
#         "com_shift_x", "com_shift_y", "com_shift_z", "fix_com_enabled",
#     ]

#     def __init__(self, propagate_model: FlowModule, propagate_prior: Prior,
#                  atomic_numbers: np.ndarray, element_idx: np.ndarray,
#                  lattice: np.ndarray, temp: float,
#                  correct_model: FlowModule | None = None,
#                  correct_prior: Prior | None = None, pbc: bool = True,
#                  scale_Li_index: int | None = None,
#                  scale_frame_index: int | None = None,
#                  fix_frame_species: list[int] | None = None):
#         self.propagate_model = propagate_model
#         self.propagate_prior = propagate_prior
#         self.atomic_numbers = atomic_numbers
#         self.elements = element_idx[atomic_numbers]
#         self.masses = ase.data.atomic_masses[atomic_numbers]
#         self.lattice, self.temp = lattice, temp
#         self.correct_model, self.correct_prior = correct_model, correct_prior
#         self.pbc, self.device = pbc, propagate_model.device
#         self.scale_Li_index, self.scale_frame_index = scale_Li_index, scale_frame_index
#         self.frame_mask = (np.isin(atomic_numbers, fix_frame_species)
#                            if fix_frame_species is not None else None)

#     def initialize_batch(self, positions: np.ndarray, prior: Prior) -> Data:
#         edge_index, shifts = get_neighbor_list_batch(
#             positions_batch=[positions], lattice=self.lattice,
#             cutoff=self.propagate_model.cfg.data.cutoff, periodic=self.pbc)
#         if isinstance(prior, NormalPrior):
#             sample = prior.sample(shape=positions.shape)
#         elif isinstance(prior, MaxwellBoltzmannPrior):
#             sample = prior.sample(temperature=self.temp, masses=self.masses,
#                                  shape=positions.shape)
#         elif isinstance(prior, AdaptiveMaxwellBoltzmannPrior):
#             sample = prior.sample(
#                 temperature=self.temp, atomic_numbers=self.atomic_numbers,
#                 masses=self.masses, scale_Li_index=self.scale_Li_index,
#                 scale_frame_index=self.scale_frame_index, shape=positions.shape)
#         elif isinstance(prior, UniformScaleNormalPrior):
#             sample = prior.sample(shape=positions.shape)
#         else:
#             raise NotImplementedError
#         data = Data(
#             num_nodes=len(self.elements),
#             positions_1=torch.tensor(positions, dtype=torch.float),
#             prior=torch.tensor(sample, dtype=torch.float),
#             positions_2=torch.tensor(positions, dtype=torch.float),
#             edge_index=torch.tensor(edge_index, dtype=torch.long),
#             shifts=torch.tensor(shifts, dtype=torch.float),
#             elements=torch.tensor(self.elements, dtype=torch.long),
#             temp=torch.full(self.elements.shape, self.temp).float())
#         data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
#         return data.to(self.device, non_blocking=True)

#     @staticmethod
#     def _min_pair_distance_torch(pos: torch.Tensor) -> float:
#         d = torch.cdist(pos, pos)
#         n = d.shape[0]
#         d[torch.arange(n), torch.arange(n)] = float("inf")
#         return d.min().item()

#     @staticmethod
#     def _to_numpy(value):
#         if isinstance(value, torch.Tensor):
#             return value.detach().cpu().numpy().copy()
#         return np.array(value, copy=True)

#     @staticmethod
#     def _record(trace, phase, positions, reference, flow_index=None,
#                 flow_time=None, sampled_prior=None, flow_prior=None,
#                 velocity=None, com_shift=None):
#         if trace is None:
#             return
#         convert = FlowSimulator._to_numpy
#         rec = dict(phase=phase, flow_index=flow_index, flow_time=flow_time)
#         for name, value in (
#             ("positions", positions), ("reference", reference),
#             ("sampled_prior", sampled_prior), ("flow_prior", flow_prior),
#             ("velocity", velocity), ("com_shift", com_shift)):
#             rec[name] = None if value is None else convert(value)
#         trace.append(rec)

#     @staticmethod
#     @torch.no_grad()
#     def propagate_flow(data: Data, model: FlowModule, flow_steps: int = 10,
#                        solver: str = "euler", tag: str = "flow", trace=None,
#                        sampled_prior=None) -> torch.Tensor:
#         if flow_steps < 1:
#             raise ValueError("flow_steps must be >= 1")
#         t_list = torch.linspace(0.0, 1.0, flow_steps + 1,
#                                 device=model.device, dtype=torch.float)
#         step_dt = t_list[1] - t_list[0]
#         prediction_mode = (model.cfg.model.prediction_mode
#                            if hasattr(model.cfg.model, "prediction_mode")
#                            else "velocity")
#         reference, flow_prior = data["positions_1"], data["prior"]
#         for i, t in enumerate(t_list[:-1]):
#             if prediction_mode == "velocity":
#                 pred_dx_dt = model(data, t_per_graph=t.unsqueeze(0))
#             elif prediction_mode == "data":
#                 pred_disp = model(data, t_per_graph=t.unsqueeze(0))
#                 pred_dx_dt = (data["positions_1"] + pred_disp
#                               - data["positions_2"]) / (1 - t)
#             else:
#                 raise NotImplementedError
#             FlowSimulator._record(
#                 trace, tag, data["positions_2"], reference,
#                 flow_index=i, flow_time=float(t.item()),
#                 sampled_prior=sampled_prior, flow_prior=flow_prior,
#                 velocity=pred_dx_dt)
#             if solver == "euler":
#                 data["positions_2"] += pred_dx_dt * step_dt
#             elif solver == "heun":
#                 next_t = (t + step_dt).unsqueeze(0)
#                 next_data = data.clone()
#                 next_data["positions_2"] += pred_dx_dt * step_dt
#                 if prediction_mode == "velocity":
#                     next_dx_dt = model(next_data, t_per_graph=next_t)
#                 elif prediction_mode == "data":
#                     next_disp = model(next_data, t_per_graph=next_t)
#                     if i == flow_steps - 1:
#                         result = next_data["positions_1"] + next_disp
#                         FlowSimulator._record(
#                             trace, tag, result, reference,
#                             flow_index=flow_steps, flow_time=1.0,
#                             sampled_prior=sampled_prior, flow_prior=flow_prior)
#                         return result
#                     next_dx_dt = (next_data["positions_1"] + next_disp
#                                   - next_data["positions_2"]) / (1 - next_t)
#                 data["positions_2"] += 0.5 * (pred_dx_dt + next_dx_dt) * step_dt
#             else:
#                 raise NotImplementedError
#         FlowSimulator._record(
#             trace, tag, data["positions_2"], reference,
#             flow_index=flow_steps, flow_time=1.0,
#             sampled_prior=sampled_prior, flow_prior=flow_prior)
#         return data["positions_2"]

#     @torch.no_grad()
#     def step(self, positions: np.ndarray, flow_steps: int = 10,
#              solver: str = "euler", use_corrector: bool = True,
#              trace=None) -> np.ndarray:
#         data = self.initialize_batch(positions, self.propagate_prior)
#         sampled = data["prior"].clone() if trace is not None else None
#         if hasattr(self.propagate_model, "use_scale") and self.propagate_model.use_scale:
#             scale = self.propagate_model.scale_model(data)
#             data["prior"] = data["prior"] * scale[:, None]
#         data["positions_2"] = data["positions_2"] + data["prior"]
#         result = self.propagate_flow(
#             data, self.propagate_model, flow_steps, solver,
#             tag="prop", trace=trace, sampled_prior=sampled)
#         new_positions = result.detach().cpu().numpy()
#         self._record(trace, "prop_end", new_positions, positions)
#         if use_corrector:
#             corr_reference = new_positions.copy() if trace is not None else None
#             data = self.initialize_batch(new_positions, self.correct_prior)
#             sampled = data["prior"].clone() if trace is not None else None
#             # Preserve the supplied implementation: no prior addition here.
#             result = self.propagate_flow(
#                 data, self.correct_model, flow_steps, solver,
#                 tag="corr", trace=trace, sampled_prior=sampled)
#             new_positions = result.detach().cpu().numpy()
#             self._record(trace, "corr_end", new_positions, corr_reference)
#         return new_positions

#     def _initialize_csv(self, trace_csv: Path):
#         trace_csv.parent.mkdir(parents=True, exist_ok=True)
#         with trace_csv.open("w", newline="", encoding="utf-8") as f:
#             csv.DictWriter(f, fieldnames=self.TRACE_FIELDS).writeheader()

#     def _write_trace(self, trace_csv, trace, positions_initial,
#                      outer_step, retry_index, fix_com):
#         with trace_csv.open("a", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=self.TRACE_FIELDS)
#             for rec in trace:
#                 for atom_index in range(len(self.atomic_numbers)):
#                     pos = rec["positions"][atom_index]
#                     ref = rec["reference"][atom_index]
#                     row = dict(
#                         outer_step=outer_step, retry_index=retry_index,
#                         phase=rec["phase"], flow_index=rec["flow_index"],
#                         flow_time=rec["flow_time"], atom_index=atom_index,
#                         atomic_number=int(self.atomic_numbers[atom_index]),
#                         mass_amu=float(self.masses[atom_index]),
#                         fix_com_enabled=bool(fix_com))
#                     for component, axis in enumerate(("x", "y", "z")):
#                         row[f"reference_{axis}"] = float(ref[component])
#                         row[f"position_{axis}"] = float(pos[component])
#                         row[f"local_d{axis}"] = float(pos[component] - ref[component])
#                         row[f"net_d{axis}"] = float(
#                             pos[component] - positions_initial[atom_index, component])
#                         for key, prefix in (
#                             ("sampled_prior", "sampled_prior_d"),
#                             ("flow_prior", "flow_prior_d"),
#                             ("velocity", "velocity_")):
#                             value = rec[key]
#                             row[f"{prefix}{axis}"] = (
#                                 "" if value is None
#                                 else float(value[atom_index, component]))
#                         shift = rec["com_shift"]
#                         row[f"com_shift_{axis}"] = (
#                             "" if shift is None else float(shift[component]))
#                     writer.writerow(row)

#     @torch.no_grad()
#     def run(self, positions: np.ndarray, steps: int = 25,
#             flow_steps: int = 10, solver: str = "euler", verbose: bool = False,
#             fix_com: bool = True, num_retries: int = 2,
#             corrector_every: int = 1,
#             trace_csv: str | Path | None = None) -> np.ndarray:
#         """Export all atoms and accepted steps to one CSV; overwrite on each run."""
#         if corrector_every < 1:
#             raise ValueError("corrector_every must be >= 1")
#         if trace_csv is not None:
#             trace_csv = Path(trace_csv)
#             self._initialize_csv(trace_csv)
#         traj = [positions]
#         it = tqdm(range(steps)) if verbose else range(steps)
#         for i_step in it:
#             positions_prev = positions.copy()
#             success = False
#             for i_retry in range(num_retries + 1):
#                 trace = [] if trace_csv is not None else None
#                 self._record(trace, "outer_start", positions_prev, positions_prev)
#                 use_corrector = (((i_step + 1) % corrector_every == 0)
#                                  and self.correct_model is not None)
#                 positions = self.step(positions_prev, flow_steps, solver,
#                                       use_corrector, trace=trace)
#                 displacement = positions - positions_prev
#                 com_diff = np.sum(displacement * self.masses[:, None], axis=0) / np.sum(self.masses)
#                 applied_shift = com_diff if fix_com else np.zeros(3)
#                 self._record(trace, "before_com", positions, positions_prev,
#                              com_shift=applied_shift)
#                 if fix_com:
#                     positions -= com_diff
#                 self._record(trace, "after_com", positions, positions_prev,
#                              com_shift=applied_shift)
#                 if not (np.isnan(positions).any()
#                         or np.abs(positions - positions_prev).max() > 1e3):
#                     success = True
#                     break
#             if not success:
#                 break
#             if trace_csv is not None:
#                 self._write_trace(trace_csv, trace, positions_prev,
#                                   i_step + 1, i_retry, fix_com)
#             traj.append(positions)
#         return traj
