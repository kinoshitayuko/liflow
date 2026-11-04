# -*- coding: utf-8 -*-
"""Element-wise, component-wise signed-log transforms for LiFlow (plan A).

No interpolation, integration, velocity conversion, or statistics estimation.
Use the same instance/coefficients for displacements, priors and inverse maps.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn


class SignedLogTransform(nn.Module):
    """Fixed signed-log transform with persistent coefficient buffers.

    values: floating tensor [N, 3], physical displacements in angstrom
    elements: integer tensor [N], INTERNAL model element IDs (not atomic numbers)

    Buffers are included in state_dict when attached to FlowModule as a child.
    The JSON file is needed for construction, not for each forward call.
    """

    def __init__(self, parameters: dict):
        super().__init__()
        if parameters.get("schema_version") != 1:
            raise ValueError("Unsupported schema_version")
        if parameters.get("ready_for_model_use") is not True:
            raise ValueError("JSON is not ready_for_model_use")
        if parameters.get("component_order") != ["x", "y", "z"]:
            raise ValueError("component_order must be ['x', 'y', 'z']")
        if parameters.get("mean_centering") is not False:
            raise ValueError("Only mean_centering=false is supported")
        if parameters.get("formula") != "sign(d)*log1p(lambda*K*abs(d)/sigma)":
            raise ValueError("Unexpected transform formula")

        strength = self._positive(parameters.get("lambda"), "lambda")
        entries = parameters.get("elements")
        if not isinstance(entries, list) or not entries:
            raise ValueError("elements must be a nonempty list")
        ids = [entry.get("element_id") for entry in entries]
        if any(type(e) is not int or e < 0 for e in ids):
            raise ValueError("element_id must be a nonnegative integer")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate element IDs")
        if max(ids) > 10000:
            raise ValueError("Unexpectedly large internal element ID")

        size = max(ids) + 1
        a = torch.zeros((size, 3), dtype=torch.float64)
        sigma = torch.zeros_like(a)
        kval = torch.zeros_like(a)
        valid = torch.zeros(size, dtype=torch.bool)
        symbols = {}
        for entry in entries:
            e = entry["element_id"]
            s = self._triple(entry.get("sigma_A"), f"ID {e}: sigma_A")
            k = self._triple(entry.get("K"), f"ID {e}: K")
            saved = self._triple(entry.get("a_per_A"), f"ID {e}: a_per_A")
            calculated = [strength * ki / si for si, ki in zip(s, k)]
            if any(not math.isfinite(x) or x <= 0 for x in calculated):
                raise ValueError(f"ID {e}: invalid lambda*K/sigma")
            if not all(math.isclose(x, y, rel_tol=1e-10, abs_tol=0.0)
                       for x, y in zip(saved, calculated)):
                raise ValueError(f"ID {e}: a_per_A does not match lambda*K/sigma")
            a[e] = torch.tensor(saved, dtype=torch.float64)
            sigma[e] = torch.tensor(s, dtype=torch.float64)
            kval[e] = torch.tensor(k, dtype=torch.float64)
            valid[e] = True
            symbols[e] = entry.get("species")

        self.register_buffer("a_per_A", a)
        self.register_buffer("sigma_A", sigma)
        self.register_buffer("kurtosis", kval)
        self.register_buffer("valid_ids", valid)
        self.register_buffer("log_strength", torch.tensor(strength, dtype=torch.float64))
        # Preserve construction metadata for checkpoint integration.
        self._parameters_metadata = json.loads(json.dumps(parameters, allow_nan=False))
        self.species_by_id = symbols

    @staticmethod
    def _positive(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a positive finite number")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return value

    @classmethod
    def _triple(cls, values, name):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"{name} must have three values in xyz order")
        return [cls._positive(v, name) for v in values]

    @classmethod
    def from_json(cls, path):
        with Path(path).open(encoding="utf-8-sig") as f:
            return cls(json.load(f))

    def export_parameters(self):
        """JSON-compatible construction metadata for saving with model config."""
        return json.loads(json.dumps(self._parameters_metadata, allow_nan=False))

    def _coefficients(self, values, elements):
        if not isinstance(values, torch.Tensor) or not isinstance(elements, torch.Tensor):
            raise TypeError("values and elements must be torch tensors")
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"values must have shape [N, 3], got {tuple(values.shape)}")
        if elements.ndim != 1 or elements.shape[0] != values.shape[0]:
            raise ValueError("elements must have shape [N]")
        if values.dtype not in (torch.float32, torch.float64):
            raise TypeError("Use float32 or float64; cast mixed-precision inputs explicitly")
        if elements.dtype not in (torch.int32, torch.int64):
            raise TypeError("elements must be int32 or int64")
        if not (values.device == elements.device == self.a_per_A.device):
            raise ValueError("Move transform, values and elements to the same device")
        if not torch.isfinite(values).all():
            raise ValueError("Input contains NaN or Inf")
        ids = elements.long()
        if ((ids < 0) | (ids >= self.valid_ids.numel())).any():
            raise ValueError("Element ID outside coefficient-table range")
        if not self.valid_ids[ids].all():
            raise ValueError("Unregistered element ID")
        a = self.a_per_A[ids].to(dtype=values.dtype)
        if not torch.isfinite(a).all() or (a <= 0).any():
            raise ValueError("Invalid coefficients in computation dtype")
        return a

    def forward(self, displacement, elements):
        return self.transform(displacement, elements)

    def transform(self, displacement, elements):
        """Physical displacement [angstrom] -> dimensionless displacement.

        Separate positive/negative branches preserve derivative a at zero.
        Do not decorate with no_grad: input gradients may be needed later.
        """
        a = self._coefficients(displacement, elements)
        positive = torch.log1p(a * displacement.clamp_min(0))
        negative = -torch.log1p(a * (-displacement).clamp_min(0))
        result = torch.where(displacement >= 0, positive, negative)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite signed-log output")
        return result

    def inverse(self, transformed_displacement, elements):
        """Dimensionless displacement -> physical displacement [angstrom].

        Raise on overflow rather than silently clipping the generated state.
        """
        z = transformed_displacement
        a = self._coefficients(z, elements)
        positive = torch.expm1(z.clamp_min(0)) / a
        negative = -torch.expm1((-z).clamp_min(0)) / a
        result = torch.where(z >= 0, positive, negative)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Inverse overflow: generated transformed displacement is too large")
        return result
    
    
    