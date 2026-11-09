import torch
import torch.nn as nn
from torch_geometric.data import Data

from liflow.model.layers import (
    BesselBasis,
    CosineCutoff,
    DualMessageBlock,
    GatedEquivariantBlock,
    GaussianFourierBasis,
    UpdateBlock,
)
from liflow.utils.geometry import get_unit_vectors_and_lengths


class DualPaiNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_radial_basis: int,
        num_layers: int,
        num_elements: int,
        r_max: float,
        r_offset: float = 0.0,
        ref_temp: float = 1000.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_radial_basis = num_radial_basis
        self.num_layers = num_layers
        self.r_max = r_max
        self.r_offset = r_offset
        self.ref_temp = ref_temp  # reference temperature for scaling

        assert num_features % 2 == 0, "Number of features must be even"
        self.atom_embedding = nn.Embedding(num_elements, num_features)
        self.time_embedding = GaussianFourierBasis(num_basis=num_features // 2)
        self.temp_embedding = GaussianFourierBasis(num_basis=num_features // 2)
        self.radial_embedding = BesselBasis(num_basis=num_radial_basis, r_max=r_max)
        self.cutoff_fn = CosineCutoff(r_max=r_max)
        self.linear_v = nn.Linear(1, num_features, bias=False)

        messages, updates = [], []
        for _ in range(num_layers):
            messages.append(DualMessageBlock(num_features, num_radial_basis))
            updates.append(UpdateBlock(num_features))
        self.messages = nn.ModuleList(messages)
        self.updates = nn.ModuleList(updates)
        self.output_block = GatedEquivariantBlock(
            num_scalar_inputs=num_features,
            num_vector_inputs=num_features,
        )
        
        
        
    
    def forward(self, data: Data):
        # Diagnostic threshold only.
        # It does not clip values, stop execution, or change model outputs.
        diagnostic_threshold = 1.0e6
        diagnostic_rows = []

        def watch(name, tensor):
            """Record detached scalar summaries without keeping GPU tensors."""
            with torch.no_grad():
                value = tensor.detach()
                finite_mask = torch.isfinite(value)
                nonfinite = int((~finite_mask).sum().item())

                row = {
                    "name": name,
                    "shape": tuple(value.shape),
                    "max_abs": None,
                    "min": None,
                    "max": None,
                    "max_abs_index": None,
                    "nonfinite": nonfinite,
                }

                if value.numel() > 0 and finite_mask.any().item():
                    finite_values = value[finite_mask]

                    row["min"] = float(finite_values.min().item())
                    row["max"] = float(finite_values.max().item())
                    row["max_abs"] = float(
                        finite_values.abs().max().item()
                    )

                    absolute = torch.where(
                        finite_mask,
                        value.abs(),
                        torch.full_like(value, -1.0),
                    )

                    flat_index = int(
                        absolute.reshape(-1).argmax().item()
                    )

                    index = []
                    remainder = flat_index
                    for size in reversed(value.shape):
                        index.append(remainder % size)
                        remainder //= size

                    row["max_abs_index"] = tuple(reversed(index))

                diagnostic_rows.append(row)

        # ------------------------------------------------------------
        # Geometry before radial-distance modification
        # ------------------------------------------------------------
        watch("positions_diff", data["positions_2"] - data["positions_1"])

        unit_vectors_1, lengths_1 = get_unit_vectors_and_lengths(
            data["positions_1"],
            data["edge_index"],
            data["shifts"],
        )
        unit_vectors_2, lengths_2 = get_unit_vectors_and_lengths(
            data["positions_2"],
            data["edge_index"],
            data["shifts"],
        )

        watch("lengths_1_raw", lengths_1)
        watch("lengths_2_raw", lengths_2)
        watch("unit_vectors_1", unit_vectors_1)
        watch("unit_vectors_2", unit_vectors_2)

        # ------------------------------------------------------------
        # Original radial basis and cutoff calculations
        # ------------------------------------------------------------
        lengths_1 = (lengths_1 + self.r_offset).clamp(max=self.r_max)
        lengths_2 = (lengths_2 + self.r_offset).clamp(max=self.r_max)

        watch("lengths_1_rbf_input", lengths_1)
        watch("lengths_2_rbf_input", lengths_2)

        radial_embeddings_1 = self.radial_embedding(lengths_1)
        radial_embeddings_2 = self.radial_embedding(lengths_2)
        f_cut_1 = self.cutoff_fn(lengths_1)
        f_cut_2 = self.cutoff_fn(lengths_2)

        watch("radial_embeddings_1", radial_embeddings_1)
        watch("radial_embeddings_2", radial_embeddings_2)
        watch("f_cut_1", f_cut_1)
        watch("f_cut_2", f_cut_2)

        # ------------------------------------------------------------
        # Original initial scalar and vector features
        # ------------------------------------------------------------
        s_atom = self.atom_embedding(data["elements"])
        s_time = self.time_embedding(data["time"])
        s_temp = self.temp_embedding(data["temp"] / self.ref_temp)

        s = s_atom + torch.cat([s_time, s_temp], dim=-1)
        s = s[:, None, :]

        positions_diff = data["positions_2"] - data["positions_1"]
        v = self.linear_v(positions_diff[..., None])

        watch("initial_s", s)
        watch("initial_v", v)

        # ------------------------------------------------------------
        # Original message/update calculations
        # ------------------------------------------------------------
        for layer_index, (message, update) in enumerate(
            zip(self.messages, self.updates)
        ):
            s, v = message(
                s,
                v,
                radial_embeddings_1,
                radial_embeddings_2,
                f_cut_1,
                f_cut_2,
                unit_vectors_1,
                unit_vectors_2,
                data["edge_index"],
            )

            watch(f"layer_{layer_index}.message_s", s)
            watch(f"layer_{layer_index}.message_v", v)

            s, v = update(s, v)

            watch(f"layer_{layer_index}.update_s", s)
            watch(f"layer_{layer_index}.update_v", v)

        # ------------------------------------------------------------
        # Original output calculation
        # ------------------------------------------------------------
        v_out = self.output_block(s, v).squeeze(-1)
        watch("output_velocity", v_out)

        # Keep only Python scalar summaries from the latest forward.
        self.last_forward_diagnostics = diagnostic_rows

        # Print the complete sequence when any monitored tensor is large
        # or nonfinite. The threshold is for logging only.
        abnormal = any(
            row["nonfinite"] > 0
            or (
                row["max_abs"] is not None
                and row["max_abs"] >= diagnostic_threshold
            )
            for row in diagnostic_rows
        )

        if abnormal:
            time_values = data["time"].detach().reshape(-1)

            if time_values.numel() > 0:
                t_min = float(time_values.min().item())
                t_max = float(time_values.max().item())
            else:
                t_min = None
                t_max = None

            print(
                "\n[DualPaiNN diagnostics] "
                f"time_min={t_min}, time_max={t_max}, "
                f"r_offset={self.r_offset}, r_max={self.r_max}",
                flush=True,
            )

            for row in diagnostic_rows:
                print(
                    f"  {row['name']} | "
                    f"shape={row['shape']} | "
                    f"max_abs={row['max_abs']} | "
                    f"min={row['min']} | "
                    f"max={row['max']} | "
                    f"index={row['max_abs_index']} | "
                    f"nonfinite={row['nonfinite']}",
                    flush=True,
                )

            print("[DualPaiNN diagnostics end]\n", flush=True)

        return v_out
    
    
    

    # def forward(self, data: Data):
    #     unit_vectors_1, lengths_1 = get_unit_vectors_and_lengths(
    #         data["positions_1"], data["edge_index"], data["shifts"]
    #     )
    #     unit_vectors_2, lengths_2 = get_unit_vectors_and_lengths(
    #         data["positions_2"], data["edge_index"], data["shifts"]
    #     )
        
        
    #     # ↓↓↓ ここに挿入 ↓↓↓ 2026/07/29 追加
    #     # min_len = lengths_2.min().item()
    #     # if min_len < 10.5:  # 閾値は任意（Li-Li間で通常あり得ない近さ）
    #     #     idx = torch.argmin(lengths_2)
    #     #     atom_a, atom_b = data["edge_index"][:, idx].tolist()
    #     #     elem_a = data["elements"][atom_a].item()
    #     #     elem_b = data["elements"][atom_b].item()
    #     #     if atom_a == 91 or atom_b == 91:  # H2原子のインデックスが91の場合
    #     #         print(f"[WARNING] H2 bond too short: dist={min_len:.4f}Å atoms=({atom_a}:elem{elem_a}, {atom_b}:elem{elem_b})")
    #     #     #print(f"[WARNING] dist={min_len:.4f}Å atoms=({atom_a}:elem{elem_a}, {atom_b}:elem{elem_b})")
            
    #     # ↑↑↑ ここまで ↑↑↑
        
        
    #     # # ↓↓↓ ここに挿入：atom91専用の監視 ↓↓↓
    #     # target_atom = 95
    #     # mask = (data["edge_index"][0] == target_atom) | (data["edge_index"][1] == target_atom)
    #     # if mask.any():
    #     #     target_lengths = lengths_2[mask]
    #     #     target_edges = data["edge_index"][:, mask]
    #     #     min_idx_local = torch.argmin(target_lengths)
    #     #     dist_91 = target_lengths[min_idx_local].item()
    #     #     a, b = target_edges[:, min_idx_local].tolist()
    #     #     print(f"[WATCH atom{target_atom}] dist={dist_91:.4f}Å pair=({a}:elem{data['elements'][a].item()}, {b}:elem{data['elements'][b].item()})")
    #     # # ↑↑↑ ここまで ↑↑↑
        
        
        

    #     # Compute radial basis functions
    #     lengths_1 = (lengths_1 + self.r_offset).clamp(max=self.r_max)
    #     lengths_2 = (lengths_2 + self.r_offset).clamp(max=self.r_max)
    #     radial_embeddings_1 = self.radial_embedding(lengths_1)
    #     radial_embeddings_2 = self.radial_embedding(lengths_2)
    #     f_cut_1 = self.cutoff_fn(lengths_1)
    #     f_cut_2 = self.cutoff_fn(lengths_2)

    #     # Compute initial scalar and vector features
    #     s_atom = self.atom_embedding(data["elements"])
    #     s_time = self.time_embedding(data["time"])
    #     s_temp = self.temp_embedding(data["temp"] / self.ref_temp)
    #     s = s_atom + torch.cat([s_time, s_temp], dim=-1)
    #     s = s[:, None, :]  # [n_nodes, 1, n_feats]
    #     # v = torch.zeros_like(s).repeat(1, 3, 1)  # [n_nodes, 3, n_feats]
    #     positions_diff = data["positions_2"] - data["positions_1"]
    #     v = self.linear_v(positions_diff[..., None])  # [n_nodes, 3, n_feats]

    #     for message, update in zip(self.messages, self.updates):
    #         s, v = message(
    #             s,
    #             v,
    #             radial_embeddings_1,
    #             radial_embeddings_2,
    #             f_cut_1,
    #             f_cut_2,
    #             unit_vectors_1,
    #             unit_vectors_2,
    #             data["edge_index"],
    #         )
    #         s, v = update(s, v)
    #     v_out = self.output_block(s, v).squeeze(-1)
    #     return v_out