# from pathlib import Path

# import ase
# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import Data, Dataset

# from liflow.utils.geometry import get_neighbor_list_batch
# from liflow.utils.prior import (
#     AdaptiveMaxwellBoltzmannPrior,
#     MaxwellBoltzmannPrior,
#     NormalPrior,
#     Prior,
#     UniformScaleNormalPrior,
# )
# #20261112 現在座標をラティス内にラップするための関数をインポート
# from ase.geometry import wrap_positions


# class NoisedFrameDataset(Dataset):
#     def __init__(
#         self,
#         data_path: str,
#         df: pd.DataFrame,
#         time_delay_steps: int = 100,
#         noise: UniformScaleNormalPrior = UniformScaleNormalPrior(),
#         prior: Prior = MaxwellBoltzmannPrior(scale=0.1),
#         cutoff: float = 5.0,
#         pbc: bool = True,
#         in_memory: bool = False,
#     ):
#         super().__init__()
#         self.root = Path(data_path)
#         self.df = df
#         self.skip_steps = time_delay_steps
#         self.noise = noise
#         self.prior = prior
#         self.cutoff = cutoff
#         self.pbc = pbc
#         self.in_memory = in_memory

#         self.load_data()

#     def load_data(self):
#         self.element_idx = np.load(self.root / "element_index.npy")
#         self.atomic_numbers = np.load(
#             self.root / "atomic_numbers.npy", allow_pickle=True
#         ).item()
#         self.positions = dict()
#         # Load positions for each temperature
#         temp_set = set(self.df["temp"])
#         for temp in temp_set:
#             self.positions[temp] = np.load(self.root / f"positions_{temp}K.npz")
#             if self.in_memory:  # load all positions into memory
#                 self.positions[temp] = dict(self.positions[temp])
#         self.lattice = np.load(self.root / "lattice.npy", allow_pickle=True).item()

#     def len(self):
#         return len(self.df)

#     def get(self, idx):
#         row = self.df.iloc[idx]

#         # Randomly select a timeframe
#         time = np.random.randint(row["t_start"] + self.skip_steps, row["t_end"])

#         # Get atomic numbers, positions, and lattice
#         atomic_numbers = self.atomic_numbers[row["name"]]
#         elements = self.element_idx[atomic_numbers]
#         masses = ase.data.atomic_masses[atomic_numbers]
#         positions = self.positions[row["temp"]][row["name"]][time]
#         lattice = self.lattice[row["name"]]

#         # Add noise to positions
#         positions_noisy = positions + self.noise.sample(shape=positions.shape)

#         # Get edge information
#         edge_index, shifts = get_neighbor_list_batch(
#             positions_batch=[positions_noisy],
#             lattice=lattice,
#             cutoff=self.cutoff,
#             periodic=self.pbc,
#         )
#         ref_dist = ase.data.covalent_radii[atomic_numbers[edge_index]].sum(axis=0)

#         # Add prior displacement
#         if isinstance(self.prior, NormalPrior):
#             prior = self.prior.sample(shape=positions.shape)
#         elif isinstance(self.prior, MaxwellBoltzmannPrior):
#             prior = self.prior.sample(
#                 temperature=row["temp"], masses=masses, shape=positions.shape
#             )
#         elif isinstance(self.prior, (NormalPrior, UniformScaleNormalPrior)):
#             prior = self.prior.sample(shape=positions.shape)
#         else:
#             raise ValueError("Unsupported prior")

#         return Data(
#             num_nodes=len(elements),
#             positions_1=torch.tensor(positions_noisy, dtype=torch.float),
#             prior=torch.tensor(prior, dtype=torch.float),
#             positions_2=torch.tensor(positions, dtype=torch.float),
#             edge_index=torch.tensor(edge_index, dtype=torch.long),
#             shifts=torch.tensor(shifts, dtype=torch.float),
#             ref_dist=torch.tensor(ref_dist, dtype=torch.float),
#             elements=torch.tensor(elements, dtype=torch.long),
#             temp=torch.full(elements.shape, row["temp"]).float(),
#             lattice=torch.tensor(lattice, dtype=torch.float),
#         )


# class TimeDelayedPairDataset(Dataset):
#     def __init__(
#         self,
#         data_path: str,
#         df: pd.DataFrame,
#         time_delay_steps: int = 100,
#         prior: Prior = AdaptiveMaxwellBoltzmannPrior(),
#         cutoff: float = 5.0,
#         pbc: bool = True,
#         neighbor_list_both_ends: bool = False,
#         in_memory: bool = False,
#     ):
#         super().__init__()
#         self.root = Path(data_path)
#         self.df = df
#         self.time_delay_steps = time_delay_steps
#         self.prior = prior
#         self.cutoff = cutoff
#         self.pbc = pbc
#         self.neighbor_list_both_ends = neighbor_list_both_ends
#         self.in_memory = in_memory

#         self.load_data()

#     def load_data(self):
#         self.element_idx = np.load(self.root / "element_index.npy")
#         self.atomic_numbers = np.load(
#             self.root / "atomic_numbers.npy", allow_pickle=True
#         ).item()
#         self.positions = dict()
#         # Load positions for each temperature
#         temp_set = set(self.df["temp"])
#         for temp in temp_set:
#             self.positions[temp] = np.load(self.root / f"positions_{temp}K.npz")
#             if self.in_memory:  # load all positions into memory
#                 self.positions[temp] = dict(self.positions[temp])
#         self.lattice = np.load(self.root / "lattice.npy", allow_pickle=True).item()

#     def len(self):
#         return len(self.df)

#     def get(self, idx):
#         row = self.df.iloc[idx]

#         # Randomly select a temperature and timeframe
#         start_time = np.random.randint(
#             row["t_start"], row["t_end"] - self.time_delay_steps
#         )
#         end_time = start_time + self.time_delay_steps

#         # Get atomic numbers, positions, and lattice
#         atomic_numbers = self.atomic_numbers[row["name"]]
#         elements = self.element_idx[atomic_numbers]
#         masses = ase.data.atomic_masses[atomic_numbers]
#         positions_1 = self.positions[row["temp"]][row["name"]][start_time]
#         positions_2 = self.positions[row["temp"]][row["name"]][end_time]
#         lattice = self.lattice[row["name"]]
        
        
#         #20261112 現在座標をラティス内にラップするための処理を追加
#         # Original displacement from the unwrapped trajectory.
#         displacement_original = positions_2 - positions_1

#         # Move the starting structure into the cell.
#         positions_1_wrapped = wrap_positions(
#             positions_1,
#             cell=lattice,
#             pbc=self.pbc,
#             eps=0.0,
#         )

#         # Apply the same coordinate shift to the ending structure.
#         coordinate_shift = positions_1 - positions_1_wrapped
#         positions_1 = positions_1_wrapped
#         positions_2 = positions_2 - coordinate_shift

#         # Check that the displacement has not changed.
#         assert np.allclose(
#             positions_2 - positions_1,
#             displacement_original,
#             rtol=1e-6,
#             atol=1e-6,
#         ), "Teacher displacement changed after wrapping."
        
#         # print(
#         #     "pbc =", self.pbc,
#         #     "| max coordinate shift =", np.max(np.abs(coordinate_shift)),
#         #     "| max displacement error =",
#         #     np.max(np.abs(
#         #         (positions_2 - positions_1) - displacement_original
#         #     )),
#         # )
#         #20261112 ここまで追加
        

#         # Get edge information
#         edge_index, shifts = get_neighbor_list_batch(
#             positions_batch=(
#                 [positions_1, positions_2]
#                 if self.neighbor_list_both_ends
#                 else [positions_1]
#             ),
#             lattice=lattice,
#             cutoff=self.cutoff,
#             periodic=self.pbc,
#         )
#         ref_dist = ase.data.covalent_radii[atomic_numbers[edge_index]].sum(axis=0)

#         # Add prior displacement
#         if isinstance(self.prior, NormalPrior):
#             prior = self.prior.sample(shape=positions_1.shape)
#         elif isinstance(self.prior, MaxwellBoltzmannPrior):
#             prior = self.prior.sample(
#                 temperature=row["temp"], masses=masses, shape=positions_1.shape
#             )
#         elif isinstance(self.prior, AdaptiveMaxwellBoltzmannPrior):
#             prior = self.prior.sample(
#                 temperature=row["temp"],
#                 atomic_numbers=atomic_numbers,
#                 masses=masses,
#                 scale_Li_index=int(row["prior_Li"]),
#                 scale_frame_index=int(row["prior_frame"]),
#                 shape=positions_1.shape,
#             )
#         else:
#             raise ValueError("Unsupported prior")

#         return Data(
#             num_nodes=len(elements),
#             positions_1=torch.tensor(positions_1, dtype=torch.float),
#             prior=torch.tensor(prior, dtype=torch.float),
#             positions_2=torch.tensor(positions_2, dtype=torch.float),
#             edge_index=torch.tensor(edge_index, dtype=torch.long),
#             shifts=torch.tensor(shifts, dtype=torch.float),
#             ref_dist=torch.tensor(ref_dist, dtype=torch.float),
#             elements=torch.tensor(elements, dtype=torch.long),
#             temp=torch.full(elements.shape, row["temp"]).float(),
#             lattice=torch.tensor(lattice, dtype=torch.float),
#         )


# from pathlib import Path

# import ase
# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import Data, Dataset

# from liflow.utils.geometry import get_neighbor_list_batch
# from liflow.utils.prior import (
#     AdaptiveMaxwellBoltzmannPrior,
#     MaxwellBoltzmannPrior,
#     NormalPrior,
#     Prior,
#     UniformScaleNormalPrior,
# )


# class NoisedFrameDataset(Dataset):
#     def __init__(
#         self,
#         data_path: str,
#         df: pd.DataFrame,
#         time_delay_steps: int = 100,
#         noise: UniformScaleNormalPrior = UniformScaleNormalPrior(),
#         prior: Prior = MaxwellBoltzmannPrior(scale=0.1),
#         cutoff: float = 5.0,
#         pbc: bool = True,
#         in_memory: bool = False,
#     ):
#         super().__init__()
#         self.root = Path(data_path)
#         self.df = df
#         self.skip_steps = time_delay_steps
#         self.noise = noise
#         self.prior = prior
#         self.cutoff = cutoff
#         self.pbc = pbc
#         self.in_memory = in_memory

#         self.load_data()

#     def load_data(self):
#         self.element_idx = np.load(self.root / "element_index.npy")
#         self.atomic_numbers = np.load(
#             self.root / "atomic_numbers.npy", allow_pickle=True
#         ).item()
#         self.positions = dict()
#         # Load positions for each temperature
#         temp_set = set(self.df["temp"])
#         for temp in temp_set:
#             self.positions[temp] = np.load(self.root / f"positions_{temp}K.npz")
#             if self.in_memory:  # load all positions into memory
#                 self.positions[temp] = dict(self.positions[temp])
#         self.lattice = np.load(self.root / "lattice.npy", allow_pickle=True).item()

#     def len(self):
#         return len(self.df)

#     def get(self, idx):
#         row = self.df.iloc[idx]

#         # Randomly select a timeframe
#         time = np.random.randint(row["t_start"] + self.skip_steps, row["t_end"])

#         # Get atomic numbers, positions, and lattice
#         atomic_numbers = self.atomic_numbers[row["name"]]
#         elements = self.element_idx[atomic_numbers]
#         masses = ase.data.atomic_masses[atomic_numbers]
#         positions = self.positions[row["temp"]][row["name"]][time]
#         lattice = self.lattice[row["name"]]

#         # Add noise to positions
#         positions_noisy = positions + self.noise.sample(shape=positions.shape)

#         # Get edge information
#         edge_index, shifts = get_neighbor_list_batch(
#             positions_batch=[positions_noisy],
#             lattice=lattice,
#             cutoff=self.cutoff,
#             periodic=self.pbc,
#         )
#         ref_dist = ase.data.covalent_radii[atomic_numbers[edge_index]].sum(axis=0)

#         # Add prior displacement
#         if isinstance(self.prior, NormalPrior):
#             prior = self.prior.sample(shape=positions.shape)
#         elif isinstance(self.prior, MaxwellBoltzmannPrior):
#             prior = self.prior.sample(
#                 temperature=row["temp"], masses=masses, shape=positions.shape
#             )
#         elif isinstance(self.prior, (NormalPrior, UniformScaleNormalPrior)):
#             prior = self.prior.sample(shape=positions.shape)
#         else:
#             raise ValueError("Unsupported prior")

#         return Data(
#             num_nodes=len(elements),
#             positions_1=torch.tensor(positions_noisy, dtype=torch.float),
#             prior=torch.tensor(prior, dtype=torch.float),
#             positions_2=torch.tensor(positions, dtype=torch.float),
#             edge_index=torch.tensor(edge_index, dtype=torch.long),
#             shifts=torch.tensor(shifts, dtype=torch.float),
#             ref_dist=torch.tensor(ref_dist, dtype=torch.float),
#             elements=torch.tensor(elements, dtype=torch.long),
#             temp=torch.full(elements.shape, row["temp"]).float(),
#             lattice=torch.tensor(lattice, dtype=torch.float),
#         )


# class TimeDelayedPairDataset(Dataset):
#     def __init__(
#         self,
#         data_path: str,
#         df: pd.DataFrame,
#         time_delay_steps: int = 100,
#         prior: Prior = AdaptiveMaxwellBoltzmannPrior(),
#         cutoff: float = 5.0,
#         pbc: bool = True,
#         neighbor_list_both_ends: bool = False,
#         in_memory: bool = False,
#     ):
#         super().__init__()
#         self.root = Path(data_path)
#         self.df = df
#         self.time_delay_steps = time_delay_steps
#         self.prior = prior
#         self.cutoff = cutoff
#         self.pbc = pbc
#         self.neighbor_list_both_ends = neighbor_list_both_ends
#         self.in_memory = in_memory

#         self.load_data()

#     def load_data(self):
#         self.element_idx = np.load(self.root / "element_index.npy")
#         self.atomic_numbers = np.load(
#             self.root / "atomic_numbers.npy", allow_pickle=True
#         ).item()
#         self.positions = dict()
#         # Load positions for each temperature
#         temp_set = set(self.df["temp"])
#         for temp in temp_set:
#             self.positions[temp] = np.load(self.root / f"positions_{temp}K.npz")
#             if self.in_memory:  # load all positions into memory
#                 self.positions[temp] = dict(self.positions[temp])
#         self.lattice = np.load(self.root / "lattice.npy", allow_pickle=True).item()

#     def len(self):
#         return len(self.df)

#     def get(self, idx):
#         row = self.df.iloc[idx]

#         # Randomly select a temperature and timeframe
#         start_time = np.random.randint(
#             row["t_start"], row["t_end"] - self.time_delay_steps
#         )
#         end_time = start_time + self.time_delay_steps

#         # Get atomic numbers, positions, and lattice
#         atomic_numbers = self.atomic_numbers[row["name"]]
#         elements = self.element_idx[atomic_numbers]
#         masses = ase.data.atomic_masses[atomic_numbers]
#         positions_1 = self.positions[row["temp"]][row["name"]][start_time]
#         positions_2 = self.positions[row["temp"]][row["name"]][end_time]
#         lattice = self.lattice[row["name"]]

#         # Get edge information
#         edge_index, shifts = get_neighbor_list_batch(
#             positions_batch=(
#                 [positions_1, positions_2]
#                 if self.neighbor_list_both_ends
#                 else [positions_1]
#             ),
#             lattice=lattice,
#             cutoff=self.cutoff,
#             periodic=self.pbc,
#         )
#         ref_dist = ase.data.covalent_radii[atomic_numbers[edge_index]].sum(axis=0)

#         # Add prior displacement
#         if isinstance(self.prior, NormalPrior):
#             prior = self.prior.sample(shape=positions_1.shape)
#         elif isinstance(self.prior, MaxwellBoltzmannPrior):
#             prior = self.prior.sample(
#                 temperature=row["temp"], masses=masses, shape=positions_1.shape
#             )
#         elif isinstance(self.prior, AdaptiveMaxwellBoltzmannPrior):
#             prior = self.prior.sample(
#                 temperature=row["temp"],
#                 atomic_numbers=atomic_numbers,
#                 masses=masses,
#                 scale_Li_index=int(row["prior_Li"]),
#                 scale_frame_index=int(row["prior_frame"]),
#                 shape=positions_1.shape,
#             )
#         else:
#             raise ValueError("Unsupported prior")

#         return Data(
#             num_nodes=len(elements),
#             positions_1=torch.tensor(positions_1, dtype=torch.float),
#             prior=torch.tensor(prior, dtype=torch.float),
#             positions_2=torch.tensor(positions_2, dtype=torch.float),
#             edge_index=torch.tensor(edge_index, dtype=torch.long),
#             shifts=torch.tensor(shifts, dtype=torch.float),
#             ref_dist=torch.tensor(ref_dist, dtype=torch.float),
#             elements=torch.tensor(elements, dtype=torch.long),
#             temp=torch.full(elements.shape, row["temp"]).float(),
#             lattice=torch.tensor(lattice, dtype=torch.float),
#         )




from pathlib import Path

import ase
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from liflow.utils.geometry import get_neighbor_list_batch
from liflow.utils.prior import (
    AdaptiveMaxwellBoltzmannPrior,
    MaxwellBoltzmannPrior,
    NormalPrior,
    Prior,
    UniformScaleNormalPrior,
)


class NoisedFrameDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        df: pd.DataFrame,
        time_delay_steps: int = 100,
        noise: UniformScaleNormalPrior = UniformScaleNormalPrior(),
        prior: Prior = MaxwellBoltzmannPrior(scale=0.1),
        cutoff: float = 5.0,
        pbc: bool = True,
        in_memory: bool = False,
    ):
        super().__init__()
        self.root = Path(data_path)
        self.df = df
        self.skip_steps = time_delay_steps
        self.noise = noise
        self.prior = prior
        self.cutoff = cutoff
        self.pbc = pbc
        self.in_memory = in_memory

        self.load_data()

    def load_data(self):
        self.element_idx = np.load(
            self.root / "element_index.npy"
        )
        self.atomic_numbers = np.load(
            self.root / "atomic_numbers.npy",
            allow_pickle=True,
        ).item()

        self.positions = dict()

        for temp in set(self.df["temp"]):
            self.positions[temp] = np.load(
                self.root / f"positions_{temp}K.npz"
            )

            if self.in_memory:
                self.positions[temp] = dict(self.positions[temp])

        self.lattice = np.load(
            self.root / "lattice.npy",
            allow_pickle=True,
        ).item()

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]

        time = np.random.randint(
            row["t_start"] + self.skip_steps,
            row["t_end"],
        )

        atomic_numbers = self.atomic_numbers[row["name"]]
        elements = self.element_idx[atomic_numbers]
        masses = ase.data.atomic_masses[atomic_numbers]
        positions = self.positions[row["temp"]][row["name"]][time]
        lattice = self.lattice[row["name"]]

        positions_noisy = positions + self.noise.sample(
            shape=positions.shape
        )

        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=[positions_noisy],
            lattice=lattice,
            cutoff=self.cutoff,
            periodic=self.pbc,
        )

        ref_dist = ase.data.covalent_radii[
            atomic_numbers[edge_index]
        ].sum(axis=0)

        if isinstance(self.prior, NormalPrior):
            prior = self.prior.sample(
                shape=positions.shape
            )

        elif isinstance(self.prior, MaxwellBoltzmannPrior):
            prior = self.prior.sample(
                temperature=row["temp"],
                masses=masses,
                shape=positions.shape,
            )

        elif isinstance(
            self.prior,
            (NormalPrior, UniformScaleNormalPrior),
        ):
            prior = self.prior.sample(
                shape=positions.shape
            )

        else:
            raise ValueError("Unsupported prior")

        return Data(
            num_nodes=len(elements),
            positions_1=torch.tensor(
                positions_noisy,
                dtype=torch.float,
            ),
            prior=torch.tensor(
                prior,
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
            ref_dist=torch.tensor(
                ref_dist,
                dtype=torch.float,
            ),
            elements=torch.tensor(
                elements,
                dtype=torch.long,
            ),
            temp=torch.full(
                elements.shape,
                row["temp"],
            ).float(),
            lattice=torch.tensor(
                lattice,
                dtype=torch.float,
            ),
        )


class TimeDelayedPairDataset(Dataset):
    """
    CSVの各行を、固定した開始・終了frameのペアとして扱う。

    必須列:
        name, temp, t_start, t_end,
        prior_Li, prior_frame, is_large

    固定ペアの条件:
        t_end - t_start == time_delay_steps + 1

    実際に使用するframe:
        start_time = t_start
        end_time = t_start + time_delay_steps

    sample_weights:
        初期値は全て1。
        train用の逆確率補正係数はDataModuleから設定する。
    """

    def __init__(
        self,
        data_path: str,
        df: pd.DataFrame,
        time_delay_steps: int = 100,
        prior: Prior = AdaptiveMaxwellBoltzmannPrior(),
        cutoff: float = 5.0,
        pbc: bool = True,
        neighbor_list_both_ends: bool = False,
        in_memory: bool = False,
    ):
        super().__init__()

        self.root = Path(data_path)
        self.df = df.copy().reset_index(drop=True)

        if (
            isinstance(time_delay_steps, bool)
            or not isinstance(time_delay_steps, (int, np.integer))
            or time_delay_steps < 1
        ):
            raise ValueError(
                "time_delay_stepsは1以上の整数にしてください。"
            )

        self.time_delay_steps = int(time_delay_steps)
        self.prior = prior
        self.cutoff = cutoff
        self.pbc = pbc
        self.neighbor_list_both_ends = neighbor_list_both_ends
        self.in_memory = in_memory

        self._validate_pair_table()

        # DataModuleがtrain用の係数を設定する。
        # validationでは全て1のまま使用する。
        self.sample_weights = np.ones(
            len(self.df),
            dtype=np.float64,
        )

        self.load_data()

    def _validate_pair_table(self):
        required_columns = [
            "name",
            "temp",
            "t_start",
            "t_end",
            "prior_Li",
            "prior_frame",
            "is_large",
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"固定ペアCSVに必要な列がありません: {missing}"
            )

        if len(self.df) == 0:
            raise ValueError("固定ペアCSVが空です。")

        if self.df[required_columns].isna().any().any():
            raise ValueError(
                "固定ペアCSVの必須列に欠損値があります。"
            )

        integer_columns = [
            "t_start",
            "t_end",
            "prior_Li",
            "prior_frame",
            "is_large",
        ]

        for column in integer_columns:
            values = pd.to_numeric(
                self.df[column],
                errors="raise",
            ).to_numpy(dtype=np.float64)

            if (
                not np.isfinite(values).all()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError(
                    f"{column}列は有限の整数にしてください。"
                )

            self.df[column] = values.astype(np.int64)

        if (self.df["t_start"] < 0).any():
            raise ValueError(
                "t_startに負のframe番号があります。"
            )

        expected_width = self.time_delay_steps + 1
        widths = self.df["t_end"] - self.df["t_start"]
        invalid = widths != expected_width

        if invalid.any():
            examples = self.df.loc[
                invalid,
                ["name", "temp", "t_start", "t_end"],
            ].head()

            raise ValueError(
                "固定ペアCSVの区間幅が不正です。"
                f" t_end - t_startは{expected_width}が必要です。\n"
                f"{examples.to_string(index=False)}"
            )

        if not self.df["is_large"].isin([0, 1]).all():
            raise ValueError(
                "is_large列には0または1を指定してください。"
            )

    def load_data(self):
        # sample_weightsはここで初期化しない。
        # worker側の再読込でも、設定済みの係数を維持する。
        self.element_idx = np.load(
            self.root / "element_index.npy"
        )

        self.atomic_numbers = np.load(
            self.root / "atomic_numbers.npy",
            allow_pickle=True,
        ).item()

        self.positions = dict()

        for temp in set(self.df["temp"]):
            self.positions[temp] = np.load(
                self.root / f"positions_{temp}K.npz"
            )

            if self.in_memory:
                self.positions[temp] = dict(self.positions[temp])

        self.lattice = np.load(
            self.root / "lattice.npy",
            allow_pickle=True,
        ).item()

    def len(self):
        return len(self.df)

    def get(self, idx):
        idx = int(idx)
        row = self.df.iloc[idx]

        # CSVの各行が指す固定ペアを使用する。
        start_time = int(row["t_start"])
        end_time = start_time + self.time_delay_steps

        atomic_numbers = self.atomic_numbers[row["name"]]
        elements = self.element_idx[atomic_numbers]
        masses = ase.data.atomic_masses[atomic_numbers]

        trajectory = self.positions[row["temp"]][row["name"]]

        if end_time >= len(trajectory):
            raise IndexError(
                "終了frameが座標配列の範囲外です。"
                f" idx={idx}, name={row['name']},"
                f" start={start_time}, end={end_time},"
                f" n_frames={len(trajectory)}"
            )

        # 保存済みのunwrapped座標をそのまま使用する。
        positions_1 = trajectory[start_time]
        positions_2 = trajectory[end_time]
        lattice = self.lattice[row["name"]]

        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=(
                [positions_1, positions_2]
                if self.neighbor_list_both_ends
                else [positions_1]
            ),
            lattice=lattice,
            cutoff=self.cutoff,
            periodic=self.pbc,
        )

        ref_dist = ase.data.covalent_radii[
            atomic_numbers[edge_index]
        ].sum(axis=0)

        # priorは固定せず、アクセスごとに従来どおり生成する。
        if isinstance(self.prior, NormalPrior):
            prior = self.prior.sample(
                shape=positions_1.shape
            )

        elif isinstance(self.prior, MaxwellBoltzmannPrior):
            prior = self.prior.sample(
                temperature=row["temp"],
                masses=masses,
                shape=positions_1.shape,
            )

        elif isinstance(
            self.prior,
            AdaptiveMaxwellBoltzmannPrior,
        ):
            prior = self.prior.sample(
                temperature=row["temp"],
                atomic_numbers=atomic_numbers,
                masses=masses,
                scale_Li_index=int(row["prior_Li"]),
                scale_frame_index=int(row["prior_frame"]),
                shape=positions_1.shape,
            )

        else:
            raise ValueError("Unsupported prior")

        sample_weight = float(self.sample_weights[idx])

        if (
            not np.isfinite(sample_weight)
            or sample_weight <= 0.0
        ):
            raise ValueError(
                "sample_weightは有限の正数が必要です。"
                f" idx={idx}, value={sample_weight}"
            )

        return Data(
            num_nodes=len(elements),

            # 構造ごとに1要素。batch化後は[構造数]になる。
            sample_weight=torch.tensor(
                [sample_weight],
                dtype=torch.float,
            ),
            is_large=torch.tensor(
                [int(row["is_large"])],
                dtype=torch.long,
            ),

            positions_1=torch.tensor(
                positions_1,
                dtype=torch.float,
            ),
            prior=torch.tensor(
                prior,
                dtype=torch.float,
            ),
            positions_2=torch.tensor(
                positions_2,
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
            ref_dist=torch.tensor(
                ref_dist,
                dtype=torch.float,
            ),
            elements=torch.tensor(
                elements,
                dtype=torch.long,
            ),
            temp=torch.full(
                elements.shape,
                row["temp"],
            ).float(),
            lattice=torch.tensor(
                lattice,
                dtype=torch.float,
            ),
        )