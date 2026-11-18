import random
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    RandomSampler,
    Sampler,
    WeightedRandomSampler,
)
from torch_geometric.loader import DataLoader

from liflow.data.dataset import (
    NoisedFrameDataset,
    TimeDelayedPairDataset,
)
from liflow.utils.prior import (
    AdaptiveMaxwellBoltzmannPrior,
    MaxwellBoltzmannPrior,
    Prior,
    UniformScaleNormalPrior,
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() & 0xFFFFFFFF
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset.load_data()


def _temporal_train_valid_split(
    df: pd.DataFrame,
    valid_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    現在の分割規則を維持する。
    valid_fraction=0.2なら各グループの0, 5, 10, ...行をvalidにする。
    """
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError(
            "valid_fractionは0より大きく1未満にしてください。"
        )

    df = df.sort_values(
        ["name", "temp", "t_start"]
    ).reset_index(drop=True)

    step = max(1, round(1.0 / valid_fraction))
    valid_indices = []

    for _, group in df.groupby(["name", "temp"], sort=False):
        valid_indices.extend(group.index[::step].tolist())

    valid_mask = df.index.isin(valid_indices)

    return (
        df.loc[~valid_mask].copy(),
        df.loc[valid_mask].copy(),
    )


class FullCoverageLarge50Sampler(Sampler):
    """
    全インデックスを1回ずつ使い、大変位を追加して50:50にする。

    各epoch:
        1. 全件を1回ずつ並べる。
        2. 大変位群から不足分を復元抽出する。
        3. 全体をshuffleする。

    各batchではなく、epoch全体で50:50となる。
    """

    def __init__(
        self,
        is_large,
        generator: torch.Generator = None,
    ):
        labels = torch.as_tensor(is_large)

        if labels.ndim != 1 or labels.numel() == 0:
            raise ValueError(
                "is_largeは空でない1次元配列にしてください。"
            )

        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError(
                "is_largeには0または1を指定してください。"
            )

        labels = labels.to(dtype=torch.long, device="cpu")

        self.n_total = labels.numel()
        self.large_indices = torch.where(labels == 1)[0]
        self.n_large = self.large_indices.numel()
        self.n_normal = self.n_total - self.n_large
        self.generator = generator

        if self.n_large == 0:
            raise ValueError(
                "大変位ペアが0件なので、50%提示にできません。"
            )

        if self.n_large > self.n_normal:
            raise ValueError(
                "大変位ペアが既に50%を超えています。"
                "大変位の追加だけでは50%にできません。"
            )

        self.n_add = self.n_normal - self.n_large
        self.epoch_size = self.n_total + self.n_add

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        all_indices = torch.arange(
            self.n_total,
            dtype=torch.long,
        )

        if self.n_add > 0:
            selected = torch.randint(
                low=0,
                high=self.n_large,
                size=(self.n_add,),
                generator=self.generator,
            )

            extra_indices = self.large_indices[selected]

            indices = torch.cat(
                [all_indices, extra_indices]
            )
        else:
            indices = all_indices

        order = torch.randperm(
            indices.numel(),
            generator=self.generator,
        )

        return iter(indices[order].tolist())


class NoisedFrameDataModule(L.LightningDataModule):
    """
    元のNoisedFrame用samplingを維持する。
    今回の大変位50%方式は適用しない。
    """

    def __init__(
        self,
        data_path: str,
        index_files: list[str],
        time_delay_steps: int = 100,
        cutoff: float = 5.0,
        pbc: bool = True,
        in_memory: bool = False,
        batch_size: int = 16,
        num_train_samples: int = 4000,
        num_valid_samples: int = 400,
        valid_fraction: float = 0.2,
        train_valid_split: bool = True,
        sample_weight_comp: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        noise: UniformScaleNormalPrior = UniformScaleNormalPrior(),
        prior: Prior = MaxwellBoltzmannPrior(scale=0.1),
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.index_files = index_files
        self.time_delay_steps = time_delay_steps
        self.cutoff = cutoff
        self.pbc = pbc
        self.in_memory = in_memory
        self.batch_size = batch_size
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples
        self.valid_fraction = valid_fraction
        self.train_valid_split = train_valid_split
        self.sample_weight_comp = sample_weight_comp
        self.num_workers = num_workers
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.noise = noise
        self.prior = prior

    def setup(self, stage: str = None):
        if stage not in (None, "fit", "validate"):
            raise NotImplementedError(
                f"Unsupported stage: {stage}"
            )

        df = pd.concat(
            [
                pd.read_csv(self.data_path / filename)
                for filename in self.index_files
            ],
            ignore_index=True,
        )

        if self.train_valid_split:
            self.df_train, self.df_valid = (
                _temporal_train_valid_split(
                    df,
                    self.valid_fraction,
                )
            )
        else:
            self.df_train = df.copy()

        if "weight" in self.df_train.columns:
            self.weights = self.df_train["weight"].to_numpy()
            print(
                "CSVのweight列をNoisedFrameの"
                "学習samplingに使用します。"
            )
        else:
            self.weights = np.ones(
                len(self.df_train),
                dtype=float,
            )

            if self.sample_weight_comp:
                comp_count = (
                    self.df_train.groupby("comp")
                    .transform("count")["name"]
                )
                self.weights *= 1.0 / comp_count.to_numpy()

        if self.train_valid_split:
            if "weight" in self.df_valid.columns:
                self.valid_weights = (
                    self.df_valid["weight"].to_numpy()
                )
            else:
                self.valid_weights = np.ones(
                    len(self.df_valid),
                    dtype=float,
                )

        dataset_kwargs = dict(
            data_path=self.data_path,
            time_delay_steps=self.time_delay_steps,
            noise=self.noise,
            prior=self.prior,
            cutoff=self.cutoff,
            pbc=self.pbc,
            in_memory=self.in_memory,
        )

        self.train_dataset = NoisedFrameDataset(
            df=self.df_train,
            **dataset_kwargs,
        )

        if self.train_valid_split:
            self.valid_dataset = NoisedFrameDataset(
                df=self.df_valid,
                **dataset_kwargs,
            )
        else:
            self.valid_dataset = self.train_dataset

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.num_train_samples,
            replacement=True,
            generator=self.generator,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        sampler = RandomSampler(
            self.valid_dataset,
            replacement=True,
            num_samples=self.num_valid_samples,
            generator=self.generator,
        )

        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator,
            sampler=sampler,
            pin_memory=True,
        )

    def test_dataloader(self):
        raise NotImplementedError


class TimeDelayedPairDataModule(L.LightningDataModule):
    """
    固定ペアCSVを用いた、大変位50%提示用DataModule。

    train:
        全ペアを1回 + 大変位を追加し、epoch全体で50:50。

    validation:
        全ペアを1回ずつ使用。追加抽出なし。

    sample_weights:
        trainの元の等重みペア平均へ補正する係数を設定する。
        lossへの適用はFlowModule側で行う。

    単一プロセス・単一GPU向け。
    """

    def __init__(
        self,
        data_path: str,
        index_files: list[str],
        time_delay_steps: int = 100,
        cutoff: float = 5.0,
        pbc: bool = True,
        in_memory: bool = False,
        batch_size: int = 16,
        num_train_samples: int = 4000,
        num_valid_samples: int = 400,
        valid_fraction: float = 0.2,
        train_valid_split: bool = True,
        sample_weight_comp: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        prior: Prior = AdaptiveMaxwellBoltzmannPrior(),
        neighbor_list_both_ends: bool = False,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.index_files = index_files
        self.time_delay_steps = time_delay_steps
        self.cutoff = cutoff
        self.pbc = pbc
        self.in_memory = in_memory
        self.batch_size = batch_size

        # 設定ファイルとの互換性のため保持するが、
        # 今回の全件使用方式では提示数の指定には使わない。
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples

        self.valid_fraction = valid_fraction
        self.train_valid_split = train_valid_split
        self.sample_weight_comp = sample_weight_comp
        self.num_workers = num_workers
        self.seed = seed

        # samplingとworker用の乱数列を分離する。
        self.sampler_generator = (
            torch.Generator().manual_seed(seed)
        )
        self.generator = (
            torch.Generator().manual_seed(seed + 1)
        )

        self.prior = prior
        self.neighbor_list_both_ends = neighbor_list_both_ends

    def _check_single_process(self):
        trainer = getattr(self, "_trainer", None)

        if trainer is not None and trainer.world_size != 1:
            raise RuntimeError(
                "このLarge50 samplerは単一プロセス用です。"
                "今回はdevices=1で実行してください。"
            )

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() != 1
        ):
            raise RuntimeError(
                "分散学習ではこのLarge50 samplerを使用できません。"
            )

    def setup(self, stage: str = None):
        if stage not in (None, "fit", "validate"):
            raise NotImplementedError(
                f"Unsupported stage: {stage}"
            )

        self._check_single_process()

        if not self.train_valid_split:
            raise ValueError(
                "今回のLarge50実験では"
                "train_valid_split=Trueにしてください。"
            )

        if self.sample_weight_comp:
            raise ValueError(
                "今回のLarge50実験では"
                "sample_weight_comp=Falseにしてください。"
            )

        if not self.index_files:
            raise ValueError("index_filesが空です。")

        df = pd.concat(
            [
                pd.read_csv(self.data_path / filename)
                for filename in self.index_files
            ],
            ignore_index=True,
        )

        if "weight" in df.columns:
            raise ValueError(
                "今回のLarge50実験ではCSVのweight列を使いません。"
                "新しく作成したis_large列付きCSVを指定してください。"
            )

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
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"CSVに必要な列がありません: {missing}"
            )

        if df.empty:
            raise ValueError("CSVが空です。")

        # 同じframeペアの重複を黙って学習に入れない。
        pair_columns = [
            "name",
            "temp",
            "t_start",
            "t_end",
        ]

        duplicate_mask = df.duplicated(
            subset=pair_columns,
            keep=False,
        )

        if duplicate_mask.any():
            examples = df.loc[
                duplicate_mask,
                pair_columns + ["prior_Li", "prior_frame"],
            ].head()

            raise ValueError(
                "同じframeペアの重複があります。"
                "prior設定が異なる場合も、意図を確認してください。\n"
                f"{examples.to_string(index=False)}"
            )

        self.df_train, self.df_valid = (
            _temporal_train_valid_split(
                df,
                self.valid_fraction,
            )
        )

        self.df_train = self.df_train.reset_index(drop=True)
        self.df_valid = self.df_valid.reset_index(drop=True)

        if self.df_train.empty or self.df_valid.empty:
            raise ValueError(
                "trainまたはvalidationが空です。"
                "CSVと分割条件を確認してください。"
            )

        dataset_kwargs = dict(
            data_path=self.data_path,
            time_delay_steps=self.time_delay_steps,
            prior=self.prior,
            cutoff=self.cutoff,
            pbc=self.pbc,
            neighbor_list_both_ends=self.neighbor_list_both_ends,
            in_memory=self.in_memory,
        )

        self.train_dataset = TimeDelayedPairDataset(
            df=self.df_train,
            **dataset_kwargs,
        )

        self.valid_dataset = TimeDelayedPairDataset(
            df=self.df_valid,
            **dataset_kwargs,
        )

        # Dataset側で検証・整数化されたラベルを使う。
        labels = self.train_dataset.df[
            "is_large"
        ].to_numpy(dtype=np.int64)

        self.train_sampler = FullCoverageLarge50Sampler(
            is_large=labels,
            generator=self.sampler_generator,
        )

        n_total = self.train_sampler.n_total
        n_large = self.train_sampler.n_large
        n_normal = self.train_sampler.n_normal

        large_fraction = n_large / n_total

        self.normal_loss_weight = 2.0 * (
            1.0 - large_fraction
        )
        self.large_loss_weight = 2.0 * large_fraction

        self.train_dataset.sample_weights = np.where(
            labels == 1,
            self.large_loss_weight,
            self.normal_loss_weight,
        ).astype(np.float64)

        self.valid_dataset.sample_weights = np.ones(
            len(self.valid_dataset),
            dtype=np.float64,
        )

        # 今回の原子平均loss補正は、全構造で原子数が同じ前提。
        all_names = pd.unique(df["name"])

        atom_counts = {
            len(self.train_dataset.atomic_numbers[name])
            for name in all_names
        }

        if len(atom_counts) != 1:
            raise ValueError(
                "構造間で原子数が異なります。"
                "今回のloss集計をそのまま使えません。"
            )

        self._check_sampler(labels)

        n_valid = len(self.valid_dataset)
        n_valid_large = int(
            self.valid_dataset.df["is_large"].sum()
        )

        print(
            "\n[Large50 setup]\n"
            f"train_pairs={n_total}\n"
            f"train_normal={n_normal}\n"
            f"train_large={n_large}\n"
            f"train_large_fraction={large_fraction:.10f}\n"
            f"extra_large={self.train_sampler.n_add}\n"
            f"epoch_presentations={len(self.train_sampler)}\n"
            f"normal_loss_weight={self.normal_loss_weight:.12g}\n"
            f"large_loss_weight={self.large_loss_weight:.12g}\n"
            f"validation_pairs={n_valid}\n"
            f"validation_large={n_valid_large}\n"
            "validation_sample_weight=1\n"
            "num_train_samples / num_valid_samples are ignored.\n"
        )

    def _check_sampler(self, labels):
        # 検査で本学習用の乱数列を進めない。
        state = self.sampler_generator.get_state().clone()

        try:
            sampled = np.asarray(
                list(iter(self.train_sampler)),
                dtype=np.int64,
            )
        finally:
            self.sampler_generator.set_state(state)

        n_total = len(labels)

        counts = np.bincount(
            sampled,
            minlength=n_total,
        )

        expected_per_group = self.train_sampler.n_normal

        if len(sampled) != 2 * expected_per_group:
            raise RuntimeError("samplerの総提示数が不正です。")

        if not np.all(counts >= 1):
            raise RuntimeError(
                "1回も提示されないtrainペアがあります。"
            )

        if not np.all(counts[labels == 0] == 1):
            raise RuntimeError(
                "通常ペアの提示回数が1回ではありません。"
            )

        n_large_presented = int(labels[sampled].sum())

        if n_large_presented != expected_per_group:
            raise RuntimeError(
                "大変位の提示割合が50%ではありません。"
            )

        mean_weight = float(
            self.train_dataset.sample_weights[sampled].mean()
        )

        if not np.isclose(
            mean_weight,
            1.0,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                "補正係数のepoch平均が1ではありません。"
                f" mean_weight={mean_weight}"
            )

        print(
            "[Large50 sampler check] "
            f"presentations={len(sampled)}, "
            f"large_presentations={n_large_presented}, "
            f"min_coverage={counts.min()}, "
            f"mean_weight={mean_weight:.12g}"
        )

    def train_dataloader(self):
        self._check_single_process()

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            drop_last=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        self._check_single_process()

        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator,
            pin_memory=True,
            persistent_workers=False,
        )

    def test_dataloader(self):
        raise NotImplementedError