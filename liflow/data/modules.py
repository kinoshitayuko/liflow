import random
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from liflow.data.dataset import NoisedFrameDataset, TimeDelayedPairDataset
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
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.load_data()  # reload npz arrays

# # ← 変更①: ヘルパー関数を追加 ソートして後半をvalidにする（時系列的に近いものがtrain/valid両方に入るのを防ぐ）
# def _temporal_train_valid_split(
#     df: pd.DataFrame, valid_fraction: float
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     df = df.sort_values(["name", "temp", "t_start"]).reset_index(drop=True)
#     valid_indices = []
#     for _, group in df.groupby(["name", "temp"], sort=False):
#         n = len(group)
#         n_valid = max(1, round(n * valid_fraction))
#         valid_indices.extend(group.index[-n_valid:].tolist())
#     mask = df.index.isin(valid_indices)
#     return df[~mask].copy(), df[mask].copy()

def _temporal_train_valid_split(
    df: pd.DataFrame, valid_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["name", "temp", "t_start"]).reset_index(drop=True)
    valid_indices = []
    for _, group in df.groupby(["name", "temp"], sort=False):
        n = len(group)
        step = max(1, round(1.0 / valid_fraction))  # valid_fraction=0.2 → step=5
        # 0, 5, 10, 15, ... 番目をval → 全体に均等に分布
        val_idx = group.index[::step].tolist()
        valid_indices.extend(val_idx)
    mask = df.index.isin(valid_indices)
    return df[~mask].copy(), df[mask].copy()

class NoisedFrameDataModule(L.LightningDataModule):
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
        valid_fraction: float = 0.2,        # ← 変更②: 追加
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
        self.valid_fraction = valid_fraction        # ← 変更②: 追加
        self.train_valid_split = train_valid_split
        self.sample_weight_comp = sample_weight_comp
        self.num_workers = num_workers
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.noise = noise
        self.prior = prior

    def setup(self, stage: str = None):
        # Load train + valid data
        df = pd.concat([pd.read_csv(self.data_path / f) for f in self.index_files])
        # if self.train_valid_split:
        #     # select num_valid_samples from df_train
        #     self.df_valid = df.sample(
        #         n=self.num_valid_samples, replace=False, random_state=self.seed
        #     ).copy()
        #     self.df_train = df.drop(self.df_valid.index).copy()
        # else:
        #     self.df_train = df.copy()
        
        # ← 変更③: 分割ロジックをまるごと差し替え
        if self.train_valid_split:
            self.df_train, self.df_valid = _temporal_train_valid_split(
                df, self.valid_fraction
            )
        else:
            self.df_train = df.copy()

        # # 以下は元のコードのまま
        # # Sample weights for training
        # self.weights = np.ones(len(self.df_train), dtype=float)
        # if self.sample_weight_comp:
        #     comp_count = self.df_train.groupby("comp").transform("count")["name"]
        #     self.weights *= 1.0 / comp_count.to_numpy()
        
        # 20260728追加: CSVのweight列をそのまま使うように変更
        # Sample weights for training
        if "weight" in self.df_train.columns:
            # 先ほど作成したCSVの "weight" 列をそのままサンプリングの確率に使う！
            self.weights = self.df_train["weight"].to_numpy()
            print("CSVから weight 列を読み込んでサンプリングに適用します。")
        else:
            # もし weight 列がない昔のCSVを読み込んだ時のための予備処理（今まで通り）
            self.weights = np.ones(len(self.df_train), dtype=float)
            if self.sample_weight_comp:
                comp_count = self.df_train.groupby("comp").transform("count")["name"]
                self.weights *= 1.0 / comp_count.to_numpy()
                
                
        # Valid用の重み 20260730追加: CSVのweight列をそのまま使うように変更
        if self.train_valid_split:
            if "weight" in self.df_valid.columns:
                self.valid_weights = self.df_valid["weight"].to_numpy()
            else:
                self.valid_weights = np.ones(len(self.df_valid), dtype=float)
                

        dataset_kwargs = dict(
            data_path=self.data_path,
            time_delay_steps=self.time_delay_steps,
            noise=self.noise,
            prior=self.prior,
            cutoff=self.cutoff,
            pbc=self.pbc,
            in_memory=self.in_memory,
        )

        if stage in ["fit", "validate"]:
            self.train_dataset = NoisedFrameDataset(df=self.df_train, **dataset_kwargs)
            if self.train_valid_split:
                self.valid_dataset = NoisedFrameDataset(
                    df=self.df_valid, **dataset_kwargs
                )
            else:
                self.valid_dataset = self.train_dataset
        else:  # test
            raise NotImplementedError

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
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.train_valid_split:
            # 20260729修正: num_valid_samples が効くように sampler を追加
            sampler = RandomSampler(
                self.valid_dataset,
                replacement=True,
                num_samples=self.num_valid_samples,
                generator=self.generator,
            )
            
            # # 2026/07/29追加
            # # WeightedRandomSampler で num_valid_samples 件サンプリング → 400サンプル
            # sampler = WeightedRandomSampler(
            # weights=self.valid_weights,
            # num_samples=self.num_valid_samples,
            # replacement=True,
            # generator=self.generator,
            # )
            
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                sampler=sampler,           # ← 追加
                #shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker,
                generator=self.generator,  # ← 追加
                pin_memory=True,
                # shuffle=False は sampler 使用時は指定しない（削除）
            )
            
            # #元々のコード
            # return DataLoader(
            #     self.valid_dataset,
            #     batch_size=self.batch_size,
            #     shuffle=False,
            #     num_workers=self.num_workers,
            #     worker_init_fn=seed_worker,
            #     pin_memory=True,
            # )
        else:  # sample from train dataset
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
        valid_fraction: float = 0.2,        # ← 変更②: 追加
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
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples
        self.valid_fraction = valid_fraction        # ← 変更②: 追加
        self.train_valid_split = train_valid_split
        self.sample_weight_comp = sample_weight_comp
        self.num_workers = num_workers
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.prior = prior
        self.neighbor_list_both_ends = neighbor_list_both_ends

    def setup(self, stage: str = None):
        # Load train + valid data
        df = pd.concat([pd.read_csv(self.data_path / f) for f in self.index_files])
        
        # if self.train_valid_split:
        #     # select num_valid_samples from df_train
        #     self.df_valid = df.sample(
        #         n=self.num_valid_samples, replace=False, random_state=self.seed
        #     ).copy()
        #     self.df_train = df.drop(self.df_valid.index).copy()
        # else:
        #     self.df_train = df.copy()
        
        # ← 変更③: 分割ロジックをまるごと差し替え（NoisedFrame と同じ）
        if self.train_valid_split:
            self.df_train, self.df_valid = _temporal_train_valid_split(
                df, self.valid_fraction
            )
        else:
            self.df_train = df.copy()

        # # 以下は元のコードのまま
        # # Sample weights for training
        # self.weights = np.ones(len(self.df_train), dtype=float)
        # if self.sample_weight_comp:
        #     comp_count = self.df_train.groupby("comp").transform("count")["name"]
        #     self.weights *= 1.0 / comp_count.to_numpy()
            
        # 20260728追加: CSVのweight列をそのまま使うように変更
        # Sample weights for training
        if "weight" in self.df_train.columns:
            # 先ほど作成したCSVの "weight" 列をそのままサンプリングの確率に使う！
            self.weights = self.df_train["weight"].to_numpy()
            print("CSVから weight 列を読み込んでサンプリングに適用します。")
        else:
            # もし weight 列がない昔のCSVを読み込んだ時のための予備処理（今まで通り）
            self.weights = np.ones(len(self.df_train), dtype=float)
            if self.sample_weight_comp:
                comp_count = self.df_train.groupby("comp").transform("count")["name"]
                self.weights *= 1.0 / comp_count.to_numpy()
        
        
        # Valid用の重み 20260730追加: CSVのweight列をそのまま使うように変更
        if self.train_valid_split:
            if "weight" in self.df_valid.columns:
                self.valid_weights = self.df_valid["weight"].to_numpy()
            else:
                self.valid_weights = np.ones(len(self.df_valid), dtype=float)

        dataset_kwargs = dict(
            data_path=self.data_path,
            time_delay_steps=self.time_delay_steps,
            prior=self.prior,
            cutoff=self.cutoff,
            pbc=self.pbc,
            neighbor_list_both_ends=self.neighbor_list_both_ends,
            in_memory=self.in_memory,
        )

        if stage in ["fit", "validate"]:
            self.train_dataset = TimeDelayedPairDataset(
                df=self.df_train, **dataset_kwargs
            )
            if self.train_valid_split:
                self.valid_dataset = TimeDelayedPairDataset(
                    df=self.df_valid, **dataset_kwargs
                )
            else:
                self.valid_dataset = self.train_dataset
        else:  # test
            raise NotImplementedError

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.num_train_samples,
            replacement=True,
            generator=self.generator,
        )
        
        ## 20260804コメントアウト
        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     worker_init_fn=seed_worker,
        #     generator=self.generator,
        #     sampler=sampler,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )
        # 20260804修正: samplerを使う場合は shuffle=False にする必要があるので、明示的に指定
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.generator,
            #sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        
        

    def val_dataloader(self):
        if self.train_valid_split:
            # #2026/05/19追加
            # # RandomSampler で num_valid_samples 件サンプリング → 400サンプル
            # sampler = RandomSampler(
            #     self.valid_dataset,
            #     replacement=True,
            #     num_samples=self.num_valid_samples,  # デフォルト 400
            #     generator=self.generator,
            # )
            
            # # 2026/07/30追加
            # # WeightedRandomSampler で num_valid_samples 件サンプリング → 400サンプル
            # sampler = WeightedRandomSampler(
            # weights=self.valid_weights,
            # num_samples=self.num_valid_samples,
            # replacement=True,
            # generator=self.generator,
            # )
            
            # 20260804コメントアウト
            # return DataLoader(
            # self.valid_dataset,
            # batch_size=self.batch_size,
            # sampler=sampler,           # ← samplerを渡す
            # num_workers=self.num_workers,
            # worker_init_fn=seed_worker,
            # generator=self.generator,  # ← generatorも渡す
            # pin_memory=True,
            # # shuffle=False は不要（sampler使用時は指定しない）
            # )
            
            # 20260804修正: samplerを使う場合は shuffle=False にする必要があるので、明示的に指定
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
            )
            
        else:  # sample from train dataset
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