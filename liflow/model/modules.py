# 大変位50%提示 + 構造単位の逆確率補正用FlowModule。
# 従来の原子別・距離ヒストグラム重みは使用しない。

import csv
import os

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch

from liflow.model.models import DualPaiNN


# 化学元素の原子番号ではなく、モデル内部の元素ID。
# 今回のデータでは、CSV作成時と同じ対象原子を指定する。
TARGET_ELEM_ID = 1

# 診断専用。大変位ペアの分類閾値とは別。
DISP_LOG_BINS_A = np.array(
    [0.0, 1.0, 3.0, np.inf],
    dtype=np.float64,
)


class FlowModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        self.model = DualPaiNN(
            num_features=cfg.model.num_features,
            num_radial_basis=cfg.model.num_radial_basis,
            num_layers=cfg.model.num_layers,
            num_elements=cfg.model.num_elements,
            r_max=cfg.model.r_max,
            r_offset=cfg.model.r_offset,
            ref_temp=cfg.model.ref_temp,
        )

        self.save_hyperparameters()

        # 学習lossには使わない診断用集計。
        self._loss_stats = {}

    def setup(self, stage: str = None):
        pass

    def _use_active_weight(self) -> bool:
        value = self.cfg.get(
            "active_learning_weight",
            False,
        )

        if not isinstance(value, bool):
            raise ValueError(
                "active_learning_weightにはbool値の"
                "TrueまたはFalseを指定してください。"
                f" 現在の値={value!r}"
            )

        return value

    def on_fit_start(self):
        if self._use_active_weight():
            raise ValueError(
                "今回のLarge50実験では"
                "active_learning_weight=Falseにしてください。"
                "従来の距離ヒストグラム重みは併用しません。"
            )

        if self.trainer.world_size != 1:
            raise RuntimeError(
                "今回のLarge50実験は単一プロセス用です。"
                "devices=1で実行してください。"
            )

        datamodule = self.trainer.datamodule

        if (
            datamodule is None
            or not hasattr(datamodule, "train_sampler")
            or not hasattr(datamodule, "normal_loss_weight")
            or not hasattr(datamodule, "large_loss_weight")
        ):
            raise RuntimeError(
                "Large50対応のTimeDelayedPairDataModuleを"
                "使用してください。"
            )

        self.print(
            "[Large50 loss] "
            "active_learning_weight=False, "
            "train_sampling_correction=True, "
            "validation_sampling_correction=False, "
            f"target_element_id={TARGET_ELEM_ID}"
        )

        self.print(
            "[Large50 loss] "
            f"normal_weight={datamodule.normal_loss_weight:.12g}, "
            f"large_weight={datamodule.large_loss_weight:.12g}, "
            "reduction=mean(sample_weight * sum_xyz_squared_error), "
            "batch_weight_renormalization=False"
        )

    def forward(
        self,
        batch: Batch,
        t_per_graph: torch.Tensor,
    ) -> torch.Tensor:
        batch["time"] = t_per_graph[batch.batch]
        return self.model(batch)

    # =====================================================
    # 損失計算
    # =====================================================
    def _compute_loss_details(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        elements: torch.Tensor = None,
        true_disp: torch.Tensor = None,
        atom_weight: torch.Tensor = None,
    ):
        """
        elementsとtrue_dispは既存の呼出形式との互換性のため残す。
        従来の距離ヒストグラム重みには使用しない。
        """
        if self._use_active_weight():
            raise ValueError(
                "このFlowModuleでは従来の距離重みを使用しません。"
                "active_learning_weight=Falseにしてください。"
            )

        if (
            outputs.ndim != 2
            or outputs.shape[-1] != 3
            or outputs.shape != targets.shape
        ):
            raise ValueError(
                "outputsとtargetsは同じ形状[N_atoms, 3]が必要です。"
                f" outputs={tuple(outputs.shape)},"
                f" targets={tuple(targets.shape)}"
            )

        # 元の尺度を維持：xyzを合計してから原子平均する。
        per_atom_sq_err = (
            (outputs - targets).pow(2).sum(-1)
        )

        if atom_weight is None:
            weight = torch.ones_like(per_atom_sq_err)
        else:
            if atom_weight.shape != per_atom_sq_err.shape:
                raise ValueError(
                    "atom_weightの形状が不正です。"
                    f" weight={tuple(atom_weight.shape)},"
                    f" error={tuple(per_atom_sq_err.shape)}"
                )

            weight = atom_weight.detach().to(
                device=per_atom_sq_err.device,
                dtype=per_atom_sq_err.dtype,
            )

            if (
                not torch.isfinite(weight).all()
                or not (weight > 0).all()
            ):
                raise ValueError(
                    "atom_weightは有限の正数が必要です。"
                )

        # 重みの合計では割らない。
        loss = (per_atom_sq_err * weight).mean()

        return loss, per_atom_sq_err, weight

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        elements: torch.Tensor = None,
        true_disp: torch.Tensor = None,
        atom_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        loss, _, _ = self._compute_loss_details(
            outputs=outputs,
            targets=targets,
            elements=elements,
            true_disp=true_disp,
            atom_weight=atom_weight,
        )

        return loss

    def _get_sampling_weights(self, batch: Batch):
        """
        構造ごとのsample_weightを全原子へ展開する。
        """
        if "sample_weight" not in batch:
            raise ValueError(
                "batchにsample_weightがありません。"
                "変更済みのDatasetを使用してください。"
            )

        if "is_large" not in batch:
            raise ValueError(
                "batchにis_largeがありません。"
                "is_large列付きCSVと変更済みDatasetが必要です。"
            )

        pair_weight = batch["sample_weight"].reshape(-1)
        is_large = batch["is_large"].reshape(-1)

        if pair_weight.numel() != batch.num_graphs:
            raise ValueError(
                "sample_weightは構造ごとに1個必要です。"
            )

        if is_large.numel() != batch.num_graphs:
            raise ValueError(
                "is_largeは構造ごとに1個必要です。"
            )

        if (
            not torch.isfinite(pair_weight).all()
            or not (pair_weight > 0).all()
        ):
            raise ValueError(
                "sample_weightは有限の正数が必要です。"
            )

        if not torch.all((is_large == 0) | (is_large == 1)):
            raise ValueError(
                "is_largeには0または1が必要です。"
            )

        # DataModuleでも確認するが、batch内でも誤使用を防ぐ。
        atom_counts = batch.ptr[1:] - batch.ptr[:-1]

        if not torch.all(atom_counts == atom_counts[0]):
            raise ValueError(
                "構造間で原子数が異なります。"
                "今回の原子平均loss補正は等原子数を前提とします。"
            )

        atom_weight = pair_weight[batch.batch]

        return pair_weight, atom_weight, is_large

    # =====================================================
    # 対象原子の正解変位量別の診断
    # sampling補正前の誤差を記録する。
    # =====================================================
    def _reset_loss_stats(self, stage: str):
        num_bins = len(DISP_LOG_BINS_A) - 1

        self._loss_stats[stage] = {
            "counts": torch.zeros(
                num_bins,
                dtype=torch.int64,
                device=self.device,
            ),
            "sq_error_sums": torch.zeros(
                num_bins,
                dtype=torch.float64,
                device=self.device,
            ),
            "nonfinite_batches": 0,
        }

    @torch.no_grad()
    def _accumulate_loss_stats(
        self,
        stage: str,
        per_atom_sq_err: torch.Tensor,
        elements: torch.Tensor,
        true_disp: torch.Tensor,
    ):
        stats = self._loss_stats[stage]

        err = per_atom_sq_err.detach()
        elem = elements.detach()
        disp = true_disp.detach()

        if (
            err.ndim != 1
            or elem.shape != err.shape
            or disp.shape != (err.shape[0], 3)
        ):
            raise ValueError(
                "診断用テンソルの形状が不正です。"
                f" error={tuple(err.shape)},"
                f" elements={tuple(elem.shape)},"
                f" true_disp={tuple(disp.shape)}"
            )

        mask = elem == TARGET_ELEM_ID

        if not mask.any():
            return

        target_error = err[mask].to(
            dtype=torch.float64
        )
        target_disp = disp[mask].to(
            dtype=torch.float64
        ).norm(dim=-1)

        if (
            not torch.isfinite(target_error).all()
            or not torch.isfinite(target_disp).all()
        ):
            raise ValueError(
                "対象原子の診断集計に非有限値があります。"
            )

        boundaries = torch.as_tensor(
            DISP_LOG_BINS_A[1:-1],
            dtype=torch.float64,
            device=target_disp.device,
        )

        bin_idx = torch.bucketize(
            target_disp,
            boundaries,
            right=True,
        )

        num_bins = len(DISP_LOG_BINS_A) - 1

        stats["counts"].add_(
            torch.bincount(
                bin_idx,
                minlength=num_bins,
            )
        )

        stats["sq_error_sums"].scatter_add_(
            0,
            bin_idx,
            target_error,
        )

    @torch.no_grad()
    def _write_loss_stats(self, stage: str):
        if stage not in self._loss_stats:
            return

        stats = self._loss_stats[stage]

        counts = stats["counts"].clone()
        error_sums = stats["sq_error_sums"].clone()

        nonfinite_batches = torch.tensor(
            stats["nonfinite_batches"],
            dtype=torch.int64,
            device=self.device,
        )

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(counts)
            torch.distributed.all_reduce(error_sums)
            torch.distributed.all_reduce(nonfinite_batches)

        if not self.trainer.is_global_zero:
            return

        counts = counts.cpu().tolist()
        error_sums = error_sums.cpu().tolist()

        total_count = sum(counts)
        total_error_sum = sum(error_sums)

        split = (
            "validation"
            if stage == "valid"
            else stage
        )

        rows = []

        for bin_id, count_value in enumerate(counts):
            count = int(count_value)
            error_sum = float(error_sums[bin_id])

            rows.append({
                "epoch": int(self.current_epoch),
                "split": split,
                "prediction_mode": str(
                    self.cfg.model.prediction_mode
                ),
                "active_learning_weight": False,
                "disp_min_A": float(
                    DISP_LOG_BINS_A[bin_id]
                ),
                "disp_max_A": float(
                    DISP_LOG_BINS_A[bin_id + 1]
                ),
                "count": count,
                "mean_sq_error": (
                    error_sum / count
                    if count > 0
                    else float("nan")
                ),
                "sum_sq_error": error_sum,
                "H_error_fraction": (
                    error_sum / total_error_sum
                    if total_error_sum > 0
                    else float("nan")
                ),
                "nonfinite_batches": int(
                    nonfinite_batches.item()
                ),
            })

        output_dir = self.cfg.trainer.get(
            "output_dir",
            ".",
        )

        if output_dir is None:
            output_dir = "."

        save_dir = os.path.join(
            str(output_dir),
            "loss_logs",
        )
        os.makedirs(save_dir, exist_ok=True)

        # 従来のログと混ぜない。
        # Hという列名・ファイル名は既存形式との互換性のため維持。
        filename = (
            "loss_by_H_displacement_epoch_"
            "large50_importance_corrected.csv"
        )

        csv_path = os.path.join(
            save_dir,
            filename,
        )

        fieldnames = [
            "epoch",
            "split",
            "prediction_mode",
            "active_learning_weight",
            "disp_min_A",
            "disp_max_A",
            "count",
            "mean_sq_error",
            "sum_sq_error",
            "H_error_fraction",
            "nonfinite_batches",
        ]

        file_has_content = (
            os.path.isfile(csv_path)
            and os.path.getsize(csv_path) > 0
        )

        if file_has_content:
            with open(
                csv_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                existing_header = next(
                    csv.reader(f),
                    None,
                )

            if existing_header != fieldnames:
                raise ValueError(
                    "既存CSVの列構成が異なります。"
                    "出力先を変更するか、既存CSVを退避してください: "
                    f"{csv_path}"
                )

        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            if not file_has_content:
                writer.writeheader()

            writer.writerows(rows)

        mean_target_error = (
            total_error_sum / total_count
            if total_count > 0
            else float("nan")
        )

        self.print(
            "[Target displacement loss CSV] "
            f"epoch={self.current_epoch}, "
            f"split={split}, "
            f"count={total_count}, "
            f"mean_target_sq_error={mean_target_error:.6g}, "
            f"nonfinite_batches={int(nonfinite_batches.item())}"
        )

    # =====================================================
    # エポック開始・終了
    # =====================================================
    def on_train_epoch_start(self):
        self._reset_loss_stats("train")

    def on_train_epoch_end(self):
        self._write_loss_stats("train")

    def on_validation_epoch_start(self):
        self._reset_loss_stats("valid")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        self._write_loss_stats("valid")

    # =====================================================
    # Training
    # =====================================================
    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ):
        pair_weight, atom_weight, is_large = (
            self._get_sampling_weights(batch)
        )

        t = torch.rand(
            (batch.num_graphs,),
            device=self.device,
            dtype=torch.float,
        )

        x_cond = batch["positions_1"].clone()

        # priorを加える前の正解変位。
        disp = (
            batch["positions_2"]
            - batch["positions_1"]
        ).clone()

        batch["positions_1"] = (
            batch["positions_1"]
            + batch["prior"]
        )

        _t = t[batch.batch][:, None]

        x_t = (
            (1.0 - _t) * batch["positions_1"]
            + _t * batch["positions_2"]
        )

        dx_dt = (
            batch["positions_2"]
            - batch["positions_1"]
        )

        batch["positions_1"] = x_cond
        batch["positions_2"] = x_t

        outputs = self(
            batch,
            t_per_graph=t,
        )

        if self.cfg.model.prediction_mode == "velocity":
            target = dx_dt
        elif self.cfg.model.prediction_mode == "data":
            target = disp
        else:
            raise ValueError(
                "Unknown prediction mode: "
                f"{self.cfg.model.prediction_mode}"
            )

        loss, per_atom_sq_err, _ = (
            self._compute_loss_details(
                outputs=outputs,
                targets=target,
                atom_weight=atom_weight,
            )
        )

        # 提示割合は、loss異常の有無にかかわらず記録する。
        self.log(
            "train/large_fraction",
            is_large.float().mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        self.log(
            "train/sample_weight_mean",
            pair_weight.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        if not torch.isfinite(loss):
            self._loss_stats["train"][
                "nonfinite_batches"
            ] += 1

            self.print(
                f"[WARNING] Skipping batch {batch_idx} "
                f"(epoch {self.current_epoch}): "
                f"non-finite loss={loss.item()}"
            )

            # 元の単一GPU向けスキップ動作を維持。
            return None

        self._accumulate_loss_stats(
            stage="train",
            per_atom_sq_err=per_atom_sq_err,
            elements=batch["elements"],
            true_disp=disp,
        )

        # optimizerへ渡すsampling補正後のloss。
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        # 50%提示分布上の未補正loss。学習には使わない。
        self.log(
            "train/loss_uncorrected",
            per_atom_sq_err.detach().mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        return loss

    # =====================================================
    # Optimizer
    # =====================================================
    def configure_optimizers(self):
        if self.cfg.optimizer.class_name != "Adam":
            raise ValueError(
                "Unknown optimizer class: "
                f"{self.cfg.optimizer.class_name}"
            )

        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
        )

    # =====================================================
    # Validation
    # =====================================================
    @torch.no_grad()
    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> torch.Tensor:
        if self._use_active_weight():
            raise ValueError(
                "active_learning_weight=Falseにしてください。"
            )

        if self.cfg.valid.num_time_steps < 1:
            raise ValueError(
                "valid.num_time_stepsは1以上にしてください。"
            )

        # validationは補正不要。Dataset側も全て1であることを確認。
        if "sample_weight" in batch:
            valid_weight = batch["sample_weight"].reshape(-1)

            if (
                valid_weight.numel() != batch.num_graphs
                or not torch.all(valid_weight == 1)
            ):
                raise ValueError(
                    "validationのsample_weightは"
                    "構造ごとに1個、全て1が必要です。"
                )

        t_list = torch.linspace(
            0.0,
            1.0,
            self.cfg.valid.num_time_steps,
            device=self.device,
            dtype=torch.float,
        )

        orig_batch = batch.clone()
        loss_list = []

        record_stats = not self.trainer.sanity_checking
        batch_has_nonfinite = False

        for t in t_list:
            t_per_graph = (
                t.expand(batch.num_graphs).clone()
            )

            x_cond = (
                orig_batch["positions_1"].clone()
            )

            disp = (
                orig_batch["positions_2"]
                - orig_batch["positions_1"]
            ).clone()

            batch["positions_1"] = (
                orig_batch["positions_1"]
                + orig_batch["prior"]
            ).clone()

            batch["positions_2"] = (
                orig_batch["positions_2"].clone()
            )

            _t = t_per_graph[batch.batch][:, None]

            x_t = (
                (1.0 - _t) * batch["positions_1"]
                + _t * batch["positions_2"]
            )

            dx_dt = (
                batch["positions_2"]
                - batch["positions_1"]
            )

            batch["positions_1"] = x_cond
            batch["positions_2"] = x_t

            outputs = self(
                batch,
                t_per_graph=t_per_graph,
            )

            if self.cfg.model.prediction_mode == "velocity":
                target = dx_dt
            elif self.cfg.model.prediction_mode == "data":
                target = disp
            else:
                raise ValueError(
                    "Unknown prediction mode: "
                    f"{self.cfg.model.prediction_mode}"
                )

            # validationではatom_weightを渡さない。
            loss, per_atom_sq_err, _ = (
                self._compute_loss_details(
                    outputs=outputs,
                    targets=target,
                )
            )

            if not torch.isfinite(loss):
                batch_has_nonfinite = True

                self.print(
                    "[WARNING] Non-finite validation loss: "
                    f"epoch={self.current_epoch}, "
                    f"batch={batch_idx}, "
                    f"t={t.item():.4f}, "
                    f"loss={loss.item()}"
                )

            elif record_stats:
                self._accumulate_loss_stats(
                    stage="valid",
                    per_atom_sq_err=per_atom_sq_err,
                    elements=batch["elements"],
                    true_disp=disp,
                )

            # 非有限値はvalid/lossに残し、異常を隠さない。
            loss_list.append(loss)

        if record_stats and batch_has_nonfinite:
            self._loss_stats["valid"][
                "nonfinite_batches"
            ] += 1

        loss = torch.stack(loss_list).mean()

        self.log(
            "valid/loss",
            loss,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        return loss