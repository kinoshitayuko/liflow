# # 正解変位に基づく固定重みで学習し、
# # 重み付け前の誤差を正解変位量別にCSVへ集計する。
# import csv
# import os

# import lightning as L
# import numpy as np
# import torch
# from omegaconf import DictConfig
# from torch_geometric.data import Batch

# from liflow.model.models import DualPaiNN


# # =========================================================
# # 学習用：データセットの正解変位分布から事前計算した固定重み
# #
# # alpha = 1.0
# # クリッピング範囲 = [0.1, 10.0]
# # クリッピング後の再正規化は行わない。
# # 学習中にテーブルを更新しない。
# # =========================================================
# HIST_BINS_NP = np.array(
#     [
#         0, 0.3, 0.6, 0.9, 1.2,
#         1.5, 1.8, 2.1, 2.4, 2.7,
#         3, 3.3, 3.6, 3.9, 4.2,
#         4.5, 4.8, 5.1, 5.4, 5.7,
#         6, 6.3, 6.6, 6.9, 7.2,
#         7.5, 7.8, 8.1, 8.4, 8.7,
#         9, 9.3, 9.6, 9.9, 10.2,
#         10.5, 10.8, 11.1, 11.4, 11.7,
#         12, 12.3, 12.6, 12.9, 13.2,
#         13.5, 13.8, 14.1, 14.4, 14.7,
#         15,
#     ],
#     dtype=np.float64,
# )

# HIST_WEIGHTS_NP = np.array(
#     [
#         2.36031418894,
#         0.414914648069,
#         0.21845354135,
#         0.1763533682,
#         0.184284684506,
#         0.2180503666,
#         0.269042339741,
#         0.328591643226,
#         0.415157643423,
#         0.509193088578,
#         0.613750311671,
#         0.732189606321,
#         0.82797142917,
#         0.959677113801,
#         1.07108757131,
#         1.23786042266,
#         1.40604564498,
#         1.50986434808,
#         1.7212818478,
#         1.95284067172,
#         2.10558746419,
#         2.29164132852,
#         2.81488104552,
#         3.24183262675,
#         3.72115787236,
#         4.19455868025,
#         4.62816282371,
#         5.30334200161,
#         6.32928308693,
#         7.63604240965,
#         9.6227865399,
#         10, 10, 10, 10, 10,
#         10, 10, 10, 10, 10,
#         10, 10, 10, 10, 10,
#         10, 10, 10, 10,
#     ],
#     dtype=np.float64,
# )

# # データセット内の内部元素ID。
# # 化学元素の原子番号ではない。
# TARGET_ELEM_ID = 1


# # =========================================================
# # 診断用：対象原子の正解変位量ビン［Å］
# # [0, 1), [1, 3), [3, inf)
# #
# # 学習用のHIST_BINS_NPとは別。
# # 既存CSVとの互換性のため、列名・ファイル名のHは維持する。
# # =========================================================
# DISP_LOG_BINS_A = np.array(
#     [0.0, 1.0, 3.0, np.inf],
#     dtype=np.float64,
# )


# class FlowModule(L.LightningModule):
#     def __init__(self, cfg: DictConfig):
#         super().__init__()

#         self.cfg = cfg

#         self._validate_weight_table()

#         self.model = DualPaiNN(
#             num_features=cfg.model.num_features,
#             num_radial_basis=cfg.model.num_radial_basis,
#             num_layers=cfg.model.num_layers,
#             num_elements=cfg.model.num_elements,
#             r_max=cfg.model.r_max,
#             r_offset=cfg.model.r_offset,
#             ref_temp=cfg.model.ref_temp,
#         )

#         self.save_hyperparameters()

#         # 診断用。モデルパラメータや損失計算には使用しない。
#         self._loss_stats = {}

#     @staticmethod
#     def _validate_weight_table():
#         if HIST_BINS_NP.ndim != 1:
#             raise ValueError(
#                 "HIST_BINS_NPは1次元配列にしてください。"
#             )

#         if HIST_WEIGHTS_NP.ndim != 1:
#             raise ValueError(
#                 "HIST_WEIGHTS_NPは1次元配列にしてください。"
#             )

#         if len(HIST_WEIGHTS_NP) < 1:
#             raise ValueError(
#                 "固定重みテーブルが空です。"
#             )

#         if len(HIST_BINS_NP) != len(HIST_WEIGHTS_NP) + 1:
#             raise ValueError(
#                 "bin境界の個数は重みの個数+1である必要があります。"
#                 f" bins={len(HIST_BINS_NP)},"
#                 f" weights={len(HIST_WEIGHTS_NP)}"
#             )

#         if not np.isfinite(HIST_BINS_NP).all():
#             raise ValueError(
#                 "bin境界にNaNまたはInfが含まれています。"
#             )

#         if not np.all(np.diff(HIST_BINS_NP) > 0):
#             raise ValueError(
#                 "bin境界は狭義単調増加にしてください。"
#             )

#         if (
#             not np.isfinite(HIST_WEIGHTS_NP).all()
#             or not np.all(HIST_WEIGHTS_NP > 0)
#         ):
#             raise ValueError(
#                 "固定重みはすべて有限の正数にしてください。"
#             )

#     def setup(self, stage: str):
#         pass

#     def on_fit_start(self):
#         # __init__ではなく、fit開始時点のcfgを表示する。
#         use_active_weight = self._use_active_weight()

#         self.print(
#             "[Loss weighting] "
#             f"active_learning_weight={use_active_weight}, "
#             f"target_element_id={TARGET_ELEM_ID}"
#         )

#         if use_active_weight:
#             self.print(
#                 "[Loss weighting] "
#                 f"fixed_bins={len(HIST_WEIGHTS_NP)}, "
#                 f"range=[{HIST_BINS_NP[0]:g}, "
#                 f"{HIST_BINS_NP[-1]:g}] A, "
#                 f"weight_min={HIST_WEIGHTS_NP.min():.12g}, "
#                 f"weight_max={HIST_WEIGHTS_NP.max():.12g}, "
#                 "table_renormalization=False, "
#                 "reduction=sum(weight * error) / sum(weight)"
#             )

#     def _use_active_weight(self) -> bool:
#         value = self.cfg.get(
#             "active_learning_weight", False
#         )

#         if not isinstance(value, bool):
#             raise ValueError(
#                 "active_learning_weightにはbool値の"
#                 "TrueまたはFalseを指定してください。"
#                 f" 現在の値={value!r}"
#             )

#         return value

#     def forward(
#         self,
#         batch: Batch,
#         t_per_graph: torch.Tensor,
#     ) -> torch.Tensor:
#         batch["time"] = t_per_graph[batch.batch]
#         return self.model(batch)

#     # =====================================================
#     # 損失計算
#     # =====================================================
#     @torch.no_grad()
#     def _get_loss_weights(
#         self,
#         per_atom_sq_err: torch.Tensor,
#         elements: torch.Tensor = None,
#         true_disp: torch.Tensor = None,
#     ) -> torch.Tensor:
#         weight = torch.ones_like(per_atom_sq_err)

#         if not self._use_active_weight():
#             return weight

#         if elements is None or true_disp is None:
#             raise ValueError(
#                 "active_learning_weight=Trueの場合、"
#                 "elementsとtrue_dispが必要です。"
#             )

#         if (
#             per_atom_sq_err.ndim != 1
#             or elements.shape != per_atom_sq_err.shape
#             or true_disp.shape != (per_atom_sq_err.shape[0], 3)
#         ):
#             raise ValueError(
#                 "重み計算用テンソルの形状が不正です。"
#                 f" error={tuple(per_atom_sq_err.shape)},"
#                 f" elements={tuple(elements.shape)},"
#                 f" true_disp={tuple(true_disp.shape)}"
#             )

#         if (
#             elements.device != per_atom_sq_err.device
#             or true_disp.device != per_atom_sq_err.device
#         ):
#             raise ValueError(
#                 "error、elements、true_dispは"
#                 "同じデバイス上に配置してください。"
#             )

#         mask = elements == TARGET_ELEM_ID

#         if not mask.any():
#             return weight

#         # priorを加える前の正解変位［Å］を使用する。
#         # bin判定はfloat64で行い、配列側の境界精度を維持する。
#         magnitudes = true_disp[mask].to(
#             dtype=torch.float64
#         ).norm(dim=-1)

#         if not torch.isfinite(magnitudes).all():
#             raise ValueError(
#                 "重み計算対象の正解変位に"
#                 "NaNまたはInfが含まれています。"
#             )

#         ratio_weights = torch.as_tensor(
#             HIST_WEIGHTS_NP,
#             dtype=per_atom_sq_err.dtype,
#             device=per_atom_sq_err.device,
#         )

#         # 内部境界だけを使用する。
#         # r=0.3は[0.3, 0.6)へ入る。
#         boundaries = torch.as_tensor(
#             HIST_BINS_NP[1:-1],
#             dtype=torch.float64,
#             device=magnitudes.device,
#         )

#         bin_idx = torch.bucketize(
#             magnitudes,
#             boundaries,
#             right=True,
#         )

#         # 内部境界だけで判定するため、範囲外の有限値も
#         # 自動的に先頭または末尾のbinへ割り当てられる。
#         weight[mask] = ratio_weights[bin_idx]

#         return weight

#     def _compute_loss_details(
#         self,
#         outputs: torch.Tensor,
#         targets: torch.Tensor,
#         elements: torch.Tensor = None,
#         true_disp: torch.Tensor = None,
#     ):
#         per_atom_sq_err = (
#             (outputs - targets).pow(2).sum(-1)
#         )

#         weight = self._get_loss_weights(
#             per_atom_sq_err,
#             elements=elements,
#             true_disp=true_disp,
#         )

#         if self._use_active_weight():
#             loss = (
#                 (per_atom_sq_err * weight).sum()
#                 / weight.sum()
#             )
#         else:
#             loss = per_atom_sq_err.mean()

#         return loss, per_atom_sq_err, weight

#     def compute_loss(
#         self,
#         outputs: torch.Tensor,
#         targets: torch.Tensor,
#         elements: torch.Tensor = None,
#         true_disp: torch.Tensor = None,
#     ) -> torch.Tensor:
#         loss, _, _ = self._compute_loss_details(
#             outputs,
#             targets,
#             elements=elements,
#             true_disp=true_disp,
#         )
#         return loss

#     # =====================================================
#     # Hの正解変位量別ロス集計
#     # 重み付け前の誤差を記録する。
#     # =====================================================
#     def _reset_loss_stats(self, stage: str):
#         num_bins = len(DISP_LOG_BINS_A) - 1

#         self._loss_stats[stage] = {
#             "counts": torch.zeros(
#                 num_bins,
#                 dtype=torch.int64,
#                 device=self.device,
#             ),
#             "sq_error_sums": torch.zeros(
#                 num_bins,
#                 dtype=torch.float64,
#                 device=self.device,
#             ),
#             "nonfinite_batches": 0,
#         }

#     @torch.no_grad()
#     def _accumulate_loss_stats(
#         self,
#         stage: str,
#         per_atom_sq_err: torch.Tensor,
#         elements: torch.Tensor,
#         true_disp: torch.Tensor,
#     ):
#         stats = self._loss_stats[stage]

#         err = per_atom_sq_err.detach()
#         elem = elements.detach()
#         disp = true_disp.detach()

#         if (
#             err.ndim != 1
#             or elem.shape != err.shape
#             or disp.shape != (err.shape[0], 3)
#         ):
#             raise ValueError(
#                 "集計用テンソルの形状が不正です。"
#                 f" error={tuple(err.shape)},"
#                 f" elements={tuple(elem.shape)},"
#                 f" true_disp={tuple(disp.shape)}"
#             )

#         mask = elem == TARGET_ELEM_ID

#         if not mask.any():
#             return

#         h_error = err[mask].to(dtype=torch.float64)

#         h_disp = disp[mask].to(
#             dtype=torch.float64
#         ).norm(dim=-1)

#         if (
#             not torch.isfinite(h_error).all()
#             or not torch.isfinite(h_disp).all()
#         ):
#             raise ValueError(
#                 "Hの診断集計に非有限値が含まれています。"
#             )

#         boundaries = torch.as_tensor(
#             DISP_LOG_BINS_A[1:-1],
#             dtype=torch.float64,
#             device=h_disp.device,
#         )

#         # r=1は[1, 3)、r=3は[3, inf)に入る。
#         bin_idx = torch.bucketize(
#             h_disp,
#             boundaries,
#             right=True,
#         )

#         num_bins = len(DISP_LOG_BINS_A) - 1

#         stats["counts"].add_(
#             torch.bincount(
#                 bin_idx,
#                 minlength=num_bins,
#             )
#         )

#         stats["sq_error_sums"].scatter_add_(
#             0,
#             bin_idx,
#             h_error,
#         )

#     @torch.no_grad()
#     def _write_loss_stats(self, stage: str):
#         if stage not in self._loss_stats:
#             return

#         stats = self._loss_stats[stage]

#         counts = stats["counts"].clone()
#         error_sums = stats["sq_error_sums"].clone()

#         nonfinite_batches = torch.tensor(
#             stats["nonfinite_batches"],
#             dtype=torch.int64,
#             device=self.device,
#         )

#         if (
#             torch.distributed.is_available()
#             and torch.distributed.is_initialized()
#         ):
#             torch.distributed.all_reduce(counts)
#             torch.distributed.all_reduce(error_sums)
#             torch.distributed.all_reduce(
#                 nonfinite_batches
#             )

#         if not self.trainer.is_global_zero:
#             return

#         counts = counts.cpu().tolist()
#         error_sums = error_sums.cpu().tolist()

#         total_count = sum(counts)
#         total_error_sum = sum(error_sums)

#         split = (
#             "validation" if stage == "valid"
#             else stage
#         )

#         use_active_weight = self._use_active_weight()
#         nonfinite_count = int(nonfinite_batches.item())

#         rows = []

#         for bin_id in range(len(counts)):
#             count = int(counts[bin_id])
#             error_sum = float(error_sums[bin_id])

#             rows.append({
#                 "epoch": int(self.current_epoch),
#                 "split": split,
#                 "prediction_mode":
#                     str(self.cfg.model.prediction_mode),
#                 "active_learning_weight":
#                     use_active_weight,
#                 "disp_min_A":
#                     float(DISP_LOG_BINS_A[bin_id]),
#                 "disp_max_A":
#                     float(DISP_LOG_BINS_A[bin_id + 1]),
#                 "count": count,
#                 "mean_sq_error": (
#                     error_sum / count
#                     if count > 0
#                     else float("nan")
#                 ),
#                 "sum_sq_error": error_sum,
#                 "H_error_fraction": (
#                     error_sum / total_error_sum
#                     if total_error_sum > 0
#                     else float("nan")
#                 ),
#                 "nonfinite_batches":
#                     nonfinite_count,
#             })

#         output_dir = self.cfg.trainer.get(
#             "output_dir", "."
#         )
#         if output_dir is None:
#             output_dir = "."

#         save_dir = os.path.join(
#             str(output_dir),
#             "loss_logs",
#         )
#         os.makedirs(save_dir, exist_ok=True)

#         filename = (
#             "loss_by_H_displacement_epoch_fixed_weighted_20261029.csv"
#             if use_active_weight
#             # else "loss_by_H_displacement_epoch_unweighted_normal_20261029.csv"
#             else "loss_by_H_displacement_epoch_unweighted_normal_20261102.csv"
#         )

#         csv_path = os.path.join(
#             save_dir,
#             filename,
#         )

#         fieldnames = [
#             "epoch",
#             "split",
#             "prediction_mode",
#             "active_learning_weight",
#             "disp_min_A",
#             "disp_max_A",
#             "count",
#             "mean_sq_error",
#             "sum_sq_error",
#             "H_error_fraction",
#             "nonfinite_batches",
#         ]

#         file_has_content = (
#             os.path.isfile(csv_path)
#             and os.path.getsize(csv_path) > 0
#         )

#         if file_has_content:
#             with open(
#                 csv_path,
#                 "r",
#                 newline="",
#                 encoding="utf-8",
#             ) as f:
#                 existing_header = next(
#                     csv.reader(f), None
#                 )

#             if existing_header != fieldnames:
#                 raise ValueError(
#                     "既存CSVの列構成が異なります。"
#                     "出力先を変更するか、"
#                     "既存CSVを退避してください: "
#                     f"{csv_path}"
#                 )

#         with open(
#             csv_path,
#             "a",
#             newline="",
#             encoding="utf-8",
#         ) as f:
#             writer = csv.DictWriter(
#                 f,
#                 fieldnames=fieldnames,
#             )

#             if not file_has_content:
#                 writer.writeheader()

#             writer.writerows(rows)

#         mean_h_error = (
#             total_error_sum / total_count
#             if total_count > 0
#             else float("nan")
#         )

#         print(
#             "[H displacement loss CSV] "
#             f"epoch={self.current_epoch}, "
#             f"split={split}, "
#             f"count={total_count}, "
#             f"mean_H_sq_error={mean_h_error:.6g}, "
#             f"nonfinite_batches={nonfinite_count}"
#         )

#     # =====================================================
#     # エポック開始・終了
#     # =====================================================
#     def on_train_epoch_start(self):
#         self._reset_loss_stats("train")

#     def on_train_epoch_end(self):
#         self._write_loss_stats("train")

#     def on_validation_epoch_start(self):
#         self._reset_loss_stats("valid")

#     def on_validation_epoch_end(self):
#         if self.trainer.sanity_checking:
#             return

#         self._write_loss_stats("valid")

#     # =====================================================
#     # Training
#     # =====================================================
#     def training_step(
#         self,
#         batch: Batch,
#         batch_idx: int,
#     ):
#         t = torch.rand(
#             (batch.num_graphs,),
#             device=self.device,
#             dtype=torch.float,
#         )

#         x_cond = batch["positions_1"].clone()

#         # priorを加える前に、重み判定用の正解変位を保存する。
#         disp = (
#             batch["positions_2"]
#             - batch["positions_1"]
#         ).clone()

#         batch["positions_1"] = (
#             batch["positions_1"] + batch["prior"]
#         )

#         _t = t[batch.batch][:, None]

#         x_t = (
#             (1.0 - _t) * batch["positions_1"]
#             + _t * batch["positions_2"]
#         )

#         dx_dt = (
#             batch["positions_2"]
#             - batch["positions_1"]
#         )

#         batch["positions_1"] = x_cond
#         batch["positions_2"] = x_t

#         outputs = self(batch, t_per_graph=t)

#         if self.cfg.model.prediction_mode == "velocity":
#             target = dx_dt
#         elif self.cfg.model.prediction_mode == "data":
#             target = disp
#         else:
#             raise ValueError(
#                 "Unknown prediction mode: "
#                 f"{self.cfg.model.prediction_mode}"
#             )

#         loss, per_atom_sq_err, _ = (
#             self._compute_loss_details(
#                 outputs,
#                 target,
#                 elements=batch["elements"],
#                 true_disp=disp,
#             )
#         )

#         if not torch.isfinite(loss):
#             self._loss_stats["train"][
#                 "nonfinite_batches"
#             ] += 1

#             print(
#                 f"[WARNING] Skipping batch {batch_idx} "
#                 f"(epoch {self.current_epoch}): "
#                 f"non-finite loss={loss.item()}"
#             )

#             # 既存のスキップ動作を維持。
#             # 主に単一GPU・自動最適化を想定。
#             return None

#         self._accumulate_loss_stats(
#             stage="train",
#             per_atom_sq_err=per_atom_sq_err,
#             elements=batch["elements"],
#             true_disp=disp,
#         )

#         self.log(
#             "train/loss",
#             loss,
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             batch_size=batch.num_graphs,
#         )

#         return loss

#     # =====================================================
#     # Optimizer
#     # =====================================================
#     def configure_optimizers(self):
#         if self.cfg.optimizer.class_name == "Adam":
#             optimizer_class = torch.optim.Adam
#         else:
#             raise ValueError(
#                 "Unknown optimizer class: "
#                 f"{self.cfg.optimizer.class_name}"
#             )

#         return optimizer_class(
#             self.model.parameters(),
#             lr=self.cfg.optimizer.lr,
#         )

#     # =====================================================
#     # Validation
#     # =====================================================
#     @torch.no_grad()
#     def validation_step(
#         self,
#         batch: Batch,
#         batch_idx: int,
#     ) -> torch.Tensor:
#         if self.cfg.valid.num_time_steps < 1:
#             raise ValueError(
#                 "valid.num_time_stepsは1以上にしてください。"
#             )

#         t_list = torch.linspace(
#             0.0,
#             1.0,
#             self.cfg.valid.num_time_steps,
#             device=self.device,
#             dtype=torch.float,
#         )

#         orig_batch = batch.clone()
#         loss_list = []

#         record_stats = not self.trainer.sanity_checking
#         batch_has_nonfinite = False

#         for t in t_list:
#             t_per_graph = (
#                 t.expand(batch.num_graphs).clone()
#             )

#             x_cond = (
#                 orig_batch["positions_1"].clone()
#             )

#             disp = (
#                 orig_batch["positions_2"]
#                 - orig_batch["positions_1"]
#             ).clone()

#             batch["positions_1"] = (
#                 orig_batch["positions_1"]
#                 + orig_batch["prior"]
#             ).clone()

#             batch["positions_2"] = (
#                 orig_batch["positions_2"].clone()
#             )

#             _t = t_per_graph[batch.batch][:, None]

#             x_t = (
#                 (1.0 - _t) * batch["positions_1"]
#                 + _t * batch["positions_2"]
#             )

#             dx_dt = (
#                 batch["positions_2"]
#                 - batch["positions_1"]
#             )

#             batch["positions_1"] = x_cond
#             batch["positions_2"] = x_t

#             outputs = self(
#                 batch,
#                 t_per_graph=t_per_graph,
#             )

#             if self.cfg.model.prediction_mode == "velocity":
#                 target = dx_dt
#             elif self.cfg.model.prediction_mode == "data":
#                 target = disp
#             else:
#                 raise ValueError(
#                     "Unknown prediction mode: "
#                     f"{self.cfg.model.prediction_mode}"
#                 )

#             loss, per_atom_sq_err, _ = (
#                 self._compute_loss_details(
#                     outputs,
#                     target,
#                     elements=batch["elements"],
#                     true_disp=disp,
#                 )
#             )

#             if not torch.isfinite(loss):
#                 batch_has_nonfinite = True

#                 print(
#                     "[WARNING] Non-finite validation loss: "
#                     f"epoch={self.current_epoch}, "
#                     f"batch={batch_idx}, "
#                     f"t={t.item():.4f}, "
#                     f"loss={loss.item()}"
#                 )

#             elif record_stats:
#                 self._accumulate_loss_stats(
#                     stage="valid",
#                     per_atom_sq_err=per_atom_sq_err,
#                     elements=batch["elements"],
#                     true_disp=disp,
#                 )

#             # 異常値はvalid/lossに残し、異常を隠さない。
#             loss_list.append(loss)

#         if record_stats and batch_has_nonfinite:
#             self._loss_stats["valid"][
#                 "nonfinite_batches"
#             ] += 1

#         loss = torch.stack(loss_list).mean()

#         self.log(
#             "valid/loss",
#             loss,
#             on_epoch=True,
#             batch_size=batch.num_graphs,
#         )

#         return loss




# # -*- coding: utf-8 -*-
# """LiFlow training with optional element-wise signed-log flow (plan A).

# Signed-log ON: physical coordinate inputs, transformed velocity outputs.
# Signed-log OFF: original physical-space flow. No fixed loss weighting.
# The inference integrator must separately support signed-log checkpoints.
# """
# import csv
# import os

# import lightning as L
# import torch
# from omegaconf import DictConfig, OmegaConf
# from torch_geometric.data import Batch

# from liflow.model.models import DualPaiNN
# from liflow.utils.transforms import SignedLogTransform


# TARGET_ELEM_ID = 1
# DISP_LOG_BINS_A = (0.0, 1.0, 3.0, float("inf"))


# class FlowModule(L.LightningModule):
#     def __init__(self, cfg: DictConfig, signed_log_parameters=None,
#                  signed_log_enabled=None):
#         super().__init__()
#         self.cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
#         section = self.cfg.get("signed_log", {})
#         requested = section.get("enabled", False)
#         if type(requested) is not bool:
#             raise ValueError("signed_log.enabled must be True or False")
#         if signed_log_enabled is not None and type(signed_log_enabled) is not bool:
#             raise ValueError("signed_log_enabled must be a bool")
#         # Explicit constructor argument is checkpoint-owned when restoring.
#         self.signed_log_enabled = (requested if signed_log_enabled is None
#                                    else signed_log_enabled)
#         if self.signed_log_enabled != requested:
#             raise ValueError("Checkpoint/config signed-log settings disagree")
#         if self.cfg.get("active_learning_weight", False):
#             raise ValueError("Fixed loss weighting was removed; remove active_learning_weight=True")
#         self._prediction_mode = str(self.cfg.model.prediction_mode)
#         self._validate_mode()

#         self.model = DualPaiNN(
#             num_features=self.cfg.model.num_features,
#             num_radial_basis=self.cfg.model.num_radial_basis,
#             num_layers=self.cfg.model.num_layers,
#             num_elements=self.cfg.model.num_elements,
#             r_max=self.cfg.model.r_max,
#             r_offset=self.cfg.model.r_offset,
#             ref_temp=self.cfg.model.ref_temp,
#         )
#         self.displacement_transform = None
#         payload = None
#         if self.signed_log_enabled:
#             if signed_log_parameters is None:
#                 path = section.get("parameters_path", None)
#                 if not path:
#                     raise ValueError("Set signed_log.parameters_path for a new signed-log model")
#                 self.displacement_transform = SignedLogTransform.from_json(path)
#             else:
#                 if OmegaConf.is_config(signed_log_parameters):
#                     signed_log_parameters = OmegaConf.to_container(
#                         signed_log_parameters, resolve=True)
#                 self.displacement_transform = SignedLogTransform(signed_log_parameters)
#             payload = self.displacement_transform.export_parameters()
#             ids = payload["elements"]
#             if max(x["element_id"] for x in ids) >= self.cfg.model.num_elements:
#                 raise ValueError("Transform ID exceeds cfg.model.num_elements")
#             delay = self.cfg.data.get("time_delay_steps", None)
#             if delay is not None and payload.get("lag_frames") != int(delay):
#                 raise ValueError("JSON lag_frames differs from data.time_delay_steps")

#         self.save_hyperparameters({
#             "cfg": self.cfg,
#             "signed_log_parameters": payload,
#             "signed_log_enabled": self.signed_log_enabled,
#         })
#         self._loss_stats = {}

#     @property
#     def error_space(self):
#         if self.signed_log_enabled:
#             return "transformed_velocity"
#         return ("physical_velocity" if self._prediction_mode == "velocity"
#                 else "physical_displacement")

#     def _validate_mode(self):
#         if self._prediction_mode not in ("velocity", "data"):
#             raise ValueError(f"Unknown prediction mode: {self._prediction_mode}")
#         if self.signed_log_enabled and self._prediction_mode != "velocity":
#             raise ValueError("Signed-log plan A currently supports velocity mode only")

#     def setup(self, stage: str):
#         pass

#     def on_fit_start(self):
#         # Fail if train.py replaced model.cfg with incompatible settings.
#         requested = self.cfg.get("signed_log", {}).get("enabled", False)
#         if type(requested) is not bool or requested != self.signed_log_enabled:
#             raise ValueError("model.cfg was changed incompatibly after model construction")
#         if str(self.cfg.model.prediction_mode) != self._prediction_mode:
#             raise ValueError("prediction_mode changed after model construction")
#         if self.cfg.get("active_learning_weight", False):
#             raise ValueError("Fixed loss weighting is not supported")
#         self.print(f"[Flow] signed_log={self.signed_log_enabled}, "
#                    f"prediction_mode={self._prediction_mode}, error_space={self.error_space}, "
#                    "loss=unweighted mean over ALL atoms")
#         if self.signed_log_enabled:
#             self.print("[Flow] plan A: physical coordinate input -> transformed velocity output")
#             self.print(f"[Flow] transform IDs={self.displacement_transform.species_by_id}")

#     def on_load_checkpoint(self, checkpoint):
#         # Protect against silently interpreting an old physical model as transformed.
#         hp = checkpoint.get("hyper_parameters", {})
#         saved = hp.get("signed_log_enabled", False)
#         if saved != self.signed_log_enabled:
#             raise ValueError("Checkpoint flow space differs from the constructed model")
#         if self.signed_log_enabled and not hp.get("signed_log_parameters"):
#             raise ValueError("Signed-log checkpoint lacks construction parameters")

#     def forward(self, batch: Batch, t_per_graph: torch.Tensor):
#         batch["time"] = t_per_graph[batch.batch]
#         return self.model(batch)

#     def _prepare_flow(self, original: Batch, t_per_graph: torch.Tensor):
#         # Clone to keep validation inputs independent of previous time evaluations.
#         batch = original.clone()
#         reference = original["positions_1"].clone()
#         endpoint = original["positions_2"]
#         disp = (endpoint - reference).clone()
#         prior = original["prior"]
#         t = t_per_graph[original.batch][:, None]
#         if self.signed_log_enabled:
#             elements = original["elements"]
#             z0 = self.displacement_transform.transform(prior, elements)
#             z1 = self.displacement_transform.transform(disp, elements)
#             zt = (1.0 - t) * z0 + t * z1
#             physical_disp = self.displacement_transform.inverse(zt, elements)
#             batch["positions_1"] = reference
#             batch["positions_2"] = reference + physical_disp
#             target = z1 - z0
#         else:
#             # Preserve the arithmetic order of the original physical-space code.
#             initial = reference + prior
#             batch["positions_1"] = reference
#             batch["positions_2"] = (1.0 - t) * initial + t * endpoint
#             target = endpoint - initial if self._prediction_mode == "velocity" else disp
#         return batch, target, disp

#     def _compute_loss_details(self, outputs, targets):
#         if outputs.shape != targets.shape or outputs.ndim != 2 or outputs.shape[1] != 3:
#             raise ValueError("Outputs and targets must both have shape [N, 3]")
#         per_atom_sq_err = (outputs - targets).pow(2).sum(-1)
#         return per_atom_sq_err.mean(), per_atom_sq_err

#     def compute_loss(self, outputs, targets):
#         return self._compute_loss_details(outputs, targets)[0]

#     def training_step(self, batch: Batch, batch_idx: int):
#         t = torch.rand((batch.num_graphs,), device=self.device, dtype=torch.float)
#         flow_batch, target, disp = self._prepare_flow(batch, t)
#         outputs = self(flow_batch, t_per_graph=t)
#         loss, errors = self._compute_loss_details(outputs, target)
#         if not torch.isfinite(loss):
#             # self._loss_stats["train"]["nonfinite_batches"] += 1
#             self.print(f"[WARNING] Skipping batch {batch_idx}, "
#                        f"epoch={self.current_epoch}, loss={loss.item()}")
#             # Existing single-GPU automatic-optimization behavior.
#             return None
#         #self._accumulate_loss_stats("train", errors, batch["elements"], disp)
#         self.log("train/loss", loss, on_step=True, on_epoch=True,
#                  prog_bar=True, batch_size=batch.num_graphs)
#         return loss

#     @torch.no_grad()
#     def validation_step(self, batch: Batch, batch_idx: int):
#         ntime = self.cfg.valid.num_time_steps
#         if ntime < 1:
#             raise ValueError("valid.num_time_steps must be >= 1")
#         t_list = torch.linspace(0.0, 1.0, ntime, device=self.device, dtype=torch.float)
#         losses = []
#         record = not self.trainer.sanity_checking
#         nonfinite = False
#         for t in t_list:
#             t_per_graph = t.expand(batch.num_graphs).clone()
#             flow_batch, target, disp = self._prepare_flow(batch, t_per_graph)
#             outputs = self(flow_batch, t_per_graph=t_per_graph)
#             loss, errors = self._compute_loss_details(outputs, target)
#             if not torch.isfinite(loss):
#                 nonfinite = True
#                 self.print(f"[WARNING] Nonfinite validation: epoch={self.current_epoch}, "
#                            f"batch={batch_idx}, t={t.item():.4f}, loss={loss.item()}")
#             # elif record:
#             #     self._accumulate_loss_stats("valid", errors, batch["elements"], disp)
#             losses.append(loss)
#         # if record and nonfinite:
#         #     self._loss_stats["valid"]["nonfinite_batches"] += 1
#         loss = torch.stack(losses).mean()
#         self.log("valid/loss", loss, on_epoch=True, batch_size=batch.num_graphs)
#         return loss

#     def configure_optimizers(self):
#         if self.cfg.optimizer.class_name != "Adam":
#             raise ValueError(f"Unknown optimizer: {self.cfg.optimizer.class_name}")
#         return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)

#     def _reset_loss_stats(self, stage):
#         self._loss_stats[stage] = {
#             "counts": torch.zeros(3, dtype=torch.int64, device=self.device),
#             "sq_error_sums": torch.zeros(3, dtype=torch.float64, device=self.device),
#             "nonfinite_batches": 0,
#         }

#     @torch.no_grad()
#     def _accumulate_loss_stats(self, stage, per_atom_sq_err, elements, true_disp):
#         err, elem, disp = per_atom_sq_err.detach(), elements.detach(), true_disp.detach()
#         if err.ndim != 1 or elem.shape != err.shape or disp.shape != (err.numel(), 3):
#             raise ValueError("Invalid diagnostic tensor shapes")
#         mask = elem == TARGET_ELEM_ID
#         if not mask.any():
#             return
#         selected = err[mask].to(torch.float64)
#         magnitude = disp[mask].to(torch.float64).norm(dim=-1)
#         if not torch.isfinite(selected).all() or not torch.isfinite(magnitude).all():
#             raise ValueError("Nonfinite target-element diagnostic values")
#         boundaries = torch.tensor([1.0, 3.0], dtype=torch.float64, device=self.device)
#         idx = torch.bucketize(magnitude, boundaries, right=True)
#         stats = self._loss_stats[stage]
#         stats["counts"].add_(torch.bincount(idx, minlength=3))
#         stats["sq_error_sums"].scatter_add_(0, idx, selected)

#     @torch.no_grad()
#     def _write_loss_stats(self, stage):
#         if stage not in self._loss_stats:
#             return
#         stats = self._loss_stats[stage]
#         counts = stats["counts"].clone()
#         sums = stats["sq_error_sums"].clone()
#         nonfinite = torch.tensor(stats["nonfinite_batches"], dtype=torch.int64,
#                                  device=self.device)
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             for tensor in (counts, sums, nonfinite):
#                 torch.distributed.all_reduce(tensor)
#         if not self.trainer.is_global_zero:
#             return
#         counts, sums = counts.cpu().tolist(), sums.cpu().tolist()
#         total_sum, total_count = sum(sums), sum(counts)
#         rows = []
#         for i, (count, error_sum) in enumerate(zip(counts, sums)):
#             rows.append({
#                 "epoch": int(self.current_epoch),
#                 "split": "validation" if stage == "valid" else stage,
#                 "prediction_mode": self._prediction_mode,
#                 "signed_log_enabled": self.signed_log_enabled,
#                 "error_space": self.error_space,
#                 "target_element_id": TARGET_ELEM_ID,
#                 "disp_min_A": DISP_LOG_BINS_A[i],
#                 "disp_max_A": DISP_LOG_BINS_A[i + 1],
#                 "count": count,
#                 "mean_sq_error": error_sum / count if count else float("nan"),
#                 "sum_sq_error": error_sum,
#                 "H_error_fraction": error_sum / total_sum if total_sum > 0 else float("nan"),
#                 "nonfinite_batches": int(nonfinite.item()),
#             })
#         output_dir = self.cfg.trainer.get("output_dir", ".") or "."
#         save_dir = os.path.join(str(output_dir), "loss_logs")
#         os.makedirs(save_dir, exist_ok=True)
#         filename = ("loss_by_H_displacement_epoch_signedlog_20261102.csv"
#                     if self.signed_log_enabled
#                     else "loss_by_H_displacement_epoch_physical_unweighted_20261102.csv")
#         path = os.path.join(save_dir, filename)
#         fields = list(rows[0])
#         has_content = os.path.isfile(path) and os.path.getsize(path) > 0
#         if has_content:
#             with open(path, newline="", encoding="utf-8") as f:
#                 if next(csv.reader(f), None) != fields:
#                     raise ValueError(f"Existing CSV header differs: {path}")
#         with open(path, "a", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fields)
#             if not has_content:
#                 writer.writeheader()
#             writer.writerows(rows)
#         mean = total_sum / total_count if total_count else float("nan")
#         self.print(f"[Displacement CSV] epoch={self.current_epoch}, stage={stage}, "
#                    f"space={self.error_space}, count={total_count}, mean={mean:.6g}")

#     def on_train_epoch_start(self):
#         self._reset_loss_stats("train")

#     def on_train_epoch_end(self):
#         # self._write_loss_stats("train")
#         pass

#     def on_validation_epoch_start(self):
#         self._reset_loss_stats("valid")

#     def on_validation_epoch_end(self):
#         # if not self.trainer.sanity_checking:
#         #     self._write_loss_stats("valid")
#         pass





# -*- coding: utf-8 -*-
"""LiFlow training with optional element-wise signed-log flow (plan A).

Signed-log ON: physical coordinate inputs, transformed velocity outputs.
Signed-log OFF: original physical-space flow. No fixed loss weighting.
The inference integrator must separately support signed-log checkpoints.
"""

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from liflow.model.models import DualPaiNN
from liflow.utils.transforms import SignedLogTransform


class FlowModule(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        signed_log_parameters=None,
        signed_log_enabled=None,
    ):
        super().__init__()

        self.cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)

        section = self.cfg.get("signed_log", {})
        requested = section.get("enabled", False)

        if type(requested) is not bool:
            raise ValueError("signed_log.enabled must be True or False")

        if (
            signed_log_enabled is not None
            and type(signed_log_enabled) is not bool
        ):
            raise ValueError("signed_log_enabled must be a bool")

        # Explicit constructor argument is checkpoint-owned when restoring.
        self.signed_log_enabled = (
            requested
            if signed_log_enabled is None
            else signed_log_enabled
        )

        if self.signed_log_enabled != requested:
            raise ValueError(
                "Checkpoint/config signed-log settings disagree"
            )

        if self.cfg.get("active_learning_weight", False):
            raise ValueError(
                "Fixed loss weighting was removed; "
                "remove active_learning_weight=True"
            )

        self._prediction_mode = str(self.cfg.model.prediction_mode)
        self._validate_mode()

        self.model = DualPaiNN(
            num_features=self.cfg.model.num_features,
            num_radial_basis=self.cfg.model.num_radial_basis,
            num_layers=self.cfg.model.num_layers,
            num_elements=self.cfg.model.num_elements,
            r_max=self.cfg.model.r_max,
            r_offset=self.cfg.model.r_offset,
            ref_temp=self.cfg.model.ref_temp,
        )

        self.displacement_transform = None
        payload = None

        if self.signed_log_enabled:
            if signed_log_parameters is None:
                path = section.get("parameters_path", None)

                if not path:
                    raise ValueError(
                        "Set signed_log.parameters_path "
                        "for a new signed-log model"
                    )

                self.displacement_transform = (
                    SignedLogTransform.from_json(path)
                )
            else:
                if OmegaConf.is_config(signed_log_parameters):
                    signed_log_parameters = OmegaConf.to_container(
                        signed_log_parameters,
                        resolve=True,
                    )

                self.displacement_transform = SignedLogTransform(
                    signed_log_parameters
                )

            payload = self.displacement_transform.export_parameters()
            ids = payload["elements"]

            if (
                max(x["element_id"] for x in ids)
                >= self.cfg.model.num_elements
            ):
                raise ValueError(
                    "Transform ID exceeds cfg.model.num_elements"
                )

            delay = self.cfg.data.get("time_delay_steps", None)

            if (
                delay is not None
                and payload.get("lag_frames") != int(delay)
            ):
                raise ValueError(
                    "JSON lag_frames differs from data.time_delay_steps"
                )

        self.save_hyperparameters({
            "cfg": self.cfg,
            "signed_log_parameters": payload,
            "signed_log_enabled": self.signed_log_enabled,
        })

    @property
    def error_space(self):
        if self.signed_log_enabled:
            return "transformed_velocity"

        return (
            "physical_velocity"
            if self._prediction_mode == "velocity"
            else "physical_displacement"
        )

    def _validate_mode(self):
        if self._prediction_mode not in ("velocity", "data"):
            raise ValueError(
                f"Unknown prediction mode: {self._prediction_mode}"
            )

        if (
            self.signed_log_enabled
            and self._prediction_mode != "velocity"
        ):
            raise ValueError(
                "Signed-log plan A currently supports velocity mode only"
            )

    def setup(self, stage: str):
        pass

    def on_fit_start(self):
        # Fail if train.py replaced model.cfg with incompatible settings.
        requested = self.cfg.get("signed_log", {}).get("enabled", False)

        if (
            type(requested) is not bool
            or requested != self.signed_log_enabled
        ):
            raise ValueError(
                "model.cfg was changed incompatibly "
                "after model construction"
            )

        if (
            str(self.cfg.model.prediction_mode)
            != self._prediction_mode
        ):
            raise ValueError(
                "prediction_mode changed after model construction"
            )

        if self.cfg.get("active_learning_weight", False):
            raise ValueError("Fixed loss weighting is not supported")

        self.print(
            f"[Flow] signed_log={self.signed_log_enabled}, "
            f"prediction_mode={self._prediction_mode}, "
            f"error_space={self.error_space}, "
            "loss=unweighted mean over ALL atoms"
        )

        if self.signed_log_enabled:
            self.print(
                "[Flow] plan A: physical coordinate input "
                "-> transformed velocity output"
            )
            self.print(
                "[Flow] transform IDs="
                f"{self.displacement_transform.species_by_id}"
            )

    def on_load_checkpoint(self, checkpoint):
        # Prevent interpreting a physical checkpoint as a transformed model.
        hp = checkpoint.get("hyper_parameters", {})
        saved = hp.get("signed_log_enabled", False)

        if saved != self.signed_log_enabled:
            raise ValueError(
                "Checkpoint flow space differs "
                "from the constructed model"
            )

        if (
            self.signed_log_enabled
            and not hp.get("signed_log_parameters")
        ):
            raise ValueError(
                "Signed-log checkpoint lacks construction parameters"
            )

    def forward(self, batch: Batch, t_per_graph: torch.Tensor):
        batch["time"] = t_per_graph[batch.batch]
        return self.model(batch)

    def _prepare_flow(
        self,
        original: Batch,
        t_per_graph: torch.Tensor,
    ):
        # Keep validation inputs independent of previous time evaluations.
        batch = original.clone()

        reference = original["positions_1"].clone()
        endpoint = original["positions_2"]
        disp = (endpoint - reference).clone()
        prior = original["prior"]

        t = t_per_graph[original.batch][:, None]

        if self.signed_log_enabled:
            elements = original["elements"]

            z0 = self.displacement_transform.transform(
                prior,
                elements,
            )
            z1 = self.displacement_transform.transform(
                disp,
                elements,
            )

            zt = (1.0 - t) * z0 + t * z1

            physical_disp = self.displacement_transform.inverse(
                zt,
                elements,
            )

            batch["positions_1"] = reference
            batch["positions_2"] = reference + physical_disp

            target = z1 - z0

        else:
            # Preserve the original physical-space arithmetic order.
            initial = reference + prior

            batch["positions_1"] = reference
            batch["positions_2"] = (
                (1.0 - t) * initial + t * endpoint
            )

            target = (
                endpoint - initial
                if self._prediction_mode == "velocity"
                else disp
            )

        return batch, target, disp

    def _compute_loss_details(self, outputs, targets):
        if (
            outputs.shape != targets.shape
            or outputs.ndim != 2
            or outputs.shape[1] != 3
        ):
            raise ValueError(
                "Outputs and targets must both have shape [N, 3]"
            )

        per_atom_sq_err = (
            (outputs - targets).pow(2).sum(-1)
        )

        return per_atom_sq_err.mean(), per_atom_sq_err

    def compute_loss(self, outputs, targets):
        return self._compute_loss_details(outputs, targets)[0]

    def training_step(self, batch: Batch, batch_idx: int):
        t = torch.rand(
            (batch.num_graphs,),
            device=self.device,
            dtype=torch.float,
        )

        flow_batch, target, _ = self._prepare_flow(batch, t)
        outputs = self(flow_batch, t_per_graph=t)
        loss = self.compute_loss(outputs, target)

        if not torch.isfinite(loss):
            self.print(
                f"[WARNING] Skipping batch {batch_idx}, "
                f"epoch={self.current_epoch}, "
                f"loss={loss.item()}"
            )
            return None

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int):
        ntime = self.cfg.valid.num_time_steps

        if ntime < 1:
            raise ValueError(
                "valid.num_time_steps must be >= 1"
            )

        t_list = torch.linspace(
            0.0,
            1.0,
            ntime,
            device=self.device,
            dtype=torch.float,
        )

        losses = []

        for t in t_list:
            t_per_graph = t.expand(batch.num_graphs).clone()

            flow_batch, target, _ = self._prepare_flow(
                batch,
                t_per_graph,
            )

            outputs = self(
                flow_batch,
                t_per_graph=t_per_graph,
            )

            loss = self.compute_loss(outputs, target)

            if not torch.isfinite(loss):
                self.print(
                    "[WARNING] Nonfinite validation: "
                    f"epoch={self.current_epoch}, "
                    f"batch={batch_idx}, "
                    f"t={t.item():.4f}, "
                    f"loss={loss.item()}"
                )

            losses.append(loss)

        loss = torch.stack(losses).mean()

        self.log(
            "valid/loss",
            loss,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        return loss

    def configure_optimizers(self):
        if self.cfg.optimizer.class_name != "Adam":
            raise ValueError(
                f"Unknown optimizer: "
                f"{self.cfg.optimizer.class_name}"
            )

        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
        )