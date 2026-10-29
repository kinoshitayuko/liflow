# import lightning as L
# import torch
# from omegaconf import DictConfig
# from torch_geometric.data import Batch

# from liflow.model.models import DualPaiNN

# # ★★★ ここに挿入 ★★★
# import os
# import pandas as pd
# import numpy as np


# # 20261006 コメントアウト
# # =========================================================
# # ★ アクティブラーニング（ヒストグラムベース）の重み配列 ★
# # =========================================================
# # 先ほど算出した配列をここにコピペします
# HIST_WEIGHTS_NP = np.array([
#     0.454, 0.531, 0.674, 0.800, 0.912, 0.948, 1.016, 1.132, 1.226, 1.218, 
#     1.174, 1.367, 1.426, 1.513, 1.750, 1.894, 2.011, 2.622, 2.717, 2.505, 
#     2.846, 3.044, 2.584, 3.163, 3.574, 2.899, 3.913, 3.914, 4.802, 3.715, 
#     3.538, 2.530, 2.251, 2.901, 6.570, 4.177, 5.753, 7.603, 10.000, 3.362, 
#     10.000, 10.000, 10.000, 4.402, 10.000, 1.901, 10.000, 0.100, 0.100, 0.100
# ])
# # ビンの境界（0〜15Åを50等分）
# HIST_BINS_NP = np.linspace(0, 15.0, 51)
# # =========================================================


# # # =========================================================
# # # ★ アクティブラーニング（成分別ヒストグラムベース）の重み配列 ★
# # # =========================================================
# # HIST_BINS_NP = np.linspace(-8.0, 8.0, 51)

# # HIST_WEIGHTS_X_NP = np.array([
# #     10.000, 10.000, 10.000, 10.000, 3.529, 10.000, 10.000, 10.000, 3.256, 10.000,
# #     10.000, 10.000, 7.103, 10.000, 4.600, 3.881, 5.751, 4.090, 2.703, 1.286,
# #     1.655, 1.148, 0.803, 0.618, 0.646, 0.572, 0.600, 0.768, 1.026, 0.998,
# #     1.508, 2.904, 5.265, 10.000, 8.308, 10.000, 7.183, 10.000, 10.000, 10.000,
# #     9.535, 9.019, 7.007, 10.000, 10.000, 10.000, 10.000, 10.000, 3.933, 10.000
# # ])

# # HIST_WEIGHTS_Y_NP = np.array([
# #     10.000, 10.000, 10.000, 10.000, 10.000, 10.000, 5.592, 6.394, 10.000, 10.000,
# #     10.000, 10.000, 3.063, 4.638, 4.991, 2.515, 2.954, 4.512, 1.704, 1.380,
# #     1.150, 1.057, 0.901, 0.817, 0.760, 0.706, 0.808, 0.920, 1.126, 1.222,
# #     1.283, 1.550, 2.992, 3.670, 7.737, 5.023, 5.056, 10.000, 3.628, 6.133,
# #     4.790, 3.748, 2.726, 10.000, 10.000, 10.000, 10.000, 10.000, 10.000, 10.000
# # ])

# # HIST_WEIGHTS_Z_NP = np.array([
# #     0.100, 10.000, 10.000, 10.000, 0.200, 1.061, 0.394, 10.000, 10.000, 10.000,
# #     10.000, 10.000, 2.646, 10.000, 2.423, 10.000, 4.360, 3.516, 2.836, 1.472,
# #     1.481, 1.231, 0.918, 0.841, 0.697, 0.841, 0.800, 0.910, 1.126, 1.650,
# #     1.457, 1.862, 1.883, 2.734, 8.631, 5.443, 2.705, 3.177, 4.372, 1.481,
# #     2.101, 1.681, 10.000, 10.000, 10.000, 10.000, 10.000, 10.000, 10.000, 0.100
# # ])
# # # =========================================================



# class FlowModule(L.LightningModule):
#     def __init__(self, cfg: DictConfig):
#         super().__init__()
#         self.cfg = cfg
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

#     def setup(self, stage: str):
#         pass

#     def forward(self, batch: Batch, t_per_graph: torch.Tensor) -> torch.Tensor:
#         batch["time"] = t_per_graph[batch.batch]
#         return self.model(batch)

#     # def compute_loss(
#     #     self, outputs: torch.Tensor, targets: torch.Tensor
#     # ) -> torch.Tensor:
#     #     # Sum over the xyz dimensions and average over the nodes
#     #     return (outputs - targets).pow(2).sum(-1).mean()
    
#     # # 20260721に変更 defaultのlossに戻す
#     # def compute_loss(
#     #     self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None
#     # ) -> torch.Tensor:
#     #     # Sum over the xyz dimensions and average over the nodes
#     #     return (outputs - targets).pow(2).sum(-1).mean()
    
    
#     # 20261026 原子種ごとのロスを記録して原因を調査
#     # 20261006 コメントアウト：変位ヒストグラムに基づいた重み付けを行うように変更
#     # 20260806 変更: ファインチューニング用の動的重み付け（Train/Validation共通）
#     def compute_loss(
#         self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None, true_disp: torch.Tensor = None
#     ) -> torch.Tensor:
#         # 1. 各原子の二乗誤差（この計算は微分のために必ず実行）
#         per_atom_sq_err = (outputs - targets).pow(2).sum(-1)

#         # 2. フラグのチェック
#         # cfgで active_learning_weight=True が指定されていれば、Train/Valid関係なく重みを掛ける
#         use_active_weight = self.cfg.get("active_learning_weight", False)
#         #print(use_active_weight)

#         if use_active_weight and elements is not None and true_disp is not None:
#             #print(f"[INFO] compute_loss: Applying histogram-based weights for target element.")
#             # メモリ消費を抑えるため、重み付けの計算履歴は残さない
#             with torch.no_grad():
#                 # 基本の重みはすべて 1.0 (等倍) でスタート
#                 weight = torch.ones_like(per_atom_sq_err)
                
#                 TARGET_ELEM_ID = 1  # He原子のID
#                 mask = (elements == TARGET_ELEM_ID)
                
#                 if mask.any():
#                     # ターゲット原子の「真の変位」の大きさを計算
#                     magnitudes = true_disp[mask].norm(dim=-1)

#                     # 重み配列の準備（targetsと同じデータ型に合わせる）
#                     device = targets.device
#                     dtype = targets.dtype
#                     ratio_weights = torch.tensor(HIST_WEIGHTS_NP, dtype=dtype, device=device)
#                     bins_tensor = torch.tensor(HIST_BINS_NP, dtype=dtype, device=device)

#                     # 変位がどのビンに入るか計算
#                     bin_idx = torch.bucketize(magnitudes, bins_tensor, right=False) - 1
#                     bin_idx = torch.clamp(bin_idx, min=0, max=len(ratio_weights) - 1)

#                     # ヒストグラムから取ってきた重みで上書きする
#                     weight[mask] = ratio_weights[bin_idx]
            
#             # 3a. 重み付き平均誤差を返す
#             return (per_atom_sq_err * weight).sum() / weight.sum()
            
#         else:
#             # 3b. 通常時（Phase 1学習時などフラグOFFの時）
#             # 重みを一切掛けず、純粋な平均二乗誤差を返す
#             return per_atom_sq_err.mean()
    
    
#     # # 20261006 コメントアウト：変位成分(x,y,z)に基づいた重み付けを行うように変更
#     # def compute_loss(
#     #     self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None, true_disp: torch.Tensor = None
#     # ) -> torch.Tensor:
#     #     # 1. 各原子・各成分(x, y, z)の二乗誤差: shape = [N, 3]
#     #     per_atom_comp_sq_err = (outputs - targets).pow(2)

#     #     # 2. フラグのチェック
#     #     use_active_weight = self.cfg.get("active_learning_weight", False)

#     #     if use_active_weight and elements is not None and true_disp is not None:
#     #         with torch.no_grad():
#     #             # 基本の重みはすべて 1.0 (等倍, shape: [N, 3])
#     #             weights = torch.ones_like(per_atom_comp_sq_err)

#     #             TARGET_ELEM_ID = 1  # ターゲット原子ID（HeやHなど）
#     #             mask = (elements == TARGET_ELEM_ID)

#     #             if mask.any():
#     #                 device = targets.device
#     #                 dtype = targets.dtype

#     #                 # 各成分の重みテーブルをテンソル化
#     #                 w_x = torch.tensor(HIST_WEIGHTS_X_NP, dtype=dtype, device=device)
#     #                 w_y = torch.tensor(HIST_WEIGHTS_Y_NP, dtype=dtype, device=device)
#     #                 w_z = torch.tensor(HIST_WEIGHTS_Z_NP, dtype=dtype, device=device)
#     #                 bins_tensor = torch.tensor(HIST_BINS_NP, dtype=dtype, device=device)

#     #                 # ターゲット原子の各成分変位: shape = [M, 3]
#     #                 target_disp = true_disp[mask]

#     #                 # 各軸のビンインデックスを計算
#     #                 # bucketize は単調増加配列に対してビン境界判定を行う
#     #                 idx_x = torch.clamp(torch.bucketize(target_disp[:, 0], bins_tensor, right=False) - 1, min=0, max=len(w_x) - 1)
#     #                 idx_y = torch.clamp(torch.bucketize(target_disp[:, 1], bins_tensor, right=False) - 1, min=0, max=len(w_y) - 1)
#     #                 idx_z = torch.clamp(torch.bucketize(target_disp[:, 2], bins_tensor, right=False) - 1, min=0, max=len(w_z) - 1)

#     #                 # ターゲット原子の各成分に重みを適用
#     #                 weights[mask, 0] = w_x[idx_x]
#     #                 weights[mask, 1] = w_y[idx_y]
#     #                 weights[mask, 2] = w_z[idx_z]

#     #         # 3a. 重み付き平均誤差（全原子・全成分の加重平均）
#     #         return (per_atom_comp_sq_err * weights).sum() / weights.sum()

#     #     else:
#     #         # 3b. 通常時（フラグOFFの時）
#     #         # 各原子の3成分二乗和の平均（元の per_atom_sq_err.mean() と同値）
#     #         return per_atom_comp_sq_err.sum(-1).mean()
    

#     def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        
#         # Sample flow time
#         t = torch.rand((batch.num_graphs,), device=self.device, dtype=torch.float)

#         # Interpolate
#         x_cond = batch["positions_1"].clone()
#         disp = (batch["positions_2"] - batch["positions_1"]).clone()
#         batch["positions_1"] = batch["positions_1"] + batch["prior"]

#         _t = t[batch.batch][:, None]
#         x_t = (1.0 - _t) * batch["positions_1"] + _t * batch["positions_2"]
#         dx_dt = batch["positions_2"] - batch["positions_1"]

#         # Compute the flow and loss
#         batch["positions_1"] = x_cond
#         batch["positions_2"] = x_t
#         outputs = self(batch, t_per_graph=t)
#         if self.cfg.model.prediction_mode == "velocity":
#             #loss = self.compute_loss(outputs, dx_dt)
#             #loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"])  # ★ elements追加
#             # ★ 変更: 第4引数に、真の変位である disp を渡す。こっちが使われる。
#             loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"], true_disp=disp)
#             target = dx_dt
#         elif self.cfg.model.prediction_mode == "data":
#             #loss = self.compute_loss(outputs, disp)
#             #loss = self.compute_loss(outputs, disp, elements=batch["elements"])  # ★ elements追加
#             # ★ 変更: 第4引数に、真の変位である disp を渡す。こっちは使われないが、念のため。
#             loss = self.compute_loss(outputs, disp, elements=batch["elements"], true_disp=disp)
#             target = disp
#         else:
#             raise ValueError(
#                 f"Unknown prediction mode: {self.cfg.model.prediction_mode}"
#             )
            
            
        
        
#         # ★★★ ここに追加 2026/08/20 ★★★
#         if not torch.isfinite(loss):
#             print(f"[WARNING] Skipping batch {batch_idx} (epoch {self.current_epoch}): "
#                 f"non-finite loss detected (loss={loss.item()})")
#             self.log(
#                 "train/skipped_batch",
#                 1.0,
#                 on_step=True,
#                 on_epoch=True,
#                 batch_size=batch.num_graphs,
#             )
#             return None  # ← このバッチの学習を丸ごとスキップ
#         # ★★★ ここまで ★★★
        
        
#         # # ===== 原子種ごとのロスをwandbに出力 =====
#         # per_node_loss = (outputs - target).pow(2).sum(-1)
#         # for i in range(self.cfg.model.num_elements):
#         #     mask = batch["elements"] == i
#         #     if mask.any():
#         #         self.log(
#         #             f"train/loss_elem{i:03d}",
#         #             per_node_loss[mask].mean(),
#         #             on_step=True,
#         #             on_epoch=True,
#         #             prog_bar=False,
#         #             batch_size=batch.num_graphs,
#         #         )
#         # # ==========================================
        
#         # Log the loss
#         self.log(
#             "train/loss",
#             loss,
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             batch_size=batch.num_graphs,
#         )
#         return loss

#     def configure_optimizers(self):
#         if self.cfg.optimizer.class_name == "Adam":
#             optimizer_class = torch.optim.Adam
#         else:
#             raise ValueError(
#                 f"Unknown optimizer class: {self.cfg.optimizer.class_name}"
#             )
#         return optimizer_class(self.model.parameters(), lr=self.cfg.optimizer.lr)

#     @torch.no_grad()
#     def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
#         # Preset the interpolation parameter for consistency
#         t_list = torch.linspace(
#             0.0,
#             1.0,
#             self.cfg.valid.num_time_steps,
#             device=self.device,
#             dtype=torch.float,
#         )
#         orig_batch = batch.clone()

#         # Compute the loss for each time step
#         loss_list = []
#         for t in t_list:
#             # Sample flow time
#             t_per_graph = t.expand(batch.num_graphs).clone()

#             # Interpolate
#             x_cond = orig_batch["positions_1"].clone()
#             disp = (orig_batch["positions_2"] - orig_batch["positions_1"]).clone()
#             batch["positions_1"] = (
#                 orig_batch["positions_1"] + orig_batch["prior"]
#             ).clone()
#             batch["positions_2"] = orig_batch["positions_2"].clone()
#             _t = t_per_graph[batch.batch][:, None]
#             x_t = (1.0 - _t) * batch["positions_1"] + _t * batch["positions_2"]
#             dx_dt = batch["positions_2"] - batch["positions_1"]

#             # Compute the flow and loss
#             batch["positions_1"] = x_cond
#             batch["positions_2"] = x_t
#             outputs = self(batch, t_per_graph=t_per_graph)
#             if self.cfg.model.prediction_mode == "velocity":
#                 #loss = self.compute_loss(outputs, dx_dt)
#                 #loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"])  # ★ elements追加
#                 # ★ 変更: 第4引数に、真の変位である disp を渡す。こっちが使われる。
#                 loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"], true_disp=disp)
#                 target = dx_dt          # ★ 追加: targetを明示的に保持
#             elif self.cfg.model.prediction_mode == "data":
#                 #loss = self.compute_loss(outputs, disp)
#                 #loss = self.compute_loss(outputs, disp, elements=batch["elements"])  # ★ elements追加
#                 # ★ 変更: 第4引数に、真の変位である disp を渡す。こっちは使われないが、念のため。
#                 loss = self.compute_loss(outputs, disp, elements=batch["elements"], true_disp=disp)
#                 target = disp            # ★ 追加
                
            
#             # # ★★★ ここに挿入 ★★★
#             # if batch_idx == 0:  # 検証は全ステップ保存すると重いので先頭バッチだけ等に絞る
#             #     out_np = outputs.detach().cpu().numpy()
#             #     tgt_np = target.detach().cpu().numpy()
#             #     elem_np = batch["elements"].detach().cpu().numpy()
                
                
#             #     # ★ 追加: elem=1だけを抜き出す
#             #     mask = elem_np == 1
#             #     out_np = out_np[mask]
#             #     tgt_np = tgt_np[mask]
#             #     elem_np = elem_np[mask]
                

#             #     out_norm = np.linalg.norm(out_np, axis=-1)
#             #     tgt_norm = np.linalg.norm(tgt_np, axis=-1)
#             #     cos_sim = (out_np * tgt_np).sum(-1) / (out_norm * tgt_norm + 1e-8)
#             #     sq_err = ((out_np - tgt_np) ** 2).sum(-1)

#             #     df = pd.DataFrame({
#             #         "epoch": self.current_epoch,
#             #         "t": t.item(),          # ★ どの補間時刻かも記録
#             #         "batch_idx": batch_idx,
#             #         "elem": elem_np,
#             #         "pred_vx": out_np[:, 0], "pred_vy": out_np[:, 1], "pred_vz": out_np[:, 2],
#             #         "true_vx": tgt_np[:, 0], "true_vy": tgt_np[:, 1], "true_vz": tgt_np[:, 2],
#             #         "pred_speed": out_norm,
#             #         "true_speed": tgt_norm,
#             #         "cos_sim": cos_sim,
#             #         "sq_err": sq_err,
#             #     })

#             #     save_dir = os.path.join(self.cfg.trainer.get("output_dir", "."), "velocity_logs")
#             #     os.makedirs(save_dir, exist_ok=True)
#             #     csv_path = os.path.join(save_dir, "velocity_field_validation_20260723_cor.csv")
#             #     df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
#             # # ★★★ ここまで ★★★
            
#             loss_list.append(loss)

#         loss = torch.stack(loss_list).mean()
#         self.log("valid/loss", loss, on_epoch=True, batch_size=batch.num_graphs)
#         return loss







# ここから変更する
# # 20261026 コメントアウト：上記のコードを整理して、重み計算を別関数に分離. csvファイルに記録する.
# import csv
# import os

# import lightning as L
# import numpy as np
# import torch
# from omegaconf import DictConfig
# from torch_geometric.data import Batch

# from liflow.model.models import DualPaiNN


# # =========================================================
# # ヒストグラムに基づく重み
# # =========================================================
# HIST_WEIGHTS_NP = np.array(
#     [
#         0.454, 0.531, 0.674, 0.800, 0.912,
#         0.948, 1.016, 1.132, 1.226, 1.218,
#         1.174, 1.367, 1.426, 1.513, 1.750,
#         1.894, 2.011, 2.622, 2.717, 2.505,
#         2.846, 3.044, 2.584, 3.163, 3.574,
#         2.899, 3.913, 3.914, 4.802, 3.715,
#         3.538, 2.530, 2.251, 2.901, 6.570,
#         4.177, 5.753, 7.603, 10.000, 3.362,
#         10.000, 10.000, 10.000, 4.402, 10.000,
#         1.901, 10.000, 0.100, 0.100, 0.100,
#     ],
#     dtype=float,
# )

# HIST_BINS_NP = np.linspace(0.0, 15.0, 51)

# TARGET_ELEM_ID = 1
# NUM_LOG_ELEMENTS = 5


# class FlowModule(L.LightningModule):
#     def __init__(self, cfg: DictConfig):
#         super().__init__()

#         self.cfg = cfg

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

#         # 診断用の集計値。モデルの学習パラメータには含めない。
#         self._loss_stats = {}

#     def setup(self, stage: str):
#         pass

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

#         use_active_weight = self.cfg.get(
#             "active_learning_weight", False
#         )

#         if (
#             not use_active_weight
#             or elements is None
#             or true_disp is None
#         ):
#             return weight

#         mask = elements == TARGET_ELEM_ID

#         if mask.any():
#             magnitudes = true_disp[mask].norm(dim=-1)

#             ratio_weights = torch.as_tensor(
#                 HIST_WEIGHTS_NP,
#                 dtype=per_atom_sq_err.dtype,
#                 device=per_atom_sq_err.device,
#             )

#             bins_tensor = torch.as_tensor(
#                 HIST_BINS_NP,
#                 dtype=per_atom_sq_err.dtype,
#                 device=per_atom_sq_err.device,
#             )

#             # 元のビン判定方法を維持
#             bin_idx = (
#                 torch.bucketize(
#                     magnitudes,
#                     bins_tensor,
#                     right=False,
#                 )
#                 - 1
#             )

#             bin_idx = torch.clamp(
#                 bin_idx,
#                 min=0,
#                 max=len(ratio_weights) - 1,
#             )

#             weight[mask] = ratio_weights[bin_idx]

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

#         use_active_weight = self.cfg.get(
#             "active_learning_weight", False
#         )

#         if (
#             use_active_weight
#             and elements is not None
#             and true_disp is not None
#         ):
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
#         # 従来の呼び出し形式も維持
#         loss, _, _ = self._compute_loss_details(
#             outputs,
#             targets,
#             elements=elements,
#             true_disp=true_disp,
#         )
#         return loss

#     # =====================================================
#     # エポック集計
#     # =====================================================
#     def _reset_loss_stats(self, stage: str):
#         self._loss_stats[stage] = {
#             "counts": torch.zeros(
#                 NUM_LOG_ELEMENTS,
#                 dtype=torch.int64,
#                 device=self.device,
#             ),
#             "sq_error_sums": torch.zeros(
#                 NUM_LOG_ELEMENTS,
#                 dtype=torch.float64,
#                 device=self.device,
#             ),
#             "weighted_error_sum": torch.zeros(
#                 (),
#                 dtype=torch.float64,
#                 device=self.device,
#             ),
#             "weight_sum": torch.zeros(
#                 (),
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
#         weight: torch.Tensor,
#         elements: torch.Tensor,
#     ):
#         stats = self._loss_stats[stage]

#         err = per_atom_sq_err.detach()
#         weights = weight.detach()
#         elem = elements.detach()

#         if err.ndim != 1 or elem.shape != err.shape:
#             raise ValueError(
#                 "elementsと原子ごとの誤差は、"
#                 "同じ形状の1次元テンソルである必要があります。"
#                 f" elements={tuple(elem.shape)},"
#                 f" error={tuple(err.shape)}"
#             )

#         if elem.dtype not in (
#             torch.int8,
#             torch.int16,
#             torch.int32,
#             torch.int64,
#             torch.uint8,
#         ):
#             raise ValueError(
#                 "elementsは整数の元素IDである必要があります。"
#             )

#         invalid = (elem < 0) | (elem >= NUM_LOG_ELEMENTS)

#         if invalid.any():
#             invalid_ids = (
#                 torch.unique(elem[invalid]).cpu().tolist()
#             )
#             raise ValueError(
#                 "CSV集計対象のID 0〜4以外が含まれています: "
#                 f"{invalid_ids}"
#             )

#         elem = elem.long()
#         err64 = err.to(dtype=torch.float64)
#         weights64 = weights.to(dtype=torch.float64)

#         stats["counts"].add_(
#             torch.bincount(
#                 elem,
#                 minlength=NUM_LOG_ELEMENTS,
#             )
#         )

#         stats["sq_error_sums"].scatter_add_(
#             0,
#             elem,
#             err64,
#         )

#         stats["weighted_error_sum"].add_(
#             (err64 * weights64).sum()
#         )
#         stats["weight_sum"].add_(weights64.sum())

#     def _write_loss_stats(self, stage: str):
#         if stage not in self._loss_stats:
#             return

#         stats = self._loss_stats[stage]

#         # CPUへの転送とCSV書き込みは、エポック末のみ。
#         counts = stats["counts"].cpu().tolist()
#         error_sums = stats["sq_error_sums"].cpu().tolist()

#         total_count = sum(counts)
#         total_error_sum = sum(error_sums)

#         weight_sum = stats["weight_sum"].item()
#         weighted_error_sum = (
#             stats["weighted_error_sum"].item()
#         )

#         raw_loss_total = (
#             total_error_sum / total_count
#             if total_count > 0
#             else float("nan")
#         )

#         objective_loss = (
#             weighted_error_sum / weight_sum
#             if weight_sum > 0
#             else float("nan")
#         )

#         row = {
#             "epoch": int(self.current_epoch),
#             "stage": stage,
#             "objective_loss": objective_loss,
#             "raw_loss_total": raw_loss_total,
#         }

#         for elem_id in range(NUM_LOG_ELEMENTS):
#             count = int(counts[elem_id])

#             row[f"elem{elem_id}_count"] = count
#             row[f"elem{elem_id}_loss"] = (
#                 error_sums[elem_id] / count
#                 if count > 0
#                 else float("nan")
#             )

#         row["nonfinite_batches"] = (
#             stats["nonfinite_batches"]
#         )

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

#         csv_path = os.path.join(
#             save_dir,
#             #"loss_by_element_epoch.csv",
#             "loss_by_element_epoch_finetune.csv",
#         )

#         fieldnames = list(row.keys())

#         file_has_content = (
#             os.path.isfile(csv_path)
#             and os.path.getsize(csv_path) > 0
#         )

#         # 異なる形式のCSVへ誤って追記しない。
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
#                     "出力先を変更するか、既存CSVを退避してください: "
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

#             writer.writerow(row)

#         print(
#             f"[Loss CSV] epoch={self.current_epoch}, "
#             f"stage={stage}, "
#             f"objective_loss={objective_loss:.6g}, "
#             f"raw_loss={raw_loss_total:.6g}, "
#             f"nonfinite_batches="
#             f"{stats['nonfinite_batches']}"
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

#         loss, per_atom_sq_err, weight = (
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

#             # 単一GPU・自動最適化で従来の動作を維持
#             return None

#         self._accumulate_loss_stats(
#             stage="train",
#             per_atom_sq_err=per_atom_sq_err,
#             weight=weight,
#             elements=batch["elements"],
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

#         # sanity checkは通常のCSV集計に含めない。
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

#             if (
#                 self.cfg.model.prediction_mode
#                 == "velocity"
#             ):
#                 target = dx_dt
#             elif (
#                 self.cfg.model.prediction_mode
#                 == "data"
#             ):
#                 target = disp
#             else:
#                 raise ValueError(
#                     "Unknown prediction mode: "
#                     f"{self.cfg.model.prediction_mode}"
#                 )

#             loss, per_atom_sq_err, weight = (
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
#                     weight=weight,
#                     elements=batch["elements"],
#                 )

#             # 非有限値も残し、valid/lossで異常を隠さない。
#             loss_list.append(loss)

#         if record_stats and batch_has_nonfinite:
#             # 同一バッチで複数時刻が異常でも、1バッチと数える。
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










# 重み付け前の誤差を集計する
import csv
import os

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch

from liflow.model.models import DualPaiNN


# =========================================================
# 学習用：ヒストグラムに基づく重み
# 既存の設定を維持
# =========================================================
HIST_WEIGHTS_NP = np.array(
    [
        0.454, 0.531, 0.674, 0.800, 0.912,
        0.948, 1.016, 1.132, 1.226, 1.218,
        1.174, 1.367, 1.426, 1.513, 1.750,
        1.894, 2.011, 2.622, 2.717, 2.505,
        2.846, 3.044, 2.584, 3.163, 3.574,
        2.899, 3.913, 3.914, 4.802, 3.715,
        3.538, 2.530, 2.251, 2.901, 6.570,
        4.177, 5.753, 7.603, 10.000, 3.362,
        10.000, 10.000, 10.000, 4.402, 10.000,
        1.901, 10.000, 0.100, 0.100, 0.100,
    ],
    dtype=float,
)

HIST_BINS_NP = np.linspace(0.0, 15.0, 51)

TARGET_ELEM_ID = 1


# =========================================================
# 診断用：Hの正解変位量ビン［Å］
# [0, 1), [1, 3), [3, inf)
# 学習用のHIST_BINS_NPとは別
# =========================================================
DISP_LOG_BINS_A = np.array(
    [0.0, 1.0, 3.0, np.inf],
    dtype=float,
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

        # 診断用。モデルパラメータや損失計算には使用しない。
        self._loss_stats = {}

    def setup(self, stage: str):
        pass

    def forward(
        self,
        batch: Batch,
        t_per_graph: torch.Tensor,
    ) -> torch.Tensor:
        batch["time"] = t_per_graph[batch.batch]
        return self.model(batch)

    # =====================================================
    # 損失計算：既存の動作を維持
    # =====================================================
    @torch.no_grad()
    def _get_loss_weights(
        self,
        per_atom_sq_err: torch.Tensor,
        elements: torch.Tensor = None,
        true_disp: torch.Tensor = None,
    ) -> torch.Tensor:
        weight = torch.ones_like(per_atom_sq_err)

        use_active_weight = self.cfg.get(
            "active_learning_weight", False
        )

        if (
            not use_active_weight
            or elements is None
            or true_disp is None
        ):
            return weight

        mask = elements == TARGET_ELEM_ID

        if mask.any():
            magnitudes = true_disp[mask].norm(dim=-1)

            ratio_weights = torch.as_tensor(
                HIST_WEIGHTS_NP,
                dtype=per_atom_sq_err.dtype,
                device=per_atom_sq_err.device,
            )

            bins_tensor = torch.as_tensor(
                HIST_BINS_NP,
                dtype=per_atom_sq_err.dtype,
                device=per_atom_sq_err.device,
            )

            # 学習用のビン判定は既存の方法を維持する。
            bin_idx = (
                torch.bucketize(
                    magnitudes,
                    bins_tensor,
                    right=False,
                )
                - 1
            )

            bin_idx = torch.clamp(
                bin_idx,
                min=0,
                max=len(ratio_weights) - 1,
            )

            weight[mask] = ratio_weights[bin_idx]

        return weight

    def _compute_loss_details(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        elements: torch.Tensor = None,
        true_disp: torch.Tensor = None,
    ):
        per_atom_sq_err = (
            (outputs - targets).pow(2).sum(-1)
        )

        weight = self._get_loss_weights(
            per_atom_sq_err,
            elements=elements,
            true_disp=true_disp,
        )

        use_active_weight = self.cfg.get(
            "active_learning_weight", False
        )

        if (
            use_active_weight
            and elements is not None
            and true_disp is not None
        ):
            loss = (
                (per_atom_sq_err * weight).sum()
                / weight.sum()
            )
        else:
            loss = per_atom_sq_err.mean()

        return loss, per_atom_sq_err, weight

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        elements: torch.Tensor = None,
        true_disp: torch.Tensor = None,
    ) -> torch.Tensor:
        loss, _, _ = self._compute_loss_details(
            outputs,
            targets,
            elements=elements,
            true_disp=true_disp,
        )
        return loss

    # =====================================================
    # Hの正解変位量別ロス集計
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
                "集計用テンソルの形状が不正です。"
                f" error={tuple(err.shape)},"
                f" elements={tuple(elem.shape)},"
                f" true_disp={tuple(disp.shape)}"
            )

        mask = elem == TARGET_ELEM_ID

        if not mask.any():
            return

        # 重みを掛ける前の誤差を記録する。
        h_error = err[mask].to(dtype=torch.float64)

        # true_dispがÅ単位であることを前提とする。
        h_disp = disp[mask].to(
            dtype=torch.float64
        ).norm(dim=-1)

        if (
            not torch.isfinite(h_error).all()
            or not torch.isfinite(h_disp).all()
        ):
            raise ValueError(
                "Hの診断集計に非有限値が含まれています。"
            )

        boundaries = torch.as_tensor(
            DISP_LOG_BINS_A[1:-1],
            dtype=torch.float64,
            device=h_disp.device,
        )

        # 診断用は下限以上・上限未満。
        # r=1は[1, 3)、r=3は[3, inf)に入る。
        bin_idx = torch.bucketize(
            h_disp,
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
            h_error,
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
            torch.distributed.all_reduce(
                nonfinite_batches
            )

        if not self.trainer.is_global_zero:
            return

        counts = counts.cpu().tolist()
        error_sums = error_sums.cpu().tolist()

        total_count = sum(counts)
        total_error_sum = sum(error_sums)

        split = (
            "validation" if stage == "valid"
            else stage
        )

        use_active_weight = bool(
            self.cfg.get(
                "active_learning_weight", False
            )
        )

        rows = []

        for bin_id in range(len(counts)):
            count = int(counts[bin_id])
            error_sum = float(error_sums[bin_id])

            rows.append({
                "epoch": int(self.current_epoch),
                "split": split,
                "prediction_mode":
                    str(self.cfg.model.prediction_mode),
                "active_learning_weight":
                    use_active_weight,
                "disp_min_A":
                    float(DISP_LOG_BINS_A[bin_id]),
                "disp_max_A":
                    float(DISP_LOG_BINS_A[bin_id + 1]),
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
                "nonfinite_batches":
                    int(nonfinite_batches.item()),
            })

        output_dir = self.cfg.trainer.get(
            "output_dir", "."
        )
        if output_dir is None:
            output_dir = "."

        save_dir = os.path.join(
            str(output_dir),
            "loss_logs",
        )
        os.makedirs(save_dir, exist_ok=True)

        filename = (
            "loss_by_H_displacement_epoch_weighted.csv"
            if use_active_weight
            else "loss_by_H_displacement_epoch_unweighted_addtraining_20261027_2.csv"
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
                    csv.reader(f), None
                )

            if existing_header != fieldnames:
                raise ValueError(
                    "既存CSVの列構成が異なります。"
                    "出力先を変更するか、"
                    "既存CSVを退避してください: "
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

        mean_h_error = (
            total_error_sum / total_count
            if total_count > 0
            else float("nan")
        )

        print(
            f"[H displacement loss CSV] "
            f"epoch={self.current_epoch}, "
            f"split={split}, "
            f"count={total_count}, "
            f"mean_H_sq_error={mean_h_error:.6g}, "
            f"nonfinite_batches="
            f"{int(nonfinite_batches.item())}"
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
        t = torch.rand(
            (batch.num_graphs,),
            device=self.device,
            dtype=torch.float,
        )

        x_cond = batch["positions_1"].clone()

        disp = (
            batch["positions_2"]
            - batch["positions_1"]
        ).clone()

        batch["positions_1"] = (
            batch["positions_1"] + batch["prior"]
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

        outputs = self(batch, t_per_graph=t)

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
                outputs,
                target,
                elements=batch["elements"],
                true_disp=disp,
            )
        )

        if not torch.isfinite(loss):
            self._loss_stats["train"][
                "nonfinite_batches"
            ] += 1

            print(
                f"[WARNING] Skipping batch {batch_idx} "
                f"(epoch {self.current_epoch}): "
                f"non-finite loss={loss.item()}"
            )

            # 既存のスキップ動作を維持。
            # 主に単一GPU・自動最適化を想定。
            return None

        self._accumulate_loss_stats(
            stage="train",
            per_atom_sq_err=per_atom_sq_err,
            elements=batch["elements"],
            true_disp=disp,
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        return loss

    # =====================================================
    # Optimizer
    # =====================================================
    def configure_optimizers(self):
        if self.cfg.optimizer.class_name == "Adam":
            optimizer_class = torch.optim.Adam
        else:
            raise ValueError(
                "Unknown optimizer class: "
                f"{self.cfg.optimizer.class_name}"
            )

        return optimizer_class(
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
        if self.cfg.valid.num_time_steps < 1:
            raise ValueError(
                "valid.num_time_stepsは1以上にしてください。"
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

            if (
                self.cfg.model.prediction_mode
                == "velocity"
            ):
                target = dx_dt
            elif (
                self.cfg.model.prediction_mode
                == "data"
            ):
                target = disp
            else:
                raise ValueError(
                    "Unknown prediction mode: "
                    f"{self.cfg.model.prediction_mode}"
                )

            loss, per_atom_sq_err, _ = (
                self._compute_loss_details(
                    outputs,
                    target,
                    elements=batch["elements"],
                    true_disp=disp,
                )
            )

            if not torch.isfinite(loss):
                batch_has_nonfinite = True

                print(
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

            # 既存どおり、異常値はvalid/lossには残す。
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