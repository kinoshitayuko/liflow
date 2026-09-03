import lightning as L
import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch

from liflow.model.models import DualPaiNN

# ★★★ ここに挿入 ★★★
import os
import pandas as pd
import numpy as np



# =========================================================
# ★ アクティブラーニング（ヒストグラムベース）の重み配列 ★
# =========================================================
# 先ほど算出した配列をここにコピペします
HIST_WEIGHTS_NP = np.array([
    0.454, 0.531, 0.674, 0.800, 0.912, 0.948, 1.016, 1.132, 1.226, 1.218, 
    1.174, 1.367, 1.426, 1.513, 1.750, 1.894, 2.011, 2.622, 2.717, 2.505, 
    2.846, 3.044, 2.584, 3.163, 3.574, 2.899, 3.913, 3.914, 4.802, 3.715, 
    3.538, 2.530, 2.251, 2.901, 6.570, 4.177, 5.753, 7.603, 10.000, 3.362, 
    10.000, 10.000, 10.000, 4.402, 10.000, 1.901, 10.000, 0.100, 0.100, 0.100
])
# ビンの境界（0〜15Åを50等分）
HIST_BINS_NP = np.linspace(0, 15.0, 51)
# =========================================================



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

    def setup(self, stage: str):
        pass

    def forward(self, batch: Batch, t_per_graph: torch.Tensor) -> torch.Tensor:
        batch["time"] = t_per_graph[batch.batch]
        return self.model(batch)

    # def compute_loss(
    #     self, outputs: torch.Tensor, targets: torch.Tensor
    # ) -> torch.Tensor:
    #     # Sum over the xyz dimensions and average over the nodes
    #     return (outputs - targets).pow(2).sum(-1).mean()
    
    # # 20260721に変更 defaultのlossに戻す
    # def compute_loss(
    #     self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None
    # ) -> torch.Tensor:
    #     # Sum over the xyz dimensions and average over the nodes
    #     return (outputs - targets).pow(2).sum(-1).mean()
    
    # # 20260805 変更: 原子種ごとの重み付きロス + 真の変位(true_disp)による動的重み付け
    # def compute_loss(
    #     self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None, true_disp: torch.Tensor = None
    # ) -> torch.Tensor:
    #     # 1. 各原子の二乗誤差
    #     per_atom_sq_err = (outputs - targets).pow(2).sum(-1)

    #     # 2. 基本の重みはすべて 1.0 (等倍) でスタート
    #     weight = torch.ones_like(per_atom_sq_err)

    #     # elements と true_disp が両方ある場合のみ、特定の原子種にヒストグラム重みを適用
    #     if elements is not None and true_disp is not None:
    #         #print(f"[INFO] compute_loss: Applying histogram-based weights for target element.")
    #         # ----------------------------------------------------
    #         # ★ ここをご自身の環境のターゲット原子のIDに合わせる ★
    #         TARGET_ELEM_ID = 1  # 例: 1 が He 原子の場合
    #         # ----------------------------------------------------
            
    #         # ターゲット原子のインデックス（True/False のマスク）を取得
    #         mask = (elements == TARGET_ELEM_ID)
            
    #         if mask.any():
    #             # ターゲット原子の「真の変位」の大きさを計算
    #             magnitudes = true_disp[mask].norm(dim=-1)

    #             # 重み配列の準備
    #             device = targets.device
    #             ratio_weights = torch.tensor(HIST_WEIGHTS_NP, dtype=torch.float32, device=device)
    #             bins_tensor = torch.tensor(HIST_BINS_NP, dtype=torch.float32, device=device)

    #             # 変位がどのビンに入るか計算
    #             bin_idx = torch.bucketize(magnitudes, bins_tensor, right=False) - 1
    #             bin_idx = torch.clamp(bin_idx, min=0, max=len(ratio_weights) - 1)

    #             # ターゲット原子の該当箇所の重みを、ヒストグラムから取ってきた値で上書きする
    #             weight[mask] = ratio_weights[bin_idx]

    #     # 3. 重み付き平均誤差を返す
    #     return (per_atom_sq_err * weight).sum() / weight.sum()
    
    
    # 20260806 変更: ファインチューニング用の動的重み付け（Train/Validation共通）
    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None, true_disp: torch.Tensor = None
    ) -> torch.Tensor:
        # 1. 各原子の二乗誤差（この計算は微分のために必ず実行）
        per_atom_sq_err = (outputs - targets).pow(2).sum(-1)

        # 2. フラグのチェック
        # cfgで active_learning_weight=True が指定されていれば、Train/Valid関係なく重みを掛ける
        use_active_weight = self.cfg.get("active_learning_weight", False)
        #print(use_active_weight)

        if use_active_weight and elements is not None and true_disp is not None:
            #print(f"[INFO] compute_loss: Applying histogram-based weights for target element.")
            # メモリ消費を抑えるため、重み付けの計算履歴は残さない
            with torch.no_grad():
                # 基本の重みはすべて 1.0 (等倍) でスタート
                weight = torch.ones_like(per_atom_sq_err)
                
                TARGET_ELEM_ID = 1  # He原子のID
                mask = (elements == TARGET_ELEM_ID)
                
                if mask.any():
                    # ターゲット原子の「真の変位」の大きさを計算
                    magnitudes = true_disp[mask].norm(dim=-1)

                    # 重み配列の準備（targetsと同じデータ型に合わせる）
                    device = targets.device
                    dtype = targets.dtype
                    ratio_weights = torch.tensor(HIST_WEIGHTS_NP, dtype=dtype, device=device)
                    bins_tensor = torch.tensor(HIST_BINS_NP, dtype=dtype, device=device)

                    # 変位がどのビンに入るか計算
                    bin_idx = torch.bucketize(magnitudes, bins_tensor, right=False) - 1
                    bin_idx = torch.clamp(bin_idx, min=0, max=len(ratio_weights) - 1)

                    # ヒストグラムから取ってきた重みで上書きする
                    weight[mask] = ratio_weights[bin_idx]
            
            # 3a. 重み付き平均誤差を返す
            return (per_atom_sq_err * weight).sum() / weight.sum()
            
        else:
            # 3b. 通常時（Phase 1学習時などフラグOFFの時）
            # 重みを一切掛けず、純粋な平均二乗誤差を返す
            return per_atom_sq_err.mean()
    
    
    
    
    
    
    
    
    
    
    
    
    # # 20260716 追加: 原子種ごとの重み付きロス
    # def compute_loss(
    #     self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None
    # ) -> torch.Tensor:
    #     # Sum over the xyz dimensions and average over the nodes
    #     per_atom_sq_err = (outputs - targets).pow(2).sum(-1)

    #     if elements is None:
    #         return per_atom_sq_err.mean()

    #     # ★ 拡散原子(elem=1)を重視する固定重み
    #     weights = torch.ones_like(per_atom_sq_err)
    #     weights[elements == 1] = 165.0 / 2.0  # まずは10倍から試す

    #     return (per_atom_sq_err * weights).sum() / weights.sum()
    
    
    # # 20260716 追加: 二乗誤差→四乗誤差に変更
    # def compute_loss(
    #     self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None
    # ) -> torch.Tensor:
    #     # Sum over the xyz dimensions and average over the nodes
    #     return (outputs - targets).pow(4).sum(-1).mean()
    
    
    # # 20260717 追加: 原子種ごとの重み付きロス
    # # 20260717 追加: 動き不足(speed_deficit)による動的重み付けを追加
    # def compute_loss(
    #     self, outputs: torch.Tensor, targets: torch.Tensor, elements: torch.Tensor = None
    # ) -> torch.Tensor:
    #     # Sum over the xyz dimensions and average over the nodes
    #     per_atom_sq_err = (outputs - targets).pow(2).sum(-1)

    #     # ★ 動き不足による動的重み（γ=1固定、ハイパーパラメータなし）
    #     true_speed = targets.norm(dim=-1)
    #     pred_speed = outputs.norm(dim=-1)
    #     speed_deficit = torch.clamp(true_speed - pred_speed, min=0.0)
    #     weight = 1.0 + speed_deficit

    #     if elements is not None:
    #         # ★ 拡散原子(elem=1)を重視する固定重み（既存の重みに乗算）
    #         elem_weight = torch.ones_like(per_atom_sq_err)
    #         elem_weight[elements == 1] = 165.0/2.0
    #         weight = weight * elem_weight

    #     return (per_atom_sq_err * weight).sum() / weight.sum()
    

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        
        ## 20260626 追加: batch の中身をログ出力
        # if batch_idx == 0:
        #     print("=== batch keys ===")
        #     print(batch.keys())          # キー一覧
        #     print("=== 最初の10原子 ===")
        #     for key in batch.keys():
        #         val = batch[key]
        #         if hasattr(val, 'shape'):
        #             print(f"{key}: shape={val.shape}, dtype={val.dtype}")
        #             print("elements sample:", batch["elements"][:20])
        
        # print("elements unique:", batch["elements"].unique())
        # print("elements all:", batch["elements"])
        
        # if batch_idx == 0:
        #     print("elements unique:", batch["elements"].unique())
        #     print("各インデックスの原子数:")
        #     for i in range(5):
        #         count = (batch["elements"] == i).sum().item()
        #         print(f"  elem{i:03d}: {count}個")
        
        
        # Sample flow time
        t = torch.rand((batch.num_graphs,), device=self.device, dtype=torch.float)

        # Interpolate
        x_cond = batch["positions_1"].clone()
        disp = (batch["positions_2"] - batch["positions_1"]).clone()
        batch["positions_1"] = batch["positions_1"] + batch["prior"]

        _t = t[batch.batch][:, None]
        x_t = (1.0 - _t) * batch["positions_1"] + _t * batch["positions_2"]
        dx_dt = batch["positions_2"] - batch["positions_1"]

        # Compute the flow and loss
        batch["positions_1"] = x_cond
        batch["positions_2"] = x_t
        outputs = self(batch, t_per_graph=t)
        if self.cfg.model.prediction_mode == "velocity":
            #loss = self.compute_loss(outputs, dx_dt)
            #loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"])  # ★ elements追加
            # ★ 変更: 第4引数に、真の変位である disp を渡す
            loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"], true_disp=disp)
            target = dx_dt
        elif self.cfg.model.prediction_mode == "data":
            #loss = self.compute_loss(outputs, disp)
            #loss = self.compute_loss(outputs, disp, elements=batch["elements"])  # ★ elements追加
            # ★ 変更: 第4引数に、真の変位である disp を渡す
            loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"], true_disp=disp)
            target = disp
        else:
            raise ValueError(
                f"Unknown prediction mode: {self.cfg.model.prediction_mode}"
            )
            
            
        
        
        # ★★★ ここに追加 2026/08/20 ★★★
        if not torch.isfinite(loss):
            print(f"[WARNING] Skipping batch {batch_idx} (epoch {self.current_epoch}): "
                f"non-finite loss detected (loss={loss.item()})")
            self.log(
                "train/skipped_batch",
                1.0,
                on_step=True,
                on_epoch=True,
                batch_size=batch.num_graphs,
            )
            return None  # ← このバッチの学習を丸ごとスキップ
        # ★★★ ここまで ★★★
        
        
        
        # # ===== 原子種ごとのロスをログ出力 =====
        # per_node_loss = (outputs - target).pow(2).sum(-1)
        # for elem_idx in batch["elements"].unique():
        #     mask = batch["elements"] == elem_idx
        #     self.log(f"train/loss_elem{elem_idx.item():03d}",
        #              per_node_loss[mask].mean(),
        #              on_step=True, on_epoch=True, prog_bar=False,
        #              batch_size=batch.num_graphs)
        # # ========================================
        
        
        # if batch_idx == 0:
        #     per_node_loss = (outputs - target).pow(2).sum(-1)
        #     for i in range(5):
        #         mask = batch["elements"] == i
        #         if mask.any():
        #             print(f"elem{i:03d}: loss={per_node_loss[mask].mean().item():.6f}, 原子数={mask.sum().item()}")
        
        
        # ===== 原子種ごとのロスをwandbに出力 =====
        per_node_loss = (outputs - target).pow(2).sum(-1)
        for i in range(self.cfg.model.num_elements):
            mask = batch["elements"] == i
            if mask.any():
                self.log(
                    f"train/loss_elem{i:03d}",
                    per_node_loss[mask].mean(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch.num_graphs,
                )
        # ==========================================
        
        
        
        # # ===== 原子種ごとのロスをwandbに出力 =====
        # per_node_loss = (outputs - target).pow(2).sum(-1)
        # elem_weight = torch.where(batch["elements"] == 1, 165.0 / 2.0, 1.0)  # ★追加1行

        # for i in range(self.cfg.model.num_elements):
        #     mask = batch["elements"] == i
        #     if mask.any():
        #         self.log(
        #             f"train/loss_elem{i:03d}",
        #             per_node_loss[mask].mean(),
        #             on_step=True,
        #             on_epoch=True,
        #             prog_bar=False,
        #             batch_size=batch.num_graphs,
        #         )
        #         self.log(
        #             f"train/loss_elem{i:03d}_weighted",  # ★追加
        #             (per_node_loss[mask] * elem_weight[mask]).mean(),
        #             on_step=True,
        #             on_epoch=True,
        #             prog_bar=False,
        #             batch_size=batch.num_graphs,
        #         )
        # # ==========================================
        
        
        # # ★★★ 2026/7/16 ここに挿入 ★★★
        # if batch_idx % self.cfg.trainer.get("log_velocity_every_n_steps", 50) == 0:
        #     with torch.no_grad():
        #         out_np = outputs.detach().cpu().numpy()
        #         tgt_np = target.detach().cpu().numpy()
        #         elem_np = batch["elements"].detach().cpu().numpy()

                
        #         # ★ 2026/7/17 追加: elem=1だけを抜き出す
        #         mask = elem_np == 1
        #         out_np = out_np[mask]
        #         tgt_np = tgt_np[mask]
        #         elem_np = elem_np[mask]
                
                
        #         out_norm = np.linalg.norm(out_np, axis=-1)
        #         tgt_norm = np.linalg.norm(tgt_np, axis=-1)
        #         cos_sim = (out_np * tgt_np).sum(-1) / (out_norm * tgt_norm + 1e-8)
        #         sq_err = ((out_np - tgt_np) ** 2).sum(-1)

        #         df = pd.DataFrame({
        #             "epoch": self.current_epoch,
        #             "batch_idx": batch_idx,
        #             "elem": elem_np,
        #             "pred_vx": out_np[:, 0], "pred_vy": out_np[:, 1], "pred_vz": out_np[:, 2],
        #             "true_vx": tgt_np[:, 0], "true_vy": tgt_np[:, 1], "true_vz": tgt_np[:, 2],
        #             "pred_speed": out_norm,
        #             "true_speed": tgt_norm,
        #             "cos_sim": cos_sim,
        #             "sq_err": sq_err,
        #         })

        #         save_dir = os.path.join(self.cfg.trainer.get("output_dir", "."), "velocity_logs")
        #         os.makedirs(save_dir, exist_ok=True)
        #         csv_path = os.path.join(save_dir, "velocity_field_train_20260723_cor.csv")
        #         df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        # # ★★★ 2026/7/16 ここまで ★★★
        
        

        # Log the loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def configure_optimizers(self):
        if self.cfg.optimizer.class_name == "Adam":
            optimizer_class = torch.optim.Adam
        else:
            raise ValueError(
                f"Unknown optimizer class: {self.cfg.optimizer.class_name}"
            )
        return optimizer_class(self.model.parameters(), lr=self.cfg.optimizer.lr)

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        # Preset the interpolation parameter for consistency
        t_list = torch.linspace(
            0.0,
            1.0,
            self.cfg.valid.num_time_steps,
            device=self.device,
            dtype=torch.float,
        )
        orig_batch = batch.clone()

        # Compute the loss for each time step
        loss_list = []
        for t in t_list:
            # Sample flow time
            t_per_graph = t.expand(batch.num_graphs).clone()

            # Interpolate
            x_cond = orig_batch["positions_1"].clone()
            disp = (orig_batch["positions_2"] - orig_batch["positions_1"]).clone()
            batch["positions_1"] = (
                orig_batch["positions_1"] + orig_batch["prior"]
            ).clone()
            batch["positions_2"] = orig_batch["positions_2"].clone()
            _t = t_per_graph[batch.batch][:, None]
            x_t = (1.0 - _t) * batch["positions_1"] + _t * batch["positions_2"]
            dx_dt = batch["positions_2"] - batch["positions_1"]

            # Compute the flow and loss
            batch["positions_1"] = x_cond
            batch["positions_2"] = x_t
            outputs = self(batch, t_per_graph=t_per_graph)
            if self.cfg.model.prediction_mode == "velocity":
                #loss = self.compute_loss(outputs, dx_dt)
                #loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"])  # ★ elements追加
                # ★ 変更: 第4引数に、真の変位である disp を渡す
                loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"], true_disp=disp)
                target = dx_dt          # ★ 追加: targetを明示的に保持
            elif self.cfg.model.prediction_mode == "data":
                #loss = self.compute_loss(outputs, disp)
                #loss = self.compute_loss(outputs, disp, elements=batch["elements"])  # ★ elements追加
                # ★ 変更: 第4引数に、真の変位である disp を渡す
                loss = self.compute_loss(outputs, dx_dt, elements=batch["elements"], true_disp=disp)
                target = disp            # ★ 追加
                
            
            # ★★★ ここに挿入 ★★★
            if batch_idx == 0:  # 検証は全ステップ保存すると重いので先頭バッチだけ等に絞る
                out_np = outputs.detach().cpu().numpy()
                tgt_np = target.detach().cpu().numpy()
                elem_np = batch["elements"].detach().cpu().numpy()
                
                
                # ★ 追加: elem=1だけを抜き出す
                mask = elem_np == 1
                out_np = out_np[mask]
                tgt_np = tgt_np[mask]
                elem_np = elem_np[mask]
                

                out_norm = np.linalg.norm(out_np, axis=-1)
                tgt_norm = np.linalg.norm(tgt_np, axis=-1)
                cos_sim = (out_np * tgt_np).sum(-1) / (out_norm * tgt_norm + 1e-8)
                sq_err = ((out_np - tgt_np) ** 2).sum(-1)

                df = pd.DataFrame({
                    "epoch": self.current_epoch,
                    "t": t.item(),          # ★ どの補間時刻かも記録
                    "batch_idx": batch_idx,
                    "elem": elem_np,
                    "pred_vx": out_np[:, 0], "pred_vy": out_np[:, 1], "pred_vz": out_np[:, 2],
                    "true_vx": tgt_np[:, 0], "true_vy": tgt_np[:, 1], "true_vz": tgt_np[:, 2],
                    "pred_speed": out_norm,
                    "true_speed": tgt_norm,
                    "cos_sim": cos_sim,
                    "sq_err": sq_err,
                })

                save_dir = os.path.join(self.cfg.trainer.get("output_dir", "."), "velocity_logs")
                os.makedirs(save_dir, exist_ok=True)
                csv_path = os.path.join(save_dir, "velocity_field_validation_20260723_cor.csv")
                df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
            # ★★★ ここまで ★★★
            
            loss_list.append(loss)

        loss = torch.stack(loss_list).mean()
        self.log("valid/loss", loss, on_epoch=True, batch_size=batch.num_graphs)
        return loss