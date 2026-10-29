# export LD_LIBRARY_PATH=/home/user/anaconda3/envs/liflow/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=0
# 最初にこれやる↑
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

from pathlib import Path
import argparse

import ase
import ase.io
import numpy as np
import torch

from liflow.model.modules import FlowModule
from liflow.utils.inference import FlowSimulator
from liflow.utils.prior import get_prior

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 決定論的アルゴリズムの使用を強制（速度が落ちる可能性あり）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # これが重要！
    # PyTorchの演算を決定論的にする（エラーが出る場合あり）
    torch.use_deterministic_algorithms(True) 



torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
# Input and output directories
parser.add_argument("--data-dir", type=str, default="../data/data/small_H2_300K_nvt")
parser.add_argument("--output-dir", type=str, default="traj_output_TiO2SiO2_H2")

# Checkpoint paths (set None for corrector to disable)
#parser.add_argument("--propagate-ckpt", type=str, default="ckpt/P_TiO2SiO2_H2_20250926_1-v5.ckpt")
#parser.add_argument("--propagate-ckpt", type=str, default="ckpt/P_TiO2SiO2_H2_20250825_1-v1.ckpt")
parser.add_argument("--propagate-ckpt", type=str, default="ckpt/P_TiO2SiO2_H2_20260629_2_default.ckpt")
#parser.add_argument("--propagate-ckpt", type=str, default="ckpt/P_TiO2SiO2_H2_20250825_1.ckpt")
# parser.add_argument("--correct-ckpt", type=str, default="ckpt/C_TiO2SiO2_H2_0.5_20260629_2.ckpt")
parser.add_argument("--correct-ckpt", type=str, default="ckpt/C_TiO2SiO2_H2_prior_0.5_20260609.ckpt")
#parser.add_argument("--correct-ckpt", type=str, default="ckpt/C_TiO2SiO2_H2_0.5_10_20251216_750-v1.ckpt")

# Simulation parameters: number of steps, flow integration steps, solver
parser.add_argument("--steps", type=int, default=2000) #変える
#parser.add_argument("--flow-steps", type=int, default=10)
parser.add_argument("--flow-steps", type=int, default=10)
parser.add_argument("--solver", type=str, default="euler")

# Supercell
parser.add_argument("--supercell", type=int, nargs=3, default=[1, 1, 1])

# Temperature list
parser.add_argument("--temp-list", type=int, nargs="+", default=[300])

# Miscellaneous parameters: seed, corrector application frequency, number of retries
parser.add_argument("--seed", type=int, default=42)

# メイン処理の前に呼び出す
args = parser.parse_args()
set_seed(args.seed)


#parser.add_argument("--corrector-every", type=int, default=1)
parser.add_argument("--corrector-every", type=int, default=1)  # 1ステップごとにコレクターを適用（実質的にコレクターなし）
parser.add_argument("--num-retries", type=int, default=3)

#parser.add_argument("--num-retries", type=int, default=0)

args = parser.parse_args()

data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Prior scales for LGPS
scale_Li_index, scale_frame_index = 1, 0

# Load initial structure and create supercell
#atoms = ase.io.read(data_dir / "TiO2SiO2_init_POSCAR_91800.sid")
#atoms = ase.io.read(data_dir / "TiO2SiO2_init_POSCAR_75000.sid")
atoms = ase.io.read(data_dir / "TiO2SiO2_init_POSCAR.sid")
atoms = atoms.repeat(args.supercell)

# Load element index
element_idx = np.load(data_dir / "element_index.npy")

# Load propagator model
propagate_model = FlowModule.load_from_checkpoint(args.propagate_ckpt).eval()
cfg_prior = propagate_model.cfg.propagate_prior
propagate_prior = get_prior(cfg_prior.class_name, **cfg_prior.params, seed=args.seed)

# Load corrector model
if args.correct_ckpt is not None:
    correct_model = FlowModule.load_from_checkpoint(args.correct_ckpt).eval()
    cfg_prior = correct_model.cfg.correct_prior
    correct_prior = get_prior(cfg_prior.class_name, **cfg_prior.params, seed=args.seed)
else:
    correct_model = None
    correct_prior = None
    
    
# # 新規追加: H2原子のインデックスを特定 (原子番号1の原子をペアリング)
# # 仮定: H2分子が複数ある場合、偶数番目のHをペア (例: [0,1], [2,3]...)
# atomic_numbers = atoms.get_atomic_numbers()
# h_indices = np.where(atomic_numbers == 2)[0]  # 水素原子のインデックス
# if len(h_indices) % 2 != 0:
#     raise ValueError("H原子数が奇数: H2ペアが作れません")
# h_pairs = [(h_indices[i], h_indices[i+1]) for i in range(0, len(h_indices), 2)]

# 新規追加: シンプルな結合長固定関数 (calculatorなしで手動調整)
# 各H2ペアの距離を固定値にスケーリング (エラー回避のため最適化スキップ)
# def apply_simple_bond_fix(atoms, pairs, bond_length=0.74):
#     positions = atoms.get_positions()
#     cell = atoms.cell.array
    
#     for a, b in pairs:
#         vec = positions[b] - positions[a]
#         # PBC補正（最短イメージ規則）
#         vec -= np.round(vec @ np.linalg.inv(cell)) @ cell
        
#         current_dist = np.linalg.norm(vec)
#         if current_dist > 0:
#             correction = (vec / current_dist) * bond_length
#             center = (positions[a] + positions[b]) / 2  # 重心保存
#             positions[a] = center - correction / 2
#             positions[b] = center + correction / 2
    
#     atoms.set_positions(positions)
#     return atoms

# 新規追加: H2原子のインデックスを特定 (原子番号1の原子をペアリング)
# 仮定: H2分子が複数ある場合、偶数番目のHをペア (例: [0,1], [2,3]...)
atomic_numbers = atoms.get_atomic_numbers()
h_indices = np.where(atomic_numbers == 2)[0]  # 水素原子のインデックス
if len(h_indices) % 2 != 0:
    raise ValueError("H原子数が奇数: H2ペアが作れません")
h_pairs = [(h_indices[i], h_indices[i+1]) for i in range(0, len(h_indices), 2)]

# 新規追加: シンプルな結合長固定関数 (calculatorなしで手動調整)
# 各H2ペアの距離を固定値にスケーリング (エラー回避のため最適化スキップ)
def apply_simple_bond_fix(atoms, pairs, bond_length=0.74):
    positions = atoms.get_positions()  # 位置を取得
    for a, b in pairs:
        vec = positions[b] - positions[a]  # ベクトル計算
        current_dist = np.linalg.norm(vec)  # 現在の距離
        if current_dist > 0:  # ゼロ除算回避
            # 距離を固定値にスケーリング (方向は維持)
            positions[b] = positions[a] + (vec / current_dist) * bond_length
    atoms.set_positions(positions)  # 位置を更新
    return atoms



for temperature in args.temp_list:
    print(f"Running simulation at {temperature} K")
    # Initialize simulator
    atomic_numbers = atoms.get_atomic_numbers()
    
    simulator = FlowSimulator(
        propagate_model=propagate_model,
        propagate_prior=propagate_prior,
        atomic_numbers=atomic_numbers,
        element_idx=element_idx,
        lattice=atoms.cell.array,
        temp=temperature,
        correct_model=correct_model,
        correct_prior=correct_prior,
        scale_Li_index=scale_Li_index,
        scale_frame_index=scale_frame_index,
        #fix_frame_species=[1, 8, 14, 22],  # ★ 20260526追加: 原子番号1（H）をフレーム固定
        fix_frame_species=None,  # ★ 20260526追加: 原子番号1（H）をフレーム固定
    )

    # Run simulation
    traj = simulator.run(
        positions=atoms.get_positions(),
        steps=args.steps,
        flow_steps=args.flow_steps,
        solver=args.solver,
        verbose=True,
        num_retries=args.num_retries,
        corrector_every=args.corrector_every,
    )
    
    
    # 新規追加: 軌道全体にシンプル固定を適用 (各フレームで調整)
    traj_fixed = []
    for pos in traj:
        temp_atoms = ase.Atoms(
            numbers=atomic_numbers,
            positions=pos,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        temp_atoms = apply_simple_bond_fix(temp_atoms, h_pairs)  # シンプル固定関数呼び出し
        traj_fixed.append(temp_atoms.get_positions())

    #Save trajectory (固定版を保存)
    traj_atoms = [
        ase.Atoms(
            numbers=atomic_numbers,
            positions=traj_fixed[i],
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        for i in range(len(traj_fixed))
    ]
    
    # # Save trajectory
    # traj_atoms = [
    #     ase.Atoms(
    #         numbers=atomic_numbers,
    #         positions=traj[i],
    #         cell=atoms.cell,
    #         pbc=atoms.pbc,
    #     )
    #     for i in range(len(traj))
    # ]
    ase.io.write(output_dir / f"traj_{temperature}K_{args.seed}_20260701_3.xyz", traj_atoms)
