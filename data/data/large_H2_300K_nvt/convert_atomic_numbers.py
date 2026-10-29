import numpy as np
import pandas as pd

# element_symbol_to_number = {
#     'H': 1, 'He':2, 'Li': 3, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
#     'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
#     'Cl': 17, 'K': 19, 'Ca': 20, 'Ti': 22, 'Ge': 32, 'Br': 35,
#     'I': 53, 'Sn': 50, 'Pb': 82
# }

# filename = 'trajectory_0-20000/frame_00000.cif'  # 添付ファイル名
# keyname = 'large_H2_300K_nvt'

# with open(filename) as f:
#     lines = f.readlines()

# # 1. "_atom_site_label" を起点にカラム名群→データ部までを切り出し
# column_start, data_start = None, None
# columns = []
# for i, line in enumerate(lines):
#     if line.strip().startswith("_atom_site_label"):
#         # label群の開始位置
#         column_start = i
#         # カラム群を取得
#         j = i
#         while lines[j].strip().startswith("_atom_site_"):
#             columns.append(lines[j].strip().replace("_atom_site_", ""))
#             j += 1
#         data_start = j
#         break

# # 2. データ部取得（空行・ループ・新しい_tag で終了）
# atom_data = []
# for line in lines[data_start:]:
#     if not line.strip() or line.strip().startswith("loop_") or line.strip().startswith("_"):
#         break
#     atom_data.append(line.strip().split())

# # 3. DataFrame生成
# df = pd.DataFrame(atom_data, columns=columns)

# # 4. type_symbol列から原子記号→原子番号
# element_list = df['type_symbol'].tolist()
# atomic_numbers = np.array([element_symbol_to_number[e] for e in element_list], dtype=int)
# atomic_numbers_dict = {keyname: atomic_numbers}

# np.save('atomic_numbers.npy', atomic_numbers_dict, allow_pickle=True)
# print('atomic_numbers.npy（dict型）を出力しました')

# #次はlattice.npyの作成
# import numpy as np
# # 例として、3x3の格子ベクトルを持つ構造を定義

# # 格子長さを手動で指定（例: log.lammpstrjのBOX BOUNDSから抜粋）
# lx = 23.7622      # x方向のボックス長
# ly = 23.7622      # y方向のボックス長
# lz = 23.7622      # z方向のボックス長

# # 直交格子の場合の3×3格子行列
# lattice = np.array([
#     [lx, 0.0, 0.0],
#     [0.0, ly, 0.0],
#     [0.0, 0.0, lz]
# ])

# # 構造名（用途に応じて変更）
# structure_name = 'large_H2_300K_nvt'

# # 構造名をキーとするdict型にまとめる
# lattice_dict = {structure_name: lattice}

# # npyとして保存
# np.save('lattice.npy', lattice_dict, allow_pickle=True)
# print('lattice.npy を出力しました')

# #次はpositions_300K.npzの作成
# # このスクリプトは、CIFファイルから原子座標を抽出し、指定された形式で保存するものです。
# # CIFファイルのフォーマットに依存しているため、適切なCIFファイルを用意してください。
# import numpy as np
# import os

# # --- 必要入力を設定 ---
# structure_name = 'large_H2_300K_nvt'  # 構造名 (変更可)
# natoms = 962  # 原子数 (frame_00000.cifを参照) ※実際の数に修正
# n_frames = 20001
# positions_list = []

# # CIFファイルのパス設定
# def cif_filename(idx):
#     return f'trajectory_0-20000/frame_{idx:05d}.cif'

# def extract_positions_from_cif(cif_file):
#     with open(cif_file, 'r') as f:
#         lines = f.readlines()
#     atoms = []
#     start = False
#     for line in lines:
#         if line.strip().startswith('_atom_site_occupancy'):
#             start = True
#             continue
#         if start:
#             # 行が空か、次のセクション始まりならbreak
#             if line.strip() == '' or line.startswith('loop_') or line.startswith('_geom_bond'):
#                 break
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue  # 不完全な行をスキップ
#             # CIFデータの座標抽出部分（2:x, 3:y, 4:z）
#             fract_x = float(parts[2])
#             fract_y = float(parts[3])
#             fract_z = float(parts[4])
#             atoms.append([fract_x, fract_y, fract_z])
#     return np.array(atoms)

# for idx in range(n_frames):
#     cif_file = cif_filename(idx)
#     if not os.path.isfile(cif_file):
#         print(f'ファイルが見つかりません: {cif_file}')
#         continue
#     coords = extract_positions_from_cif(cif_file)
#     assert coords.shape == (natoms, 3), f"{cif_file}: shape mismatch {coords.shape}"
#     positions_list.append(coords)

# positions_arr = np.stack(positions_list)  # (n_frames, natoms, 3)
# positions_dict = {structure_name: positions_arr}
# np.savez('positions_300K.npz', **positions_dict)
# print('positions_300K.npz を作成しました')

# import numpy as np

# # 1. 系に含まれる元素（原子番号 → インデックスへの割り当て）
# # 例: LGPSの場合
# element_numbers = [1, 8, 14, 22]  # H, He, O, Si, Ti の原子番号
# element_indices = list(range(len(element_numbers)))  # インデックス: 0, 1, 2, ...

# # 2. 配列サイズ（93要素＝最大で原子番号92:ウランまで）
# n = 93
# element_index = np.full(n, -1, dtype=int)

# # 3. 必要な場所にインデックスをセット
# for i, z in enumerate(element_numbers):
#     element_index[z] = i

# # 4. セーブ
# np.save('element_index.npy', element_index)
# print('element_index.npy を出力しました（内容: -1埋め/原子番号index指定の型）')



import ase.io
from ase.io import read
from pathlib import Path
from collections import defaultdict

# 入出力ファイル
cif_path = Path("trajectory_0-20000/frame_00000.cif")                # 入力CIFファイル名
sid_path = Path("TiO2SiO2_init_POSCAR.sid")       # 出力SIDファイル名

# CIFファイルの読み込み（最初のフレーム）
atoms = read(str(cif_path), format="cif")
atoms.set_pbc([True, True, True])

# 原子種ごとにグループ化
element_groups = defaultdict(list)
element_indices = defaultdict(list)

for i, symbol in enumerate(atoms.get_chemical_symbols()):
    element_groups[symbol].append(atoms.positions[i])
    element_indices[symbol].append(i)

# 出力元素順
priority_order = ['H', 'He', 'Si', 'O', 'Ti']
sorted_elements = [el for el in priority_order if el in element_groups] + \
                  [el for el in element_groups if el not in priority_order]

new_positions = []
new_symbols = []
element_order = []
element_counts = []

for element in sorted_elements:
    positions = element_groups[element]
    count = len(positions)
    element_order.append(element)
    element_counts.append(count)
    new_positions.extend(positions)
    new_symbols.extend([element] * count)

# Atomsオブジェクトを新しく構築
from ase import Atoms
new_atoms = Atoms(
    symbols=new_symbols,
    positions=new_positions,
    cell=atoms.cell,
    pbc=atoms.pbc
)

# カスタムPOSCAR(.sid)ファイル出力
with open(sid_path, 'w') as f:
    f.write("TiO2-SiO2_structure\n")
    f.write("1.0\n")
    cell = new_atoms.cell
    for i in range(3):
        f.write(f"   {cell[i,0]:15.10f}   {cell[i,1]:15.10f}   {cell[i,2]:15.10f}\n")
    f.write(" ".join(element_order) + "\n")
    f.write(" ".join(map(str, element_counts)) + "\n")
    f.write("Cartesian\n")
    for pos in new_positions:
        f.write(f"   {pos[0]:15.10f}   {pos[1]:15.10f}   {pos[2]:15.10f}\n")

print(f"Saved grouped structure → {sid_path.resolve()}")
print(f"Element order: {element_order}")
print(f"Element counts: {element_counts}")








# """
# lammpstrj → POSCAR (.sid) 変換スクリプト（原子種グループ化対応）
# """

# import ase.io
# import numpy as np
# from pathlib import Path
# from collections import defaultdict

# # --- 入出力ファイル ---
# dump_path = Path("log.lammpstrj")                    # lammpstrj ファイル
# sid_path   = Path("TiO2SiO2_init_POSCAR.sid")        # 出力ファイル

# # --- lammpstrj の最初のフレームを読み込む ---
# atoms = ase.io.read(dump_path, format="lammps-dump-text", index=0)
# atoms.set_pbc([True, True, True])

# # --- 原子種ごとにグループ化 ---
# # 各原子種の位置を収集
# element_groups = defaultdict(list)
# element_indices = defaultdict(list)

# for i, symbol in enumerate(atoms.get_chemical_symbols()):
#     element_groups[symbol].append(atoms.positions[i])
#     element_indices[symbol].append(i)

# # --- 新しい原子配列を作成（原子種ごとにまとめる） ---
# new_positions = []
# new_symbols = []
# element_order = []
# element_counts = []

# # 元素の順序を決定（通常の順序: H, Li, ..., Si, Ti, O など）
# # TiO2-SiO2系では H, Si, O, Ti の順序が適切
# priority_order = ['H', 'He', 'Si', 'O', 'Ti']
# sorted_elements = sorted(element_groups.keys(), 
#                         key=lambda x: priority_order.index(x) if x in priority_order else 999)

# for element in sorted_elements:
#     positions = element_groups[element]
#     count = len(positions)
    
#     element_order.append(element)
#     element_counts.append(count)
    
#     # 位置を追加
#     new_positions.extend(positions)
#     new_symbols.extend([element] * count)

# # --- 新しいAtomsオブジェクトを作成 ---
# new_atoms = ase.Atoms(
#     symbols=new_symbols,
#     positions=new_positions,
#     cell=atoms.cell,
#     pbc=atoms.pbc
# )

# # --- カスタムPOSCAR書き出し ---
# with open(sid_path, 'w') as f:
#     # コメント行
#     f.write("TiO2-SiO2_structure\n")
    
#     # スケーリング係数
#     f.write("1.0\n")
    
#     # 格子ベクトル
#     cell = new_atoms.cell
#     for i in range(3):
#         f.write(f"   {cell[i,0]:15.10f}   {cell[i,1]:15.10f}   {cell[i,2]:15.10f}\n")
    
#     # 原子種名
#     f.write(" ".join(element_order) + "\n")
    
#     # 各原子種の個数
#     f.write(" ".join(map(str, element_counts)) + "\n")
    
#     # 座標系
#     f.write("Cartesian\n")
    
#     # 原子座標
#     for pos in new_positions:
#         f.write(f"   {pos[0]:15.10f}   {pos[1]:15.10f}   {pos[2]:15.10f}\n")

# print(f"Saved grouped structure → {sid_path.resolve()}")
# print(f"Element order: {element_order}")
# print(f"Element counts: {element_counts}")




