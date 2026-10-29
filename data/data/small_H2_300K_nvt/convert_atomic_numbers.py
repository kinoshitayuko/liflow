import numpy as np
import pandas as pd

# 元素記号→原子番号辞書
element_symbol_to_number = {
    'H': 1, 'He':2, 'Li': 3, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
    'Cl': 17, 'K': 19, 'Ca': 20, 'Ti': 22, 'Ge': 32, 'Br': 35, 'I': 53, 'Sn': 50, 'Pb': 82
}

# 必要にファイル名・構造名リストなどを設定
traj_files = {
    'small_H2_300K_nvt': 'log.lammpstrj'
    # ... 必要に応じて追加
}

atomic_numbers_dict = {}

for name, path in traj_files.items():
    with open(path) as f:
        lines = f.readlines()
    # 1フレームだけで十分なら、最初のATOMS項目でOK
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: ATOMS"):
            columns = lines[i].replace("ITEM: ATOMS ", "").strip().split()
            data = [l.strip().split() for l in lines[i+1:i+1+167]]  # 原子数（例:200等）に合わせて調整
            df = pd.DataFrame(data, columns=columns)
            elist = df['element'].tolist()
            atomic_numbers = np.array([element_symbol_to_number[e] for e in elist], dtype=int)
            atomic_numbers_dict[name] = atomic_numbers
            break
        i += 1

# ファイル名指定でdict型npyをセーブ
np.save('atomic_numbers.npy', atomic_numbers_dict, allow_pickle=True)
print('atomic_numbers.npy（dict型）を出力しました')

#次はlattice.npyの作成
import numpy as np
# 例として、3x3の格子ベクトルを持つ構造を定義
import numpy as np

# 格子長さを手動で指定（例: log.lammpstrjのBOX BOUNDSから抜粋）
lx = 13.659831      # x方向のボックス長
ly = 13.659831      # y方向のボックス長
lz = 13.659831      # z方向のボックス長

# 直交格子の場合の3×3格子行列
lattice = np.array([
    [lx, 0.0, 0.0],
    [0.0, ly, 0.0],
    [0.0, 0.0, lz]
])

# 構造名（用途に応じて変更）
structure_name = 'small_H2_300K_nvt'

# 構造名をキーとするdict型にまとめる
lattice_dict = {structure_name: lattice}

# npyとして保存
np.save('lattice.npy', lattice_dict, allow_pickle=True)
print('lattice.npy を出力しました')

# 次はpositions_300K.npzの作成
import numpy as np

# --- 必要入力を設定 ---
logfile = 'log.lammpstrj'
structure_name = 'small_H2_300K_nvt'  # 構造名を適宜セット
natoms = 167                      # 原子数に合わせて設定（例: 165）

positions_list = []

with open(logfile) as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if lines[i].startswith("ITEM: TIMESTEP"):
        i += 2  # TIMESTEPとその値を飛ばす
        # 原子数
        assert "ITEM: NUMBER OF ATOMS" in lines[i]
        n_atoms = int(lines[i+1].strip())
        assert n_atoms == natoms
        i += 2
        # ボックス
        assert "ITEM: BOX BOUNDS" in lines[i]
        i += 4
        # ATOMS
        assert "ITEM: ATOMS" in lines[i]
        columns = lines[i].replace("ITEM: ATOMS ", "").strip().split()
        data = [l.strip().split() for l in lines[i+1:i+1+natoms]]
        # xu, yu, zu列を特定
        pos_idx = [columns.index('xu'), columns.index('yu'), columns.index('zu')]
        coords = np.array([[float(row[j]) for j in pos_idx] for row in data])
        positions_list.append(coords)
        i += 1 + natoms
    else:
        i += 1

# [n_frames, n_atoms, 3]形状のnumpy配列に
positions_arr = np.stack(positions_list)  # 例: (n_frames, 167, 3)

# {構造名: 配列} のdictにまとめて保存
positions_dict = {structure_name: positions_arr}
np.savez('positions_300K.npz', **positions_dict)
print('positions_300K.npz を作成しました')


import numpy as np

# 1. 系に含まれる元素（原子番号 → インデックスへの割り当て）
# 例: LGPSの場合
element_numbers = [1, 2, 8, 14, 22]  # H, He, O, Si, Ti の原子番号
element_indices = list(range(len(element_numbers)))  # インデックス: 0, 1, 2, ...

# 2. 配列サイズ（93要素＝最大で原子番号92:ウランまで）
n = 93
element_index = np.full(n, -1, dtype=int)

# 3. 必要な場所にインデックスをセット
for i, z in enumerate(element_numbers):
    element_index[z] = i

# 4. セーブ
np.save('element_index.npy', element_index)
print('element_index.npy を出力しました（内容: -1埋め/原子番号index指定の型）')



"""
lammpstrj → POSCAR (.sid) 変換スクリプト（原子種グループ化対応）
"""

import ase.io
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- 入出力ファイル ---
dump_path = Path("log.lammpstrj")                    # lammpstrj ファイル
sid_path   = Path("TiO2SiO2_init_POSCAR.sid")        # 出力ファイル

# --- lammpstrj の最初のフレームを読み込む ---
atoms = ase.io.read(dump_path, format="lammps-dump-text", index=0)
atoms.set_pbc([True, True, True])

# --- 原子種ごとにグループ化 ---
# 各原子種の位置を収集
element_groups = defaultdict(list)
element_indices = defaultdict(list)

for i, symbol in enumerate(atoms.get_chemical_symbols()):
    element_groups[symbol].append(atoms.positions[i])
    element_indices[symbol].append(i)

# --- 新しい原子配列を作成（原子種ごとにまとめる） ---
new_positions = []
new_symbols = []
element_order = []
element_counts = []

# 元素の順序を決定（通常の順序: H, Li, ..., Si, Ti, O など）
# TiO2-SiO2系では H, Si, O, Ti の順序が適切
priority_order = ['H', 'He', 'Si', 'O', 'Ti']
sorted_elements = sorted(element_groups.keys(), 
                        key=lambda x: priority_order.index(x) if x in priority_order else 999)

for element in sorted_elements:
    positions = element_groups[element]
    count = len(positions)
    
    element_order.append(element)
    element_counts.append(count)
    
    # 位置を追加
    new_positions.extend(positions)
    new_symbols.extend([element] * count)

# --- 新しいAtomsオブジェクトを作成 ---
new_atoms = ase.Atoms(
    symbols=new_symbols,
    positions=new_positions,
    cell=atoms.cell,
    pbc=atoms.pbc
)

# --- カスタムPOSCAR書き出し ---
with open(sid_path, 'w') as f:
    # コメント行
    f.write("TiO2-SiO2_structure\n")
    
    # スケーリング係数
    f.write("1.0\n")
    
    # 格子ベクトル
    cell = new_atoms.cell
    for i in range(3):
        f.write(f"   {cell[i,0]:15.10f}   {cell[i,1]:15.10f}   {cell[i,2]:15.10f}\n")
    
    # 原子種名
    f.write(" ".join(element_order) + "\n")
    
    # 各原子種の個数
    f.write(" ".join(map(str, element_counts)) + "\n")
    
    # 座標系
    f.write("Cartesian\n")
    
    # 原子座標
    for pos in new_positions:
        f.write(f"   {pos[0]:15.10f}   {pos[1]:15.10f}   {pos[2]:15.10f}\n")

print(f"Saved grouped structure → {sid_path.resolve()}")
print(f"Element order: {element_order}")
print(f"Element counts: {element_counts}")




