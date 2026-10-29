import numpy as np

# # LGPS/atomic_numbers.npy の中身を表示するPythonスクリプト
# file_path = 'atomic_numbers.npy'  # フルパスまたは相対パスを指定

# # npyファイルを辞書として読み込む
# data = np.load(file_path, allow_pickle=True).item()

# for key, value in data.items():
#     print(f'構造名: {key}')
#     print(f'原子番号配列: {value}')
#     print(f'原子数: {len(value)}')
#     print('-' * 40)


# # LGPS/element_index.npy の中身を表示するPythonスクリプト
# # ファイルパスを適宜変更してください
# file_path = 'element_index.npy'

# # npyファイルを読み込む
# element_index = np.load(file_path)

# # 配列内容の表示
# print("element_index.npy の中身:")
# print(element_index)
# print("要素数:", len(element_index))


# # LGPS/lattice.npy の中身を表示するPythonスクリプト
# # ファイルパスを適宜修正してください
# file_path = 'lattice.npy'

# # NumPyのallow_pickle=Trueで辞書として読み込み
# lattice_dict = np.load(file_path, allow_pickle=True).item()

# # 各構造の格子ベクトルを表示
# for structure_name, lattice in lattice_dict.items():
#     print(f"構造名: {structure_name}")
#     print("格子ベクトル (3x3行列):")
#     print(lattice)
#     print("\n" + "-"*50 + "\n")


# # LGPS/positions_650K.npz の中身を表示するPythonスクリプト
# # ファイルパスを適宜修正してください
# file_path = "positions_300K.npz"

# # npzファイルを読み込む
# data = np.load(file_path)

# # 構造名（キー）の一覧を表示
# print("含まれる構造名:", list(data.keys()))

# # 最初の構造のデータを例として表示
# name = list(data.keys())[0]
# positions = data[name]
# print(f"構造名: {name}")
# print(f"配列shape: {positions.shape}")  # (フレーム数, 原子数, 3)
# print("最初の2フレームの原子座標:")
# print(positions[:2])  # 最初の2フレーム分の全原子座標


#lammpstrjからxyz形式に変換するスクリプト
# lammpstrjファイルをmulti-frame xyzに変換
# lammpstrjファイルからmulti-frame XYZへ変換
input_file = 'log.lammpstrj'
output_file = 'output.xyz'

with open(input_file) as fin:
    lines = fin.readlines()

i = 0
with open(output_file, 'w') as fout:
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            # タイムステップ取得
            timestep = int(lines[i+1].strip())
            i += 2
            if not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Expected 'ITEM: NUMBER OF ATOMS' at line %d" % i)
            natoms = int(lines[i+1].strip())
            i += 2
            if not lines[i].startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Expected 'ITEM: BOX BOUNDS' at line %d" % i)
            i += 4  # BOX BOUNDSをスキップ
            if not lines[i].startswith("ITEM: ATOMS"):
                raise ValueError("Expected 'ITEM: ATOMS' at line %d" % i)
            cols = lines[i].replace('ITEM: ATOMS ', '').strip().split()
            data = [l.strip().split() for l in lines[i+1:i+1+natoms]]
            i += 1 + natoms
            if not all(col in cols for col in ['element', 'xu', 'yu', 'zu']):
                raise ValueError("element, xu, yu, zu columns are required in lammpstrj")
            idx_e = cols.index('element')
            idx_x = cols.index('xu')
            idx_y = cols.index('yu')
            idx_z = cols.index('zu')
            # xyz1フレーム出力
            fout.write(f"{natoms}\n")
            fout.write(f"Frame {timestep}\n")
            for row in data:
                fout.write(f"{row[idx_e]} {float(row[idx_x])} {float(row[idx_y])} {float(row[idx_z])}\n")
        else:
            i += 1


