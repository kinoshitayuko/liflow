import numpy as np

# LGPS/atomic_numbers.npy の中身を表示するPythonスクリプト
file_path = 'atomic_numbers.npy'  # フルパスまたは相対パスを指定

# npyファイルを辞書として読み込む
data = np.load(file_path, allow_pickle=True).item()

for key, value in data.items():
    print(f'構造名: {key}')
    print(f'原子番号配列: {value}')
    print(f'原子数: {len(value)}')
    print('-' * 40)


# LGPS/element_index.npy の中身を表示するPythonスクリプト
# ファイルパスを適宜変更してください
file_path = 'element_index.npy'

# npyファイルを読み込む
element_index = np.load(file_path)

# 配列内容の表示
print("element_index.npy の中身:")
print(element_index)
print("要素数:", len(element_index))


# LGPS/lattice.npy の中身を表示するPythonスクリプト
# ファイルパスを適宜修正してください
file_path = 'lattice.npy'

# NumPyのallow_pickle=Trueで辞書として読み込み
lattice_dict = np.load(file_path, allow_pickle=True).item()

# 各構造の格子ベクトルを表示
for structure_name, lattice in lattice_dict.items():
    print(f"構造名: {structure_name}")
    print("格子ベクトル (3x3行列):")
    print(lattice)
    print("\n" + "-"*50 + "\n")


# LGPS/positions_650K.npz の中身を表示するPythonスクリプト
# ファイルパスを適宜修正してください
file_path = "positions_650K.npz"

# npzファイルを読み込む
data = np.load(file_path)

# 構造名（キー）の一覧を表示
print("含まれる構造名:", list(data.keys()))

# 最初の構造のデータを例として表示
name = list(data.keys())[0]
positions = data[name]
print(f"構造名: {name}")
print(f"配列shape: {positions.shape}")  # (フレーム数, 原子数, 3)
print("最初の2フレームの原子座標:")
print(positions[:2])  # 最初の2フレーム分の全原子座標
