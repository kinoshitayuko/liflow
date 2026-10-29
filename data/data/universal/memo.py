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


# LGPS/element_index.npy の中身を表示するPythonスクリプト
# ファイルパスを適宜変更してください
file_path = 'element_index.npy'

# npyファイルを読み込む
element_index = np.load(file_path)

# 配列内容の表示
print("element_index.npy の中身:")
print(element_index)
print("要素数:", len(element_index))
