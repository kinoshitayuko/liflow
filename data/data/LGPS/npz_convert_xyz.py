#20250624

import numpy as np

def npz_to_xyz(npz_path, atomic_numbers_path, xyz_path):
    # 元素番号から記号への変換テーブル
    atomic_number_to_symbol = {
        1: 'H', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
        11: 'Na', 12: 'Mg', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
        19: 'K', 20: 'Ca', 32: 'Ge', 53: 'I',
        # 必要に応じて追加
    }
    
    # npzファイル読み込み
    data = np.load(npz_path, allow_pickle=True)
    positions = data['LGPS'] if 'LGPS' in data else data[data.files[0]]
    
    # 原子番号配列読み込み
    atomic_numbers_data = np.load(atomic_numbers_path, allow_pickle=True)
    
    # 0次元配列の場合の対処
    atomic_numbers_data = np.atleast_1d(atomic_numbers_data)
    
    # 配列の情報を表示（デバッグ用）
    print(f"atomic_numbers shape: {atomic_numbers_data.shape}")
    print(f"atomic_numbers content: {atomic_numbers_data}")
    
    # 辞書から実際の原子番号配列を取得
    if len(atomic_numbers_data) > 0 and isinstance(atomic_numbers_data[0], dict):
        if 'LGPS' in atomic_numbers_data[0]:
            atomic_numbers = atomic_numbers_data[0]['LGPS']
        else:
            # 辞書の最初の値を使用
            atomic_numbers = list(atomic_numbers_data[0].values())[0]
    else:
        atomic_numbers = atomic_numbers_data
    
    print(f"実際の原子番号配列 shape: {atomic_numbers.shape}")
    print(f"原子番号の例: {atomic_numbers[:10]}")  # 最初の10個を表示
    
    # 原子記号リスト作成
    symbols = [atomic_number_to_symbol.get(num, 'X') for num in atomic_numbers]
    
    n_frames, n_atoms, _ = positions.shape
    print(f"フレーム数: {n_frames}, 原子数: {n_atoms}")
    print(f"元素記号数: {len(symbols)}")
    
    # 原子記号の統計を表示
    from collections import Counter
    symbol_counts = Counter(symbols)
    print(f"元素記号の分布: {symbol_counts}")
    
    with open(xyz_path, 'w') as f:
        for frame in range(n_frames):
            f.write(f"{n_atoms}\n")
            f.write(f"Frame {frame+1}\n")
            for atom in range(n_atoms):
                # 元素記号の配列が足りない場合の対処
                if atom < len(symbols):
                    symbol = symbols[atom]
                else:
                    symbol = 'X'  # デフォルト記号
                x, y, z = positions[frame, atom]
                f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"変換完了: {xyz_path}")

if __name__ == "__main__":
    npz_path = "positions_1400K.npz"
    atomic_numbers_path = "atomic_numbers.npy"
    xyz_path = "positions_1400K.xyz"
    npz_to_xyz(npz_path, atomic_numbers_path, xyz_path)
