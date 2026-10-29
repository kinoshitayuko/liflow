import MDAnalysis as mda
from MDAnalysis.analysis import msd
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_msd_data(
    traj_file,
    atom_selection='name Li',
    timestep_fs=1000
):
    """
    MSDデータ（時間軸とMSD値）のみを計算して返す関数
    """
    if not os.path.exists(traj_file):
        print(f"ファイル {traj_file} が存在しません。")
        return None, None

    u = mda.Universe(traj_file, format='XYZ')
    msd_analysis = msd.EinsteinMSD(u, select=atom_selection, msd_type='xyz', fft=True)
    msd_analysis.run()
    
    msd_values = msd_analysis.results.timeseries  # Å²
    num_frames = len(msd_values)
    timestep_ps = timestep_fs / 1e3
    times = np.arange(num_frames) * timestep_ps
    
    return times, msd_values

# =========================================================
# 設定: 比較したい温度とファイルを指定
# =========================================================
target_temp = 650  # 比較する温度 (K)

# 1. 自分のLiFlow実験データ
# my_liflow_file_2 = 'traj_650K_42_20260513_0.1_10.xyz'
my_liflow_file_1 = 'traj_650K_42_20260513.xyz'
# my_liflow_file_3 = 'traj_650K_42_20260513_0.1_30.xyz'
my_liflow_dt_fs = 1005  # タイムステップ (fs)

# 2. データセット (AIMD) データ
dataset_file = '../../data/data/LGPS/positions_650K.xyz'
dataset_dt_fs = 15      # タイムステップ (fs)

# =========================================================
# 計算実行
# =========================================================
print(f"--- {target_temp}K MSD Comparison ---")

# LiFlowのMSD計算
times_liflow_1, msd_liflow_1 = calculate_msd_data(my_liflow_file_1, timestep_fs=my_liflow_dt_fs)
# times_liflow_2, msd_liflow_2 = calculate_msd_data(my_liflow_file_2, timestep_fs=my_liflow_dt_fs)
# times_liflow_3, msd_liflow_3 = calculate_msd_data(my_liflow_file_3, timestep_fs=my_liflow_dt_fs)

# DatasetのMSD計算
times_dataset, msd_dataset = calculate_msd_data(dataset_file, timestep_fs=dataset_dt_fs)

# =========================================================
# 重ね書きプロット
# =========================================================
plt.figure(figsize=(8, 6))

# Dataset (AIMD) をプロット (通常は黒線や実線でリファレンスとして描く)
if times_dataset is not None:
    plt.plot(times_dataset, msd_dataset, 
             color='black', linestyle='-', linewidth=2, label='Dataset (AIMD)')

# LiFlow (Self) をプロット (色付きや破線で描く)
if times_liflow_1 is not None:
    plt.plot(times_liflow_1, msd_liflow_1, 
             color='red', linestyle='--', linewidth=2, marker='o', markersize=4, markevery=10, 
             label='LiFlow (default, sigma_Li = 10)')
             #label='LiFlow (My Experiment) 0.5 times')
    
# if times_liflow_2 is not None:
#     plt.plot(times_liflow_2, msd_liflow_2, 
#              color='blue', linestyle='--', linewidth=2, marker='o', markersize=4, markevery=10, 
#              label='LiFlow (sigma_Li = 20)')
#              #label='LiFlow (My Experiment) 1.0 times')
    
# if times_liflow_3 is not None:
#     plt.plot(times_liflow_3, msd_liflow_3, 
#              color='green', linestyle='--', linewidth=2, marker='o', markersize=4, markevery=10, 
#              label='LiFlow (sigma_Li = 30)')
#              #label='LiFlow (My Experiment) 1.5 times')

plt.xlabel('Time (ps)', fontsize=14)
plt.ylabel('MSD ($\AA^2$)', fontsize=14)
plt.title(f'Li Mean Squared Displacement at {target_temp} K', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 保存
save_filename = f'20260513_msd_comparison_{target_temp}K_only_inference.png'
plt.savefig(save_filename, dpi=300)
print(f"比較グラフを {save_filename} に保存しました。")

plt.show()