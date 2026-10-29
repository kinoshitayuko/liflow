import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import numpy as np
import matplotlib.pyplot as plt

def compute_rdfs_with_mdanalysis(xyz_file, box, r_max=10.0, num_bins=100):
    """
    Compute partial RDFs for specified pairs using MDAnalysis.
    Assumes periodic boundary conditions with box [Lx, Ly, Lz].
    Returns a dictionary of (pair: (r, g_r)).
    """
    # Load the universe from XYZ file (treat as trajectory)
    universe = mda.Universe(xyz_file, format='XYZ', topology_format='XYZ')
    
    # Set box dimensions (periodic boundary conditions)
    # 非周期系の場合: universe.dimensions = None に変更
    universe.dimensions = np.array([box[0], box[1], box[2], 90, 90, 90])  # Assuming orthorhombic box
    
    # Define atom selections
    selections = {
        'Ti': universe.select_atoms('name Ti'),
        'O': universe.select_atoms('name O'),
        'Si': universe.select_atoms('name Si')
    }
    
    # Pairs to compute
    pairs = [('Ti', 'Ti'), ('Ti', 'O'), ('Ti', 'Si'), ('Si', 'Si'), ('Si', 'O'), ('O', 'O')]
    
    results = {}
    for typeA, typeB in pairs:
        groupA = selections[typeA]
        groupB = selections[typeB]
        
        # Skip if no atoms in group
        if len(groupA) == 0 or len(groupB) == 0:
            print(f"Skipping pair {typeA}-{typeB}: No atoms found.")
            continue
        
        # exclusion_block: homo-pairの場合に自己相関を除外 (1,1)で同じグループ内のペアをスキップ
        exclusion = (1, 1) if typeA == typeB else None
        
        # Compute RDF (averaged over all frames)
        rdf_analysis = rdf.InterRDF(groupA, groupB, nbins=num_bins, range=(0.0, r_max), exclusion_block=exclusion)
        rdf_analysis.run()
        
        r = rdf_analysis.results.bins
        g_r = rdf_analysis.results.rdf
        
        results[(typeA, typeB)] = (r, g_r)
    
    return results

def plot_overlapped_rdfs(results1, results2, output_file='overlapped_rdfs_20250723.png'):
    """
    Plot overlapped RDFs for each pair from two results.
    """
    # 各ペアごとに重ねてプロット（個別の図）
    for pair in results1.keys():
        r1, g_r1 = results1[pair]
        r2, g_r2 = results2[pair]
        
        plt.figure(figsize=(6, 4))
        plt.plot(r1, g_r1, label='dataset')
        plt.plot(r2, g_r2, label='inference', linestyle='--')
        plt.title(f'Overlapped RDF: {pair[0]}-{pair[1]}')
        plt.xlabel('r (Å)')
        plt.ylabel('g(r)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # 全てのペアを1つの図に重ねて配置 (2x3レイアウト)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, pair in zip(axes.flat, results1.keys()):
        r1, g_r1 = results1[pair]
        r2, g_r2 = results2[pair]
        ax.plot(r1, g_r1, label='dataset')
        ax.plot(r2, g_r2, label='inference', linestyle='--')
        ax.set_title(f'{pair[0]}-{pair[1]}')
        ax.set_xlabel('r (Å)')
        ax.set_ylabel('g(r)')
        ax.legend()
        ax.grid(True)
    
    plt.savefig(output_file)  # 保存
    plt.show()

# Example usage
if __name__ == "__main__":
    xyz_file1 = 'output.xyz'  # 最初のトラジェクトリ
    xyz_file2 = '../../../scripts/traj_output_TiO2SiO2/traj_300K_42_20250718_0.1.xyz'  # 2番目のトラジェクトリ
    box = [13.659831, 13.659831, 13.659831]  # ボックスサイズを調整
    
    results1 = compute_rdfs_with_mdanalysis(xyz_file1, box)
    results2 = compute_rdfs_with_mdanalysis(xyz_file2, box)
    
    plot_overlapped_rdfs(results1, results2)
