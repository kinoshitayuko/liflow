# import MDAnalysis as mda
# from MDAnalysis.analysis import rdf
# import numpy as np
# import matplotlib.pyplot as plt

# def compute_rdfs_with_mdanalysis(xyz_file, box, r_max=10.0, num_bins=100):
#     """
#     Compute partial RDFs for specified pairs using MDAnalysis.
#     Assumes periodic boundary conditions with box [Lx, Ly, Lz].
#     Returns a dictionary of (pair: (r, g_r)).
#     """
#     # Load the universe from XYZ file (treat as trajectory)
#     universe = mda.Universe(xyz_file, format='XYZ', topology_format='XYZ')
    
#     # Set box dimensions (periodic boundary conditions)
#     universe.dimensions = np.array([box[0], box[1], box[2], 90, 90, 90])  # Assuming orthorhombic box
    
#     # Define atom selections
#     selections = {
#         'Ti': universe.select_atoms('name Ti'),
#         'O': universe.select_atoms('name O'),
#         'Si': universe.select_atoms('name Si')
#     }
    
#     # Pairs to compute
#     pairs = [('Ti', 'Ti'), ('Ti', 'O'), ('Ti', 'Si'), ('Si', 'Si'), ('Si', 'O'), ('O', 'O')]
    
#     results = {}
#     for typeA, typeB in pairs:
#         groupA = selections[typeA]
#         groupB = selections[typeB]
        
#         # Skip if no atoms in group
#         if len(groupA) == 0 or len(groupB) == 0:
#             print(f"Skipping pair {typeA}-{typeB}: No atoms found.")
#             continue
        
#         # exclusion_block: homo-pairの場合に自己相関を除外 (1,1)で同じグループ内のペアをスキップ
#         exclusion = (1, 1) if typeA == typeB else None
        
#         # Compute RDF (averaged over all frames)
#         rdf_analysis = rdf.InterRDF(groupA, groupB, nbins=num_bins, range=(0.0, r_max))
#         rdf_analysis.run()
        
#         r = rdf_analysis.results.bins
#         g_r = rdf_analysis.results.rdf
        
#         results[(typeA, typeB)] = (r, g_r)
    
#     return results

# def plot_rdfs(results):
#     # 各RDFを別々の図としてプロット
#     for pair, (r, g_r) in results.items():
#         plt.figure(figsize=(6, 4))
#         plt.plot(r, g_r)
#         plt.title(f'Partial RDF: {pair[0]}-{pair[1]}')
#         plt.xlabel('r (Å)')
#         plt.ylabel('g(r)')
#         plt.grid(True)
#         plt.show()
    
#     # 全てのRDFを1つの図に小さなサブプロットとして配置 (2x3レイアウト)
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
#     for ax, (pair, (r, g_r)) in zip(axes.flat, results.items()):
#         ax.plot(r, g_r)
#         ax.set_title(f'{pair[0]}-{pair[1]}')
#         ax.set_xlabel('r (Å)')
#         ax.set_ylabel('g(r)')
#         ax.grid(True)
#     plt.show()
#     plt.savefig('RDF_TiO2SiO2_20250723_3.png')  # Save the combined plot as an image file

# # Example usage
# if __name__ == "__main__":
#     xyz_file = 'output.xyz'  # Replace with your actual XYZ file path
#     box = [13.659831, 13.659831, 13.659831]  # Adjust box sizes [Lx, Ly, Lz] based on your system
#     r_max = 10.0
#     num_bins = 100
    
#     results = compute_rdfs_with_mdanalysis(xyz_file, box, r_max, num_bins)
#     plot_rdfs(results)



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
        
        # デバッグ: 結果の確認
        print(f"Pair {typeA}-{typeB}: Max g(r) = {np.max(g_r)}")  # 異常値チェック
        
        results[(typeA, typeB)] = (r, g_r)
    
    return results

def plot_rdfs(results):
    # 各RDFを別々の図としてプロット
    for pair, (r, g_r) in results.items():
        plt.figure(figsize=(6, 4))
        plt.plot(r, g_r)
        plt.title(f'Partial RDF: {pair[0]}-{pair[1]}')
        plt.xlabel('r (Å)')
        plt.ylabel('g(r)')
        plt.grid(True)
        plt.show()
    
    # 全てのRDFを1つの図に小さなサブプロットとして配置 (2x3レイアウト)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (pair, (r, g_r)) in zip(axes.flat, results.items()):
        ax.plot(r, g_r)
        ax.set_title(f'{pair[0]}-{pair[1]}')
        ax.set_xlabel('r (Å)')
        ax.set_ylabel('g(r)')
        ax.grid(True)
    
    # 保存をshow前に実行
    plt.savefig('RDF_TiO2SiO2_20250723_3.png')  # Save the combined plot as an image file
    plt.show()

# Example usage
if __name__ == "__main__":
    xyz_file = 'output.xyz'  # Replace with your actual XYZ file path
    box = [13.659831, 13.659831, 13.659831]  # Adjust box sizes [Lx, Ly, Lz] based on your system
    r_max = 10.0
    num_bins = 100
    
    results = compute_rdfs_with_mdanalysis(xyz_file, box, r_max, num_bins)
    plot_rdfs(results)
