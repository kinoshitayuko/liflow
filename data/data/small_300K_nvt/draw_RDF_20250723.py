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
    universe.dimensions = np.array([box[0], box[1], box[2], 90, 90, 90])  # Assuming orthorhombic box
    
    # Define atom selections
    selections = {
        'Ti': universe.select_atoms('name Ti'),
        'O': universe.select_atoms('name O'),
        'Si': universe.select_atoms('name Si')
    }
    
    # Pairs to compute
    pairs = [('Ti', 'Ti'), ('Ti', 'O'), ('Ti', 'Si'), ('O', 'O'), ('Si', 'O'), ('Si', 'Si')]
    
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
        rdf_analysis = rdf.InterRDF(groupA, groupB, nbins=num_bins, range=(0.0, r_max))
        rdf_analysis.run()
        
        r = rdf_analysis.results.bins
        g_r = rdf_analysis.results.rdf
        
        results[(typeA, typeB)] = (r, g_r)
    
    return results

def plot_rdfs(results):
    plt.figure(figsize=(12, 8))
    for pair, (r, g_r) in results.items():
        label = f'{pair[0]}-{pair[1]}'
        plt.plot(r, g_r, label=label)
    
    plt.xlabel('r (Å)')
    plt.ylabel('g(r)')
    plt.title('Partial RDFs using MDAnalysis')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('RDF_TiO2SiO2_20250723.png')

# Example usage
if __name__ == "__main__":
    xyz_file = 'output.xyz'  # Replace with your actual XYZ file path
    box = [13.659831, 13.659831, 13.659831]  # Adjust box sizes [Lx, Ly, Lz] based on your system
    r_max = 10.0
    num_bins = 100
    
    results = compute_rdfs_with_mdanalysis(xyz_file, box, r_max, num_bins)
    plot_rdfs(results)
