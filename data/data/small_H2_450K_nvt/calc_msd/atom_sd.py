import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from matplotlib.cm import get_cmap

def calculate_squared_displacement(traj, atom_indices):
    initial_positions = traj[0].get_positions()[atom_indices]
    squared_displacements = [
        np.sum((atoms.get_positions()[atom_indices] - initial_positions)**2, axis=1)
        for atoms in traj
    ]
    return np.array(squared_displacements)

def plot_squared_displacement(time, squared_displacements, atom_indices, filename):
    plt.figure()
    cmap = get_cmap('viridis', len(atom_indices))
    for i, atom_index in enumerate(atom_indices):
        plt.plot(time, squared_displacements[:, i], color=cmap(i))
    plt.xlabel("Time (ps)")
    plt.ylabel("Squared Displacement ($\mathrm{Å^2}$)")
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.savefig(filename, dpi=600)
    plt.show()

"""Parameters"""
# Timestep size in MD simulation (ps)
timestep = 0.0005
# Frequency for dumping trajectory
dump_freq = 2000

# Read trajectory
traj = read("../log.lammpstrj", index="500:")
num_atoms = len(traj[0])
atom_indices = list(range(num_atoms))  # All atom indices

# Calculate squared displacements
squared_displacements = calculate_squared_displacement(traj, atom_indices)
time = np.arange(0, len(squared_displacements)) * timestep * dump_freq

# Plot squared displacements
plot_squared_displacement(time, squared_displacements, atom_indices, "squared_displacement.png")