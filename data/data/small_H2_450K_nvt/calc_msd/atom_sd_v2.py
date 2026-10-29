import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

timestep = 0.0005
dump_freq = 20

traj = read("../log.lammpstrj", index="500:")
num_atoms = len(traj[0])
atom_indices = list(range(num_atoms))

symbols = traj[0].get_chemical_symbols()  # ['Si','O','H', ...]

# 元素ごとの色を定義
element_colors = {'H': 'red', 'He': 'purple', 'Si': 'blue', 'O': 'green', 'Ti': 'orange'}

initial_positions = traj[0].get_positions()[atom_indices]
squared_displacements = np.array([
    np.sum((atoms.get_positions()[atom_indices] - initial_positions)**2, axis=1)
    for atoms in traj
])

time = np.arange(len(squared_displacements)) * timestep * dump_freq

plt.figure(figsize=(10, 6))
plotted_elements = set()

for i, atom_index in enumerate(atom_indices):
    elem = symbols[atom_index]
    label = elem if elem not in plotted_elements else None  # 凡例の重複防止
    plt.plot(time, squared_displacements[:, i],
             color=element_colors[elem], alpha=0.5, linewidth=0.8, label=label)
    plotted_elements.add(elem)

plt.xlabel("Time (ps)")
plt.ylabel("Squared Displacement (Å²)")
plt.yscale('log')
plt.legend()
plt.savefig("squared_displacement.png", dpi=600)
plt.show()
