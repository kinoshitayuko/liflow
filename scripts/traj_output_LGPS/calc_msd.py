import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ase.io import read

def calculate_msd(traj, atom_indices, msd_time_window, msd_time_shift):
    n_traj = len(traj)
    traj_ini = 0
    traj_fin = msd_time_window + 1
    n_count = 0
    msd = []

    while traj_fin <= n_traj:
        n_count += 1
        traj_tmp = traj[traj_ini:traj_fin]
        initial_positions = traj_tmp[0].get_positions()[atom_indices]
        msd_tmp = [
            np.mean(np.sum((atoms.get_positions()[atom_indices] - initial_positions)**2, axis=1))
            for atoms in traj_tmp
        ]
        msd.append(msd_tmp)
        traj_ini += msd_time_shift
        traj_fin += msd_time_shift

    print(f"Number of trajectories counted: {n_count}")
    return np.mean(msd, axis=0)

def plot_msd(time, msd, filename):
    plt.figure()
    plt.plot(time, msd, label="MSD")
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD ($\mathrm{Å^2}$)")
    plt.legend()
    plt.savefig(filename, dpi=600)
    plt.show()

def calculate_diffusion_coefficient(msd, timestep, dump_freq, fit_start):
    x_fit = np.arange(len(msd))[fit_start:]
    y_fit = msd[fit_start:]    
    slope, intercept, _, _, _ = stats.linregress(x_fit, y_fit)
    D = slope / (timestep * dump_freq) / 6  # in the unit of A^2/ps
    D *= 1.0e-4  # in the unit of cm^2/s
    print(f"Diffusion coefficient: {D:.4e} cm^2/s")
    return slope, intercept, x_fit, D

def plot_msd_with_fit(time, msd, x_fit, slope, intercept, D, filename):
    plt.figure()
    plt.plot(time, msd, label="MSD")
    plt.plot(x_fit * timestep * dump_freq, x_fit * slope + intercept, linestyle="dashed", label=f"Fit (D={D:.4e} cm^2/s)")
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD ($\mathrm{Å^2}$)")
    plt.legend()
    plt.savefig(filename, dpi=600)
    plt.show()

"""Parameters"""
# Timestep size in MD simulation (ps)
timestep = 0.015
# Frequency for dumping trajectory
dump_freq = 1
# Time window for MSD calculation (i.e., timestep * dump_freq * msd_time_window)
msd_time_window = 67
# Shifting the time origin for MSD calculation (i.e., timestep * dump_freq * msd_time_shift)
msd_time_shift = 15
# Start of the fitting range (i.e., timestep * dump_freq * fit_start)
fit_start = 15

# Read trajectory
#traj = read("../log.lammpstrj", index="500:")
traj = read("positions_1400K.xyz", index=":")
num_atoms = len(traj[0]) # 167
atom_indices = [
    i for i, symbol in enumerate(traj[0].get_chemical_symbols()) if symbol == "Li"
]
print(f"number of Li atoms: {len(atom_indices)}")
print(f"target index (first 10): {atom_indices[:10]}")

# Calculate MSD
msd = calculate_msd(traj, atom_indices, msd_time_window, msd_time_shift)
time = np.arange(0, len(msd)) * timestep * dump_freq

# Plot MSD
plot_msd(time, msd, "msd_1400K.png")

# Calculate diffusion coefficient
slope, intercept, x_fit, D = calculate_diffusion_coefficient(msd, timestep, dump_freq, fit_start)

# Plot MSD with fit
plot_msd_with_fit(time, msd, x_fit, slope, intercept, D, "fit_1400K.png")