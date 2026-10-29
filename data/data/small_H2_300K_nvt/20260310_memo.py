# import numpy as np
# data = np.load("positions_600K.npz", allow_pickle=True)
# print(data.files)  # → ['small_H2_300K_nvt'] や ['positions'] など

# import numpy as np
# d = np.load("atomic_numbers.npy", allow_pickle=True).item()
# print(d.keys())

import numpy as np

# atomic_numbers の修正
an = np.load("atomic_numbers.npy", allow_pickle=True).item()
an['small_H2_450K_nvt'] = an['small_H2_300K_nvt']
an['small_H2_600K_nvt'] = an['small_H2_300K_nvt']
np.save("atomic_numbers.npy", an)
print("atomic_numbers keys:", an.keys())

# lattice の修正
lat = np.load("lattice.npy", allow_pickle=True).item()
lat['small_H2_450K_nvt'] = lat['small_H2_300K_nvt']
lat['small_H2_600K_nvt'] = lat['small_H2_300K_nvt']
np.save("lattice.npy", lat)
print("lattice keys:", lat.keys())
