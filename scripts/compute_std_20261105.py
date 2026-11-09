import numpy as np

# D1=1.58e-04
# D2=1.2729e-04
# D3=1.0405e-04
# D4=1.3196e-04
# D5=1.2415e-04

D1=3.36e-04
D2=2.4654e-04
D3=2.2859e-04
D4=3.2348e-04
D5=2.7955e-04


# 各シードから求めた拡散係数を、同じ単位で入力
D = np.array([D1, D2, D3, D4, D5], dtype=float)

D_mean = np.mean(D)
D_std = np.std(D, ddof=1)

print(f"D = {D_mean:.3e} ± {D_std:.3e} cm^2/s")