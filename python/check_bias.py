import numpy as np
import creditmc_core as cpp
from reference import irb_asymptotic_loss, simulate_losses_numpy

N, PD, LGD, RHO = 1000, 0.02, 0.45, 0.15
pd = np.full(N, PD)
ead = np.ones(N)
lgd = np.full(N, LGD)
irb = N * irb_asymptotic_loss(PD, LGD, RHO, 0.999)

var_np, var_cpp = [], []
for seed in range(5):
    l_np = simulate_losses_numpy(pd, ead, lgd, RHO, 500_000, seed = seed)
    l_cpp = cpp.simulate(pd, ead, lgd, RHO, 500_000, seed = seed)
    var_np.append(np.quantile(l_np, 0.999))
    var_cpp.append(np.quantile(l_cpp, 0.999))

print(f"IRB VaR.         : {irb:.2f}")
print(f"NumPy VaR by seed: {np.round(var_np, 2)} mean {np.mean(var_np):.2f} ")
print(f"C++ VaR by seed  : {np.round(var_cpp, 2)} mean {np.mean(var_cpp):.2f}")
print(f"Mean ratio C++/Numpy: {np.mean(var_cpp) / np.mean(var_np):.3f}")