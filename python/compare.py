import time
import numpy as np
import creditmc_core as cpp
from reference import irb_asymptotic_loss, risk_metrics, simulate_losses_numpy

N_OBLIGORS = 1000
PD, LGD, EAD, RHO = 0.02, 0.45, 1.0, 0.15
N_SCENARIOS = 100_000
CONF = 0.999

pd = np.full(N_OBLIGORS, PD)
ead = np.full(N_OBLIGORS, EAD)
lgd = np.full(N_OBLIGORS, LGD)

t0 = time.perf_counter()
loss_np = simulate_losses_numpy(pd, ead, lgd, RHO, N_SCENARIOS, seed = 1)
t_np = time.perf_counter() - t0

t0 = time.perf_counter()
loss_cpp = cpp.simulate(pd, ead, lgd, RHO, N_SCENARIOS, seed = 1)
t_cpp = time.perf_counter() - t0

a = cpp.simulate(pd, ead, lgd, RHO, 20_000, seed = 7, n_threads = 1)
b = cpp.simulate(pd, ead, lgd, RHO, 20_000, seed = 7, n_threads = 4)
print(f"Thread-count reproducibility: {np.array_equal(a, b)}\n")

irb_var = N_OBLIGORS * EAD * irb_asymptotic_loss(PD, LGD, RHO, CONF)
expected_loss = N_OBLIGORS * EAD * PD * LGD
m_np, m_cpp = risk_metrics(loss_np, CONF), risk_metrics(loss_cpp, CONF)

print(f"{'metric':<14}{'Numpy':>12}{'C++':>12}")
for k in m_np:
    print(f"{k:<14}{m_np[k]:>12.3f}{m_cpp[k]:>12.3f}")
print(f"\nAnalytical EL.          : {expected_loss:.3f}")
print(f"IRB asymptotic 99.9% VaR: {irb_var:.3f}")
print(f"\nTime NumPy: {t_np:6.2f}s C++: {t_cpp:6.2f}s speedup: {t_np / t_cpp:5.1f}x")