import os

import matplotlib.pyplot as plt
import numpy as np

import creditmc_core as cpp

N_OBLIGORS = 1000
EAD = 1.0
PD, LGD = 0.02, 0.45
N_SCENARIOS = 1_000_000
N_SEEDS = 5
CONF = 0.999
RHOS = np.round(np.arange(0.05, 0.3001, 0.025), 3)

pd = np.full(N_OBLIGORS, PD)
ead = np.full(N_OBLIGORS, EAD)
lgd = np.full(N_OBLIGORS, LGD)
total_ead = N_OBLIGORS * EAD

def irb_var(rho):
    return total_ead * cpp.irb_asymptotic_loss(PD, LGD, rho, CONF)

mc_var = []
mc_err = []
for i, rho in enumerate(RHOS):
    runs = []
    for k in range(N_SEEDS):
        losses = cpp.simulate(pd, ead, lgd, float(rho), N_SCENARIOS, seed = 1000 * i + k)
        runs.append(np.quantile(losses, CONF))
    mc_var.append(np.mean(runs))
    mc_err.append(np.std(runs, ddof = 1) / np.sqrt(N_SEEDS))
    print(f"rho {rho:.3f}: MC VaR {mc_var[-1]:7.2f} +/- {mc_err[-1]:.2f}     "
          f"IRB VaR {irb_var(rho):7.2f}    ratio {mc_var[-1] / irb_var(rho):.3f}")

mc_var = np.array(mc_var)
mc_err = np.array(mc_err)
irb_at_points = np.array([irb_var(r) for r in RHOS])
rho_fine = np.linspace(RHOS[0], RHOS[-1], 200)
irb_fine = np.array([irb_var(r) for r in rho_fine])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

ax1.plot(rho_fine, irb_fine, linestyle = "--", color = "grey", label = "IRB VaR (99.9%)")
ax1.errorbar(RHOS, mc_var, yerr = mc_err, marker = "o", capsize = 3, label = f"Monte Carlo VaR (mean of {N_SEEDS} runs)")
ax1.set_xlabel("Asset correlation (rho)")
ax1.set_ylabel("99.9% VaR (exposure units)")
ax1.set_title("Tail loss rises steeply with correlation")
ax1.grid(alpha = 0.3)
ax1.legend()

ax2.errorbar(RHOS, mc_var / irb_at_points, yerr = mc_err / irb_at_points, marker = "o", capsize = 3, color = "tab:orange")
ax2.axhline(1.0, linestyle = "--", color = "grey")
ax2.set_xlabel("Asset correlation (rho)")
ax2.set_ylabel("Monte Carlo VaR / IRB VaR")
ax2.set_title("Granularity gap shrinks as correlation rises")
ax2.grid(alpha = 0.3)

fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
os.makedirs(out_dir, exist_ok = True)
out_path = os.path.join(out_dir, "correlation.png")
fig.savefig(out_path, dpi = 150)
print(f"Saved {os.path.normpath(out_path)}")