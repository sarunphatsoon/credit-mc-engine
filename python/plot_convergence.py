import os 

import matplotlib.pyplot as plt
import numpy as np

import creditmc_core as cpp

TOTAL_EAD = 1000.0
PD, LGD, RHO = 0.02, 0.45, 0.15
N_SCENARIOS = 1_000_000
N_SEEDS = 5
CONF = 0.999
OBLIGOR_COUNTS = [250, 500, 1000, 2000, 4000]

irb_var = TOTAL_EAD * cpp.irb_asymptotic_loss(PD, LGD, RHO, CONF)

ratio = []
ratio_err = []
for i, n in enumerate(OBLIGOR_COUNTS):
    pd = np.full(n, PD)
    ead = np.full(n, TOTAL_EAD / n)
    lgd = np.full(n, LGD)
    runs = []
    for k in range(N_SEEDS):
        losses = cpp.simulate(pd, ead, lgd, RHO, N_SCENARIOS, seed = 1000 * i + k)
        runs.append(np.quantile(losses, CONF))
    ratio.append(np.mean(runs) / irb_var)
    ratio_err.append(np.std(runs, ddof = 1) / np.sqrt(N_SEEDS) / irb_var)
    print(f"N = {n:5d}: MC VaR {np.mean(runs):7.2f}   IRB VaR {irb_var:7.2f}     "
          f"ratio {ratio[-1]:.4f} +/- {ratio_err[-1]:.4f}")

n_arr = np.array(OBLIGOR_COUNTS, dtype = float)
ratio = np.array(ratio)
ratio_err = np.array(ratio_err)
excess_pct = (ratio - 1.0) * 100
excess_err_pct = ratio_err * 100

inv_n = 1.0 / n_arr
slope = np.sum(inv_n * excess_pct) / np.sum(inv_n ** 2)
inv_fine = np.linspace(0, inv_n.max(), 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

ax1.errorbar(n_arr, ratio, yerr = ratio_err, marker = "o", capsize = 3)
ax1.axhline(1.0, linestyle = "--", color = "grey", label = "IRB (infinitely many obligors)")
ax1.set_xscale("log")
ax1.set_xticks(OBLIGOR_COUNTS)
ax1.set_xticklabels([str(n) for n in OBLIGOR_COUNTS])
ax1.minorticks_off()
ax1.set_xlabel("Number of obligors N (total exposure fixed)")
ax1.set_ylabel("Monte Carlo VaR / IRB VaR")
ax1.set_title("Monte Carlo VaR vs IRB as N grows")
ax1.grid(alpha = 0.3)
ax1.legend()

ax2.errorbar(inv_n, excess_pct, yerr = excess_err_pct, marker = 'o', capsize = 3, linestyle = "none", label = "Monte Carlo")
ax2.plot(inv_fine, slope * inv_fine, linestyle = "--", color = "grey", label = "Line through origin (fit)")
ax2.set_xlabel("1/N")
ax2.set_ylabel("Excess over IRB (%)")
ax2.set_title("Excess over IRB vs 1/N")
ax2.grid(alpha = 0.3)
ax2.legend()

fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
os.makedirs(out_dir, exist_ok = True)
out_path = os.path.join(out_dir, "convergence.png")
fig.savefig(out_path, dpi = 150)
print(f"Saved {os.path.normpath(out_path)}")