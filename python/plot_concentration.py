import os

import matplotlib.pyplot as plt
import numpy as np

import creditmc_core as cpp

N_OBLIGORS = 1000
N_BIG = 10
TOTAL_EAD = 1000.0
PD, LGD, RHO = 0.02, 0.45, 0.15
N_SCENARIOS = 1_000_000
CONF = 0.999
TOP_SHARES = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

def make_exposures(top_share):
    n_small = N_OBLIGORS - N_BIG
    big = TOTAL_EAD * top_share / N_BIG
    small = TOTAL_EAD * (1.0 - top_share) / n_small
    return np.concatenate([np.full(N_BIG, big), np.full(n_small, small)])

pd = np.full(N_OBLIGORS, PD)
lgd = np.full(N_OBLIGORS, LGD)
irb_var = TOTAL_EAD * cpp.irb_asymptotic_loss(PD, LGD, RHO, CONF)

mc_var = []
for share in TOP_SHARES:
    losses = cpp.simulate(pd, make_exposures(share), lgd, RHO, N_SCENARIOS, seed = 1)
    mc_var.append(np.quantile(losses, CONF))
    print(f"top-10 share {share:.0%}: VaR {mc_var[-1]:.2f}")

x = np.array(TOP_SHARES) * 100
mc_var = np.array(mc_var)
irb_line = np.full_like(mc_var, irb_var)

fig, ax = plt.subplots(figsize = (8,5))
ax.plot(x, mc_var, marker = 'o', label = "Monte Carlo VaR (99.9%)")
ax.plot(x, irb_line, linestyle = "--", color = "grey", label = f"IRB VaR ({irb_var:.1f})")
ax.fill_between(x, irb_line, mc_var, where = mc_var > irb_line, interpolate = True, alpha = 0.2, label = "Concentration add-on")
ax.set_xlabel("Share of total exposure held by the 10 largest loans (%)")
ax.set_ylabel("99.9% VaR (exposure units)")
ax.set_title("IRB formula vs Monte Carlo: the cost of concentration")
ax.grid(alpha = 0.3)
ax.legend()
fig.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "docs")
os.makedirs(out_dir, exist_ok = True)
out_path = os.path.join(out_dir, "concentration.png")
fig.savefig(out_path, dpi = 150)
print(f"Saved {os.path.normpath(out_path)}")