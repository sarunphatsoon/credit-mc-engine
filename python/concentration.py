import numpy as np
import creditmc_core as cpp

N_OBLIGORS = 1000
N_BIG = 10
TOTAL_EAD = 1000.0
PD, LGD, RHO = 0.02, 0.45, 0.15
N_SCENARIOS = 1_000_000
CONF = 0.999
TOP_SHARES = [0.01, 0.10, 0.20, 0.30, 0.40]

def make_exposures(top_share):
    n_small = N_OBLIGORS - N_BIG
    big = TOTAL_EAD * top_share / N_BIG
    small = TOTAL_EAD * (1.0 - top_share) / n_small
    return np.concatenate([np.full(N_BIG, big), np.full(n_small, small)])

pd = np.full(N_OBLIGORS, PD)
lgd = np.full(N_OBLIGORS, LGD)

irb_var = TOTAL_EAD * cpp.irb_asymptotic_loss(PD, LGD, RHO, CONF)
expected_loss = TOTAL_EAD * PD * LGD

print(f"IRB VaR: {irb_var:.2f}  Analytical EL: {expected_loss:.2f}\n")
print(f"{'top-10 share':>12}{'largest loan':>14}{'EL':>8}{'MC VaR':>10}{'MC/IRB':>9}{'ES':>9}")

for share in TOP_SHARES:
    ead = make_exposures(share)
    losses = cpp.simulate(pd, ead, lgd, RHO, N_SCENARIOS, seed = 1)
    var = np.quantile(losses, CONF)
    es = losses[losses >= var].mean()
    print(f"{share:>12.0%}{ead.max():>14.1f}{losses.mean():>8.2f}{var:>10.2f}{var / irb_var:>9.3f}{es:>9.2f}")