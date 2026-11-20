import numpy as np
from scipy.stats import norm

def simulate_losses_numpy(pd, ead, lgd, rho, n_scenarios, seed = 42, chunk = 2000):
    pd, ead, lgd = (np.asarray(x, dtype = float) for x in (pd, ead, lgd))
    rng = np.random.default_rng(seed)

    threshold = norm.ppf(pd)
    loss_if_default = ead * lgd
    n_obligors = len(pd)

    losses = np.empty(n_scenarios)
    for start in range(0, n_scenarios, chunk):
        m = min(chunk, n_scenarios - start)
        z = rng.standard_normal((m, 1))
        eps = rng.standard_normal((m, n_obligors))
        asset = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * eps
        defaults = asset < threshold
        losses[start:start + m] = defaults @ loss_if_default
    return losses

def irb_asymptotic_loss(pd, lgd, rho, confidence = 0.999):
    cond_pd = norm.cdf((norm.ppf(pd) + np.sqrt(rho) * norm.ppf(confidence)) / np.sqrt(1.0 - rho))
    return lgd * cond_pd

def risk_metrics(losses, confidence = 0.999):
    el = losses.mean()
    var = np.quantile(losses, confidence)
    es = losses[losses >= var].mean()
    return {"EL" : el, "VaR" : var, "ES": es, "UL (VaR-EL)": var - el}

