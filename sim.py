"""Sensitivity of trend estimates to GBD uncertainty, simulated from the 95% UIs.
GBD draws are not available from the Results Tool, so each year's log-rate is drawn from a
normal distribution centred on ln(estimate) with SD = (ln upper - ln lower)/3.92. Errors are
correlated across years with an AR(1) structure (rho_u = 0, 0.5, 0.9) and drawn independently
across age groups. 2000 simulations per scenario."""
import numpy as np, pandas as pd
from common import *
from trend import *
rng = np.random.default_rng(20261118)
NS = 2000
b, pop = load()

def draws(s, rho):
    mu = np.log(s.val.values); sd = (np.log(s.upper.values) - np.log(s.lower.values)) / 3.92
    z = np.empty((NS, 24)); z[:, 0] = rng.standard_normal(NS)
    for t in range(1, 24):
        z[:, t] = rho * z[:, t - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(NS)
    return mu + z * sd

X = np.column_stack([np.ones(24), YEARS - 2000])
XJ = design(YEARS, (2018, 2021))
rows = []
res = {}
for m in ["Deaths", "Prevalence"]:
    for rho in [0.0, 0.5, 0.9]:
        for a in BANDS + ["<20 years"]:
            Y = draws(series(b, m, a), rho)
            slope = np.linalg.lstsq(X, Y.T, rcond=None)[0][1]
            eapc = pct(slope)
            bj = np.linalg.lstsq(XJ, Y.T, rcond=None)[0]
            mid = pct(bj[1] + bj[2])          # APC 2018-2021 with joinpoints fixed at 2018, 2021
            res[(m, rho, a)] = eapc
            rows.append(dict(measure=m, rho=rho, age=a,
                             eapc_med=np.median(eapc), eapc_lo=np.percentile(eapc, 2.5), eapc_hi=np.percentile(eapc, 97.5),
                             p_eapc_neg=(eapc < 0).mean(), p_mid_pos=(mid > 0).mean(),
                             mid_med=np.median(mid), mid_lo=np.percentile(mid, 2.5), mid_hi=np.percentile(mid, 97.5)))
        e = {a: res[(m, rho, a)] for a in BANDS}
        rows.append(dict(measure=m, rho=rho, age="P(5-9 faster decline than 15-19)", eapc_med=(e["5-9 years"] < e["15-19 years"]).mean()))
        rows.append(dict(measure=m, rho=rho, age="P(5-9 fastest of four bands)",
                         eapc_med=np.mean(np.argmin(np.vstack([e[a] for a in BANDS]), axis=0) == 1)))
        rows.append(dict(measure=m, rho=rho, age="P(15-19 slowest of four bands)",
                         eapc_med=np.mean(np.argmax(np.vstack([e[a] for a in BANDS]), axis=0) == 3)))
d = pd.DataFrame(rows)
d.to_csv("out/ui_simulation.csv", index=False)
pd.set_option("display.width", 250)
print(d.round(3).to_string(index=False))
