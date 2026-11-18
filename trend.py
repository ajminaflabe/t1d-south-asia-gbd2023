"""Trend models for the revision.

EAPC: ln(rate) = a + b*year, fitted three ways
  - OLS (as in the submitted paper; 95% CI with z = 1.96)
  - Prais-Winsten GLS with AR(1) errors (iterative, first observation retained): primary
  - OLS with Newey-West (HAC) standard errors, 2 lags
Diagnostics on the OLS residuals: Durbin-Watson, Breusch-Pagan, Shapiro-Wilk.

Joinpoint: continuous piecewise log-linear model, grid search over integer years,
>= 2 observations from either end and between joinpoints, max 2 joinpoints, BIC.
For the selected model, APCs and AAPC are re-estimated by Prais-Winsten (AR(1) errors).
"""
import itertools
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from pw import prais_winsten

YEARS = np.arange(2000, 2024).astype(float)


def pct(b):
    return 100 * (np.exp(b) - 1)


# ---------------- EAPC ----------------
def eapc_all(y_rate):
    y = np.log(np.asarray(y_rate, float))
    X = sm.add_constant(YEARS - 2000)
    ols = sm.OLS(y, X).fit()
    b, se = ols.params[1], ols.bse[1]
    out = {"ols": (pct(b), pct(b - 1.96 * se), pct(b + 1.96 * se), ols.pvalues[1])}
    ar = prais_winsten(y, X)
    tcrit = stats.t.ppf(0.975, ar.df_resid)
    out["ar1"] = (pct(ar.params[1]), pct(ar.params[1] - tcrit * ar.bse[1]),
                  pct(ar.params[1] + tcrit * ar.bse[1]), ar.pvalues[1])
    out["ar1_b"], out["ar1_se"], out["rho"] = ar.params[1], ar.bse[1], float(ar.rho)
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    out["hac"] = (pct(hac.params[1]), pct(hac.params[1] - 1.96 * hac.bse[1]),
                  pct(hac.params[1] + 1.96 * hac.bse[1]), hac.pvalues[1])
    r = ols.resid
    out["dw"] = durbin_watson(r)
    out["bp_p"] = het_breuschpagan(r, X)[1]
    out["sw_p"] = stats.shapiro(r).pvalue
    out["fitted_change"] = 100 * (np.exp(b * 23) - 1)
    return out


# ---------------- Joinpoint ----------------
def design(x, taus):
    cols = [np.ones_like(x), x - x.min()]
    for t in taus:
        cols.append(np.clip(x - t, 0, None))
    return np.column_stack(cols)


def candidate_sets(x, k):
    grid = range(int(x.min()) + 2, int(x.max()) - 2 + 1)
    for combo in itertools.combinations(grid, k):
        if all(b - a >= 3 for a, b in zip(combo, combo[1:])):
            yield combo


_CANDS = {k: ([()] if k == 0 else list(candidate_sets(YEARS, k))) for k in range(0, 4)}
_DESIGNS = {k: [(t, design(YEARS, t)) for t in _CANDS[k]] for k in _CANDS}


def select(y, max_jp=2):
    """Return chosen joinpoints by BIC (OLS fits, as in the submitted analysis)."""
    n = len(y)
    best_all = []
    for k in range(0, max_jp + 1):
        best = None
        for taus, X in _DESIGNS[k]:
            beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ beta) ** 2))
            if best is None or sse < best[1]:
                best = (taus, sse)
        bic = n * np.log(best[1] / n) + (2 + 2 * k) * np.log(n)
        best_all.append((bic, best[0]))
    return min(best_all)[1]


def contrasts(taus, nparam):
    edges = [2000] + list(taus) + [2023]
    cs = []
    for s in range(len(edges) - 1):
        c = np.zeros(nparam)
        c[1] = 1
        c[2:2 + s] = 1
        cs.append((edges[s], edges[s + 1], c))
    w = np.zeros(nparam)
    for a, e, c in cs:
        w += c * (e - a) / 23
    return cs, w


def joinpoint(y_rate, max_jp=2, taus=None):
    y = np.log(np.asarray(y_rate, float))
    if taus is None:
        taus = select(y, max_jp)
    X = design(YEARS, taus)
    ols = sm.OLS(y, X).fit()
    ar = prais_winsten(y, X)
    cs, w = contrasts(taus, X.shape[1])
    res = {"taus": tuple(int(t) for t in taus), "dw": durbin_watson(ols.resid),
           "rho": float(ar.rho), "fitted": np.exp(X @ ols.params)}
    for name, m in (("ols", ols), ("ar1", ar)):
        cov = m.cov_params()
        tcrit = stats.t.ppf(0.975, m.df_resid)
        segs = []
        for a, e, c in cs:
            b = c @ m.params
            se = np.sqrt(c @ cov @ c)
            p = 2 * stats.t.sf(abs(b / se), m.df_resid)
            segs.append((a, e, pct(b), pct(b - tcrit * se), pct(b + tcrit * se), p))
        b = w @ m.params
        se = np.sqrt(w @ cov @ w)
        res[name] = {"segs": segs,
                     "aapc": (pct(b), pct(b - tcrit * se), pct(b + tcrit * se),
                              2 * stats.t.sf(abs(b / se), m.df_resid)),
                     "aapc_b": b, "aapc_se": se}
    return res
