"""Formal comparison of trends between non-overlapping age groups (and sexes).
Slopes from Prais-Winsten models (AR(1) errors); series treated as independent (non-overlapping groups).
Global test of equal slopes (parallelism of log-linear trends): Wald chi-square, df = k-1.
Pairwise differences: z-test, Holm-adjusted. Same for joinpoint AAPCs."""
import numpy as np, pandas as pd, itertools
from scipy import stats
from common import *
from trend import *

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj

def compare(est, labels):
    b = np.array([e[0] for e in est]); se = np.array([e[1] for e in est])
    w = 1 / se**2; bw = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - bw) ** 2); pg = stats.chi2.sf(q, len(b) - 1)
    pairs = list(itertools.combinations(range(len(b)), 2))
    rows = []
    for i, j in pairs:
        d = b[i] - b[j]; s = np.sqrt(se[i]**2 + se[j]**2)
        rows.append([labels[i], labels[j], 100 * d, 100 * (d - 1.96 * s), 100 * (d + 1.96 * s), 2 * stats.norm.sf(abs(d / s))])
    pw = pd.DataFrame(rows, columns=["a", "b", "diff_pp", "lo", "hi", "p"])
    pw["p_holm"] = holm(pw.p.values)
    return q, pg, pw

if __name__ == "__main__":
    b, pop = load()
    pd.set_option("display.width", 200)
    out = []
    for m in ["Deaths", "YLLs", "YLDs", "Prevalence", "Incidence"]:
        for kind in ["EAPC", "AAPC"]:
            est = []
            for a in BANDS:
                y = series(b, m, a).val
                if kind == "EAPC":
                    e = eapc_all(y); est.append((e["ar1_b"], e["ar1_se"]))
                else:
                    j = joinpoint(y); est.append((j["ar1"]["aapc_b"], j["ar1"]["aapc_se"]))
            q, pg, pw = compare(est, [LABEL[a] for a in BANDS])
            print(f"\n{m} {kind}: global test of equal trends across 4 age groups chi2={q:.1f}, df=3, p={pg:.2g}")
            print(pw.round(4).to_string(index=False))
            pw.insert(0, "kind", kind); pw.insert(0, "measure", m); pw["global_chi2"] = q; pw["global_p"] = pg
            out.append(pw)
    pd.concat(out).to_csv("out/age_comparisons.csv", index=False)
