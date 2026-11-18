"""Sex-specific and country-level trends (EAPC, Prais-Winsten AR(1) errors) for deaths, prevalence and incidence."""
import numpy as np, pandas as pd
from common import *
from trend import *
from compare import compare
b, pop = load()
rows = []
for loc in LOCS:
    for sex in ["Both", "Male", "Female"]:
        if loc != "South Asia" and sex != "Both":
            continue
        for m in ["Deaths", "Prevalence", "Incidence"]:
            for a in BANDS + ["<20 years"]:
                s = series(b, m, a, loc=loc, sex=sex)
                e = eapc_all(s.val)
                v = s.set_index("year").val
                rows.append(dict(location=loc, sex=sex, measure=m, age=a,
                                 rate2000=v[2000], rate2023=v[2023],
                                 eapc=e["ar1"][0], lo=e["ar1"][1], hi=e["ar1"][2], p=e["ar1"][3],
                                 b=e["ar1_b"], se=e["ar1_se"], chg_2018_2021=100 * (v[2021] / v[2018] - 1)))
d = pd.DataFrame(rows)
d.to_csv("out/strata_eapc.csv", index=False)
pd.set_option("display.width", 220)
def show(q):
    print(q[["location", "sex", "measure", "age", "rate2000", "rate2023", "eapc", "lo", "hi", "p", "chg_2018_2021"]].round(3).to_string(index=False))
print("=== SEX (South Asia) ==="); show(d[(d.location == "South Asia") & (d.sex != "Both")])
# male vs female slope difference
print("\nMale minus female EAPC difference (pp), z-test")
for m in ["Deaths", "Prevalence", "Incidence"]:
    for a in BANDS + ["<20 years"]:
        M = d[(d.location == "South Asia") & (d.sex == "Male") & (d.measure == m) & (d.age == a)].iloc[0]
        F = d[(d.location == "South Asia") & (d.sex == "Female") & (d.measure == m) & (d.age == a)].iloc[0]
        q, pg, pw = compare([(M.b, M.se), (F.b, F.se)], ["M", "F"])
        print(f"  {m:10s} {a:11s} M {M.eapc:+.2f}  F {F.eapc:+.2f}  diff {pw.diff_pp[0]:+.2f} ({pw.lo[0]:+.2f},{pw.hi[0]:+.2f}) p={pw.p[0]:.3f}")
print("\n=== COUNTRIES (both sexes) ==="); show(d[(d.location != "South Asia")])
print("\nDeath-rate age gradient by country: 5-9 minus 15-19 EAPC, and global test across 4 bands")
for loc in LOCS:
    est = [(r.b, r.se) for a in BANDS for r in [d[(d.location == loc) & (d.sex == "Both") & (d.measure == "Deaths") & (d.age == a)].iloc[0]]]
    q, pg, pw = compare(est, [LABEL[a] for a in BANDS])
    r = pw[(pw.a == "5–9") & (pw.b == "15–19")].iloc[0]
    print(f"  {loc:11s} global chi2={q:6.1f} p={pg:.2g} | 5-9 minus 15-19 = {r.diff_pp:+.2f} ({r.lo:+.2f},{r.hi:+.2f}) p={r.p:.3g}")
