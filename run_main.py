"""Main trend estimates (EAPC by three methods; joinpoint) for all outcomes and age groups."""
import json
import pandas as pd
from common import *
from trend import *
b, pop = load()
rows, jp = [], {}
for m in ["Deaths", "DALYs", "YLLs", "YLDs", "Prevalence", "Incidence"]:
    for a in AGES + OLD:
        y = series(b, m, a).val
        e = eapc_all(y)
        rows.append(dict(measure=m, age=a,
                         ols=e["ols"][0], ols_lo=e["ols"][1], ols_hi=e["ols"][2], ols_p=e["ols"][3],
                         pw=e["ar1"][0], pw_lo=e["ar1"][1], pw_hi=e["ar1"][2], pw_p=e["ar1"][3], rho=e["rho"],
                         hac_lo=e["hac"][1], hac_hi=e["hac"][2], hac_p=e["hac"][3],
                         dw=e["dw"], bp_p=e["bp_p"], sw_p=e["sw_p"], fitted_change=e["fitted_change"]))
        j = joinpoint(y)
        jp[f"{m}|{a}"] = {k: v for k, v in j.items() if k != "fitted"}
pd.DataFrame(rows).to_csv("out/eapc_main.csv", index=False)
json.dump(jp, open("out/joinpoint_main.json", "w"), default=float, indent=0)
print("ok")
