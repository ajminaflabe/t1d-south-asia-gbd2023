"""Assemble cell contents for main Tables 1-2 and Appendix A tables -> out/tables_rev.json"""
import json, itertools
import numpy as np, pandas as pd
from common import *
from trend import eapc_all, joinpoint
from compare import compare

b, pop = load()
popf = pd.read_csv("data/pop_full.csv")
MS = "−"


def n(v, d=0):
    return f"{v:,.{d}f}".replace("-", MS)


def sg(v, d=2):
    s = f"{v:+.{d}f}".replace("-", MS)
    return s.replace("+", "+") if v > 0 else s.replace("+", "")


def ci(e, lo, hi, d=2):
    return f"{sg(e, d)} ({sg(lo, d)} to {sg(hi, d)})"


def ui(v, lo, hi, d):
    return f"{n(v, d)} ({n(lo, d)}–{n(hi, d)})"


def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def row(meas, age, year, metric="Rate", loc="South Asia", sex="Both"):
    s = series(b, meas, age, loc=loc, sex=sex, metric=metric)
    return s[s.year == year].iloc[0]


T = {}
AG = AGES
# ---------------- Table 1 ----------------
rows = []
for a in AG:
    for y in (2000, 2023):
        p = popf[(popf.location_name == "South Asia") & (popf.sex_name == "Both") & (popf.age_name == a) & (popf.year == y)].val.iloc[0]
        dn = row("Deaths", a, y, "Number"); dr = row("Deaths", a, y)
        pn = row("Prevalence", a, y, "Number"); pr = row("Prevalence", a, y)
        ir = row("Incidence", a, y)
        rows.append([LABEL[a] if y == 2000 else "", str(y), n(p / 1e6, 1),
                     ui(dn.val, dn.lower, dn.upper, 0), ui(dr.val, dr.lower, dr.upper, 3),
                     ui(pn.val, pn.lower, pn.upper, 0), ui(pr.val, pr.lower, pr.upper, 1),
                     ui(ir.val, ir.lower, ir.upper, 2)])
tot = popf[(popf.location_name == "South Asia") & (popf.sex_name == "Both") & (popf.age_name == "All ages")].set_index("year").val
T["t1"] = dict(
    caption="**Table 1.** Population and type 1 diabetes burden among children and adolescents in South Asia by age group, 2000 and 2023.",
    head=["Age group (years)", "Year", "Population, millions", "Deaths, n (95% UI)", "Death rate (95% UI)",
          "Prevalent cases, n (95% UI)", "Prevalence (95% UI)", "Incidence rate (95% UI)"],
    widths=[1300, 600, 1100, 1700, 1700, 2100, 1800, 1700], rows=rows, group=2,
    notes=[f"Rates are per 100,000 population; death and incidence rates are per year. Rates for the 5-year age groups are age-specific; rates for the under-20 group are crude. The total population of the region (all ages) was {n(tot[2000] / 1e6, 1)} million in 2000 and {n(tot[2023] / 1e6, 1)} million in 2023. Annual population denominators are given in Appendix A, Table A.1.",
           "UI, uncertainty interval. Source: Global Burden of Disease Study 2023."])

# ---------------- Table 2 ----------------
OUTS = ["Deaths", "YLLs", "YLDs", "Prevalence", "Incidence"]
eap = {(m, a): eapc_all(series(b, m, a).val) for m in OUTS for a in AG}
rows = []
for a in AG:
    rows.append([LABEL[a]] + [ci(*eap[(m, a)]["ar1"][:3]) for m in OUTS])
glob = []
for m in OUTS:
    q, pg, _ = compare([(eap[(m, a)]["ar1_b"], eap[(m, a)]["ar1_se"]) for a in BANDS], BANDS)
    glob.append(pv(pg))
rows.append(["Test of equal trends across the four 5-year age groups, p^a^"] + glob)
T["t2"] = dict(
    caption="**Table 2.** Estimated annual percentage change in type 1 diabetes death, YLL, YLD, prevalence and incidence rates by age group, South Asia, 2000–2023.",
    head=["Age group (years)", "Deaths, EAPC % (95% CI)", "YLLs, EAPC % (95% CI)", "YLDs, EAPC % (95% CI)",
          "Prevalence, EAPC % (95% CI)", "Incidence, EAPC % (95% CI)"],
    widths=[2600, 2000, 2000, 2000, 2000, 2000], rows=rows, group=None,
    notes=["EAPCs were estimated by log-linear regression of each annual rate on calendar year (24 observations), using Prais–Winsten estimation to allow for first-order autoregressive errors. Prevalence, incidence and YLD rates did not change linearly over time, so their EAPCs summarise the whole period only; their joinpoint models are given in Appendix A, Table A.3.",
           "^a^ Wald test of equal slopes across the under-5, 5–9, 10–14 and 15–19 year groups (3 degrees of freedom). Pairwise comparisons are given in Appendix A, Table A.5.",
           "CI, confidence interval; EAPC, estimated annual percentage change; YLD, year lived with disability; YLL, year of life lost. Source: Global Burden of Disease Study 2023."])

# ---------------- A.1 annual population ----------------
pa = popf[(popf.location_name == "South Asia") & (popf.sex_name == "Both")].pivot(index="year", columns="age_name", values="val")
cols = ["<5 years", "5-9 years", "10-14 years", "15-19 years", "<20 years", "All ages"]
T["a1"] = dict(
    caption="**Table A.1.** Annual population of South Asia by age group, both sexes, 2000–2023 (millions).",
    head=["Year", "Under 5", "5–9", "10–14", "15–19", "Under 20", "All ages"],
    widths=[1200] + [1600] * 6,
    rows=[[str(y)] + [n(pa.loc[y, c] / 1e6, 1) for c in cols] for y in pa.index],
    notes=["Source: Global Burden of Disease Study 2023 population estimates."])

# ---------------- A.2 EAPC methods and diagnostics ----------------
rows = []
for m in OUTS:
    for a in AG:
        e = eap[(m, a)]
        rows.append([m if a == AG[0] else "", LABEL[a], ci(*e["ols"][:3]), ci(*e["ar1"][:3]), f"{e['rho']:.2f}",
                     ci(e["ols"][0], *e["hac"][1:3]), f"{e['dw']:.2f}", pv(e["bp_p"]), pv(e["sw_p"])])
T["a2"] = dict(
    caption="**Table A.2.** Estimated annual percentage change by estimation method, with residual diagnostics, by outcome and age group, South Asia, 2000–2023.",
    head=["Outcome", "Age group (years)", "OLS, EAPC % (95% CI)", "Prais–Winsten, EAPC % (95% CI)", "AR(1) ρ",
          "Newey–West SE, EAPC % (95% CI)", "Durbin–Watson", "Breusch–Pagan p", "Shapiro–Wilk p"],
    widths=[1200, 1100, 2000, 2000, 800, 2000, 1000, 1000, 1000], rows=rows, group=5,
    notes=["OLS: ordinary least squares, as in the original submission. Prais–Winsten: generalised least squares with first-order autoregressive (AR(1)) errors, retaining the first observation; this is the primary analysis. Newey–West: OLS estimate with heteroscedasticity- and autocorrelation-consistent standard errors (2 lags). Diagnostics refer to the OLS residuals: Durbin–Watson values well below 2 indicate positive serial correlation, Breusch–Pagan tests for heteroscedasticity and Shapiro–Wilk tests for normality.",
           "CI, confidence interval; EAPC, estimated annual percentage change; SE, standard error. Source: Global Burden of Disease Study 2023."])

# ---------------- A.3 joinpoint ----------------
def jrow(m, a, max_jp=2):
    j = joinpoint(series(b, m, a).val.values, max_jp=max_jp)
    segs = [f"{s0}–{s1}: {ci(apc, lo, hi)}" for s0, s1, apc, lo, hi, p in j["ar1"]["segs"]]
    return j, segs
rows = []
JP = {}
for m in OUTS:
    for a in AG:
        j, segs = jrow(m, a); JP[(m, a)] = j
        A = j["ar1"]["aapc"]
        rows.append([m if a == AG[0] else "", LABEL[a], ", ".join(map(str, j["taus"])) or "None"] +
                    segs + [""] * (3 - len(segs)) + [ci(*A[:3])])
T["a3"] = dict(
    caption="**Table A.3.** Joinpoint regression of type 1 diabetes death, YLL, YLD, prevalence and incidence rates by age group, South Asia, 2000–2023.",
    head=["Outcome", "Age group (years)", "Joinpoints", "Segment 1: years, APC % (95% CI)", "Segment 2: years, APC % (95% CI)",
          "Segment 3: years, APC % (95% CI)", "AAPC 2000–2023, % (95% CI)"],
    widths=[1100, 1000, 1000, 2500, 2500, 2500, 2000], rows=rows, group=5,
    notes=["Continuous piecewise log-linear models with a maximum of two joinpoints, each at least two observations from either end of the series and from each other, selected by the Bayesian information criterion. APCs and AAPCs, with their confidence intervals, were estimated by Prais–Winsten regression to allow for first-order autoregressive errors. Across these models, Durbin–Watson statistics of the ordinary least squares fits ranged from " + f"{min(v['dw'] for v in JP.values()):.2f} to {max(v['dw'] for v in JP.values()):.2f}" + ", and estimated AR(1) coefficients from " + f"{min(v['rho'] for v in JP.values()):.2f} to {max(v['rho'] for v in JP.values()):.2f}".replace("-", "−") + ".",
           "AAPC, average annual percentage change; APC, annual percentage change; CI, confidence interval; YLD, year lived with disability; YLL, year of life lost. Source: Global Burden of Disease Study 2023."])

# ---------------- A.4 joinpoint sensitivity ----------------
def short(j):
    return "; ".join(f"{s0}–{s1}: {sg(apc, 2)}{'*' if p < 0.05 else ''}" for s0, s1, apc, lo, hi, p in j["ar1"]["segs"])
rows = []
for m in ["Deaths", "Prevalence", "Incidence"]:
    for a in AG:
        y = series(b, m, a).val.values
        rows.append([m if a == AG[0] else "", LABEL[a], short(joinpoint(y, 1)), short(joinpoint(y, 2)), short(joinpoint(y, 3))])
T["a4"] = dict(
    caption="**Table A.4.** Sensitivity of joinpoint models to the maximum number of joinpoints allowed: annual percentage change by segment.",
    head=["Outcome", "Age group (years)", "Maximum one joinpoint", "Maximum two joinpoints (primary)", "Maximum three joinpoints"],
    widths=[1100, 1100, 3400, 3900, 4500], rows=rows, group=5,
    notes=["Values are years of each segment followed by its annual percentage change (%). * p < 0.05 (Prais–Winsten estimation, first-order autoregressive errors). Source: Global Burden of Disease Study 2023."])

# ---------------- A.5 pairwise comparisons ----------------
rows = []
for m in ["Deaths", "Prevalence", "Incidence"]:
    eb = [(eap[(m, a)]["ar1_b"], eap[(m, a)]["ar1_se"]) for a in BANDS]
    ab = [(JP[(m, a)]["ar1"]["aapc_b"], JP[(m, a)]["ar1"]["aapc_se"]) for a in BANDS]
    q1, p1, pw1 = compare(eb, [LABEL[a] for a in BANDS])
    q2, p2, pw2 = compare(ab, [LABEL[a] for a in BANDS])
    first = True
    for (_, r1), (_, r2) in zip(pw1.iterrows(), pw2.iterrows()):
        rows.append([m if first else "", f"{r1.a} vs {r1.b}", ci(r1.diff_pp, r1.lo, r1.hi), pv(r1.p_holm),
                     ci(r2.diff_pp, r2.lo, r2.hi), pv(r2.p_holm)])
        first = False
    rows.append(["", "Global test (3 df)", f"χ² = {q1:.1f}", pv(p1), f"χ² = {q2:.1f}", pv(p2)])
T["a5"] = dict(
    caption="**Table A.5.** Comparison of trends between the four 5-year age groups: differences in EAPC and AAPC, South Asia, 2000–2023.",
    head=["Outcome", "Comparison (years)", "Difference in EAPC, percentage points (95% CI)", "p (Holm)",
          "Difference in AAPC, percentage points (95% CI)", "p (Holm)"],
    widths=[1100, 1900, 3200, 1100, 3200, 1100], rows=rows, group=7,
    notes=["Differences are the first age group minus the second, on the log scale, expressed as percentage points; negative values mean a faster decline (or slower rise) in the first group. Slopes and their standard errors come from Prais–Winsten models, and the age groups do not overlap. Pairwise p values are adjusted for multiple comparisons by the Holm method. GBD estimates for neighbouring age groups are not fully independent, so these tests are approximate.",
           "AAPC, average annual percentage change; CI, confidence interval; df, degrees of freedom; EAPC, estimated annual percentage change. Source: Global Burden of Disease Study 2023."])

# ---------------- A.6 UI simulation ----------------
sim = pd.read_csv("out/ui_simulation.csv")
rows = []
for m in ["Deaths", "Prevalence"]:
    for rho in (0.0, 0.5, 0.9):
        s = sim[(sim.measure == m) & (sim.rho == rho)]
        for a in BANDS + ["<20 years"]:
            r = s[s.age == a].iloc[0]
            rows.append([m if (rho == 0.0 and a == BANDS[0]) else "", f"{rho:.1f}" if a == BANDS[0] else "", LABEL[a],
                         f"{sg(r.eapc_med)} ({sg(r.eapc_lo)} to {sg(r.eapc_hi)})",
                         f"{100 * (1 - r.p_eapc_neg):.1f}" if m == "Prevalence" else f"{100 * r.p_eapc_neg:.1f}",
                         f"{100 * r.p_mid_pos:.1f}"])
        o = s[s.age == "P(5-9 faster decline than 15-19)"].eapc_med.iloc[0]
        f5 = s[s.age == "P(5-9 fastest of four bands)"].eapc_med.iloc[0]
        s15 = s[s.age == "P(15-19 slowest of four bands)"].eapc_med.iloc[0]
        if m == "Deaths":
            rows.append(["", "", "Age ordering", f"5–9 faster than 15–19: {100 * o:.1f}%; 5–9 fastest: {100 * f5:.1f}%; 15–19 slowest: {100 * s15:.1f}%", "", ""])
T["a6"] = dict(
    caption="**Table A.6.** Sensitivity of trend estimates to GBD uncertainty: results of 2,000 simulations drawn from the published 95% uncertainty intervals.",
    head=["Outcome", "Between-year correlation (ρ)", "Age group (years)", "EAPC, % median (2.5th to 97.5th percentile)",
          "Simulations with EAPC in the direction observed, %", "Simulations with a rise in 2018–2021, %"],
    widths=[1100, 1300, 1100, 4800, 2200, 2200], rows=rows, group=None,
    notes=["GBD posterior draws are not available from the GBD Results Tool. Each simulation drew every year's log-rate from a normal distribution centred on the GBD estimate, with standard deviation (ln upper − ln lower)/3.92, and errors correlated across years with the stated AR(1) correlation; age groups were drawn independently. EAPCs were fitted by ordinary least squares in each simulation. The rise in 2018–2021 is the annual percentage change of the middle segment of a model with joinpoints fixed at 2018 and 2021. The direction observed is a decline for deaths and a rise for prevalence.",
           "EAPC, estimated annual percentage change; GBD, Global Burden of Disease."])

# ---------------- A.7 sex ----------------
st = pd.read_csv("out/strata_eapc.csv")
rows = []
for m in ["Deaths", "Prevalence", "Incidence"]:
    for a in AG:
        M = st[(st.location == "South Asia") & (st.sex == "Male") & (st.measure == m) & (st.age == a)].iloc[0]
        F = st[(st.location == "South Asia") & (st.sex == "Female") & (st.measure == m) & (st.age == a)].iloc[0]
        q, pg, pw = compare([(M.b, M.se), (F.b, F.se)], ["M", "F"])
        d = {"Deaths": 3, "Prevalence": 1, "Incidence": 2}[m]
        rows.append([m if a == AG[0] else "", LABEL[a],
                     f"{n(M.rate2000, d)} / {n(M.rate2023, d)}", ci(M.eapc, M.lo, M.hi),
                     f"{n(F.rate2000, d)} / {n(F.rate2023, d)}", ci(F.eapc, F.lo, F.hi), pv(pw.p[0])])
T["a7"] = dict(
    caption="**Table A.7.** Type 1 diabetes death, prevalence and incidence rates by sex and age group, South Asia, 2000–2023.",
    head=["Outcome", "Age group (years)", "Males: rate 2000 / 2023", "Males: EAPC % (95% CI)",
          "Females: rate 2000 / 2023", "Females: EAPC % (95% CI)", "Male vs female EAPC, p"],
    widths=[1100, 1100, 1900, 2500, 1900, 2500, 1400], rows=rows, group=5,
    notes=["Rates are per 100,000 population. EAPCs were estimated by Prais–Winsten regression; the last column tests whether the male and female slopes differ (z-test).",
           "CI, confidence interval; EAPC, estimated annual percentage change. Source: Global Burden of Disease Study 2023."])

# ---------------- A.8 countries ----------------
rows = []
for loc in LOCS:
    for a in AG:
        r = {m: st[(st.location == loc) & (st.sex == "Both") & (st.measure == m) & (st.age == a)].iloc[0] for m in ["Deaths", "Prevalence", "Incidence"]}
        pp = popf[(popf.location_name == loc) & (popf.sex_name == "Both") & (popf.age_name == a) & (popf.year == 2023)].val.iloc[0]
        rows.append([loc if a == AG[0] else "", LABEL[a], n(pp / 1e6, 1),
                     f"{n(r['Deaths'].rate2000, 3)} / {n(r['Deaths'].rate2023, 3)}", ci(r["Deaths"].eapc, r["Deaths"].lo, r["Deaths"].hi),
                     ci(r["Prevalence"].eapc, r["Prevalence"].lo, r["Prevalence"].hi), ci(r["Incidence"].eapc, r["Incidence"].lo, r["Incidence"].hi)])
grad = []
for loc in LOCS[1:]:
    est = [(r.b, r.se) for a in ["5-9 years", "15-19 years"] for r in [st[(st.location == loc) & (st.sex == "Both") & (st.measure == "Deaths") & (st.age == a)].iloc[0]]]
    q, pg, pw_ = compare(est, ["5–9", "15–19"])
    grad.append(f"{loc} {ci(pw_.diff_pp[0], pw_.lo[0], pw_.hi[0])}, p {'< 0.001' if pw_.p[0] < 0.001 else '= ' + format(pw_.p[0], '.3f')}")
T["a8"] = dict(
    caption="**Table A.8.** Type 1 diabetes trends by country and age group, both sexes, 2000–2023.",
    head=["Location", "Age group (years)", "Population 2023, millions", "Death rate 2000 / 2023", "Deaths, EAPC % (95% CI)",
          "Prevalence, EAPC % (95% CI)", "Incidence, EAPC % (95% CI)"],
    widths=[1200, 1100, 1300, 1700, 2600, 2600, 2600], rows=rows, group=5,
    notes=["The GBD South Asia region comprises Bangladesh, Bhutan, India, Nepal and Pakistan. Death rates are per 100,000 population per year. EAPCs were estimated by Prais–Winsten regression. Difference in death-rate EAPC between the 5–9 and 15–19 year groups, percentage points (95% CI): " + "; ".join(grad) + ". Estimates for Bhutan and Nepal draw heavily on shared modelled information, and their prevalence and incidence series are almost identical.",
           "CI, confidence interval; EAPC, estimated annual percentage change. Source: Global Burden of Disease Study 2023."])

# ---------------- A.9 YLL/YLD composition ----------------
rows = []
for a in AG:
    for y in (2000, 2023):
        yl = row("YLLs", a, y, "Number"); yd = row("YLDs", a, y, "Number"); da = row("DALYs", a, y, "Number")
        rows.append([LABEL[a] if y == 2000 else "", str(y), ui(yl.val, yl.lower, yl.upper, 0), ui(yd.val, yd.lower, yd.upper, 0),
                     ui(da.val, da.lower, da.upper, 0), f"{100 * yd.val / da.val:.1f}"])
T["a9"] = dict(
    caption="**Table A.9.** Years of life lost, years lived with disability and DALYs from type 1 diabetes by age group, South Asia, 2000 and 2023.",
    head=["Age group (years)", "Year", "YLLs, n (95% UI)", "YLDs, n (95% UI)", "DALYs, n (95% UI)", "YLDs as share of DALYs, %"],
    widths=[1500, 700, 2900, 2600, 2900, 1800], rows=rows, group=2,
    notes=["DALYs are the sum of YLLs and YLDs. YLLs are deaths multiplied by the standard life expectancy at the age of death; YLDs are prevalent cases multiplied by the disability weight of each sequela.",
           "DALY, disability-adjusted life year; UI, uncertainty interval; YLD, year lived with disability; YLL, year of life lost. Source: Global Burden of Disease Study 2023."])

# ---------------- A.10 continuity with original 10-19 group ----------------
rows = []
for m in ["Deaths", "DALYs", "Prevalence"]:
    e = eapc_all(series(b, m, "10-19 years").val)
    j = joinpoint(series(b, m, "10-19 years").val.values)
    rows.append([m, ci(*e["ols"][:3]), ci(*e["ar1"][:3]),
                 "; ".join(f"{s0}–{s1}: {ci(apc, lo, hi)}" for s0, s1, apc, lo, hi, p in j["ar1"]["segs"])])
T["a10"] = dict(
    caption="**Table A.10.** Trend estimates for the 10–19 year group, South Asia, 2000–2023.",
    head=["Outcome", "EAPC, OLS % (95% CI)", "EAPC, Prais–Winsten % (95% CI)", "Joinpoint segments: years, APC % (95% CI), Prais–Winsten"],
    widths=[1300, 2600, 2600, 7800], rows=rows, group=None,
    notes=["The 10–19 year group overlaps the 10–14 and 15–19 year groups and is not part of the main analysis. It is shown for comparison with studies that report this age group.",
           "APC, annual percentage change; CI, confidence interval; EAPC, estimated annual percentage change; OLS, ordinary least squares. Source: Global Burden of Disease Study 2023."])


# ---------------- renumber appendix tables in order of first citation in the text ----------------
NEWNUM = {"a1": 1, "a9": 2, "a5": 3, "a2": 4, "a10": 5, "a3": 6, "a4": 7, "a6": 8, "a7": 9, "a8": 10}
OLDNUM = {k: int(k[1:]) for k in NEWNUM}
import re as _re
def _ren(text):
    return _re.sub(r"Table A\.(\d+)", lambda m: "Table A.@" + str({OLDNUM[k]: NEWNUM[k] for k in NEWNUM}[int(m.group(1))]), text).replace("A.@", "A.")
for k in list(T):
    T[k]["caption"] = _ren(T[k]["caption"])
    T[k]["notes"] = [_ren(x) for x in T[k]["notes"]]
T["order"] = sorted(NEWNUM, key=lambda k: NEWNUM[k])
json.dump(T, open("out/tables_rev.json", "w"), ensure_ascii=False, indent=1)
