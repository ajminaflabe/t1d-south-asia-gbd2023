"""Shared data loading for the DRCP revision analyses (GBD 2023, T1D, South Asia)."""
import numpy as np
import pandas as pd

BURDEN = "data/burden_both.csv"
POP = "data/pop_full.csv"

MEAS = {
    "Deaths": "Deaths",
    "DALYs": "DALYs (Disability-Adjusted Life Years)",
    "YLLs": "YLLs (Years of Life Lost)",
    "YLDs": "YLDs (Years Lived with Disability)",
    "Prevalence": "Prevalence",
    "Incidence": "Incidence",
}
INV = {v: k for k, v in MEAS.items()}
AGES = ["<5 years", "5-9 years", "10-14 years", "15-19 years", "<20 years"]   # main analysis
BANDS = AGES[:4]                                                              # non-overlapping
OLD = ["10-19 years"]                                                         # kept for continuity
LOCS = ["South Asia", "Bangladesh", "Bhutan", "India", "Nepal", "Pakistan"]
LABEL = {"<5 years": "Under 5", "5-9 years": "5–9", "10-14 years": "10–14",
         "15-19 years": "15–19", "<20 years": "Under 20", "10-19 years": "10–19"}


def load():
    b = pd.read_csv(BURDEN)
    b["measure"] = b.measure_name.map(INV)
    p = pd.read_csv(POP)
    pop = p[["location_name", "age_name", "year", "sex_name", "val"]]
    pop = pop.rename(columns={"val": "pop"})
    return b, pop


def series(b, measure, age, loc="South Asia", sex="Both", metric="Rate"):
    s = b[(b.measure == measure) & (b.age_name == age) & (b.location_name == loc)
          & (b.sex_name == sex) & (b.metric_name == metric)].sort_values("year")
    assert len(s) == 24, (measure, age, loc, sex, metric, len(s))
    return s
