# Type 1 diabetes trends in South Asian children and adolescents, 2000–2023 (GBD 2023)

Analysis code and data for:

Flabe AH, Zabeen B. *Age-specific trends in type 1 diabetes mortality and prevalence among children and adolescents in South Asia, 2000–2023: a Global Burden of Disease 2023 analysis.* Diabetes Research and Clinical Practice (under review).

## Data

`data/` contains the estimates used, extracted from the IHME GBD Results Tool (https://vizhub.healthdata.org/gbd-results/):

| File | Contents |
|---|---|
| `burden_both.csv` | Type 1 diabetes (cause ID 975): deaths, DALYs, YLLs, YLDs, prevalence and incidence; number and rate per 100,000; South Asia region, Bangladesh, Bhutan, India, Nepal and Pakistan; ages <5, 5–9, 10–14, 15–19, 10–19 and <20 years; both sexes, males and females; 2000–2023 |
| `pop_full.csv` | GBD 2023 population estimates for the same locations, ages (plus all ages), sexes and years |

Source: Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2023 (GBD 2023) Results. Seattle, United States: Institute for Health Metrics and Evaluation (IHME). The data are shared for non-commercial use under IHME's terms; please cite IHME when using them.

## Methods implemented

* `pw.py`: Prais–Winsten regression (AR(1) errors, first observation retained).
* `trend.py`: estimated annual percentage change (EAPC) by OLS, Prais–Winsten and OLS with Newey–West standard errors, with Durbin–Watson, Breusch–Pagan and Shapiro–Wilk diagnostics; joinpoint regression (continuous piecewise log-linear model, grid search over integer years, at least two observations from either end and between joinpoints, up to two joinpoints chosen by BIC), with APCs and AAPC estimated by Prais–Winsten.
* `compare.py`: Wald test of equal slopes across non-overlapping age groups and pairwise comparisons with Holm adjustment (EAPC and AAPC).
* `sim.py`: sensitivity to GBD uncertainty, with 2,000 simulated series per outcome and age group drawn from the published 95% uncertainty intervals (between-year correlation 0, 0.5 and 0.9).
* `strata.py`: sex-specific and country-level trends.

## How to run

```
pip install -r requirements.txt
python run_main.py       # EAPCs and joinpoint models   -> out/eapc_main.csv, out/joinpoint_main.json
python compare.py        # age-group comparisons         -> out/age_comparisons.csv
python sim.py            # uncertainty simulation        -> out/ui_simulation.csv
python strata.py         # sex and country analyses      -> out/strata_eapc.csv
python make_tables.py    # cell contents of all tables   -> out/tables_rev.json
python figures_rev.py    # Figures 1-3 and A.1-A.2       -> figs/
```

`out/tables_rev.json` holds the contents of Tables 1–2 and Appendix A, Tables A.1–A.10, in the order used in the paper.

Software used: Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, matplotlib 3.11.

## Contact

Ajmina Hasan Flabe, BRAC, Dhaka, Bangladesh (ahflabe1996@gmail.com)
