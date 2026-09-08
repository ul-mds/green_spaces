# Greenspace & Mental Health — Analysis Pipeline

Reproducible pipeline studying the association between residential green
exposure (NDVI, land-use greenspace quartiles) and mental health outcomes
(CES-D, GAD-7, SWLS), including stratified subgroup and
interaction (effect-modification) analyses. Supports both OLS linear regression and ordinal logistic via a shared, parallel API.

## Setup

```bash
git clone <repo-url> && cd project_root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure

```
project_root/
├── preprocess.ipynb              # Build datasets from raw Excel files
├── analysis_stats.ipynb          # Univariate/bivariate EDA & correlations
├── analysis_ordinal_model.ipynb  # Ordinal logistic regression experiments
├── analysis_linear_model.ipynb   # OLS linear regression experiments
├── utils/
│   ├── columns.py               # Column name constants
│   ├── transform.py             # categorize / binarize / rename_columns
│   ├── psychometrics.py         # Imputation & reliability testing
│   ├── univariate.py            # Univariate EDA
│   ├── bivariate.py             # Bivariate EDA
│   ├── ordinal_regression.py    # Proportional-odds regression
│   ├── linear_regression.py     # OLS regression (parallel API)
│   └── experiment.py            # Grid runners + save/load
├── final_data/{raw,processed}/  # Source and processed datasets
└── outputs/{ordinal,linear}/    # Saved model results (.pkl + .csv)
```

## Pipeline

```
raw Excel → preprocess.ipynb → processed datasets → analysis_stats.ipynb (EDA)
                                                    → analysis_ordinal_model.ipynb
                                                    → analysis_linear_model.ipynb
```

Each model notebook runs, in order: main-effects models (NDVI + quartiles,
adjusted/unadjusted) → forest/profile plots → SES/Age/Sex-stratified models →
SES/Age/Sex interaction (effect-modification) tests. Results are saved under
`outputs/{ordinal,linear}/.../{adjusted,unadjusted}/`.

## Quickstart

```python
%load_ext autoreload
%autoreload 2

import utils.columns as cols
import utils.transform as transform
import utils.ordinal_regression as ordinal
import utils.experiment as experiment

df = transform.rename_columns(df)

outcome_direction_map = {
    "CES-D Ordinal": {"direction": "higher_is_worse", "label": "more severe depressive symptoms"},
}

results_df, store = experiment.run_ordinal_grid(
    data=df, outcomes=cols.mental_ordinal, predictors=cols.greenspace_quartiles,
    outcome_direction_map=outcome_direction_map,
    covariates=["Age", "Sex", "Socioeconomic Status (Tiers)"],
)
experiment.save_results(results_df, store, path="outputs/ordinal/mental_ordinal/quartiles/adjusted/ordinal_results")
```

## Modules

| Module | Purpose |
|---|---|
| `columns` | Column name constants (`cols.mental_numeric`, `cols.ndvi`, `cols.greenspace_quartiles`, …) |
| `transform` | `rename_columns`, `categorize`, `binarize` |
| `psychometrics` | `person_mean_imputation`, `reliability_test` (Omega/Alpha) |
| `univariate` | `univar_analysis`, `summarize_numeric`, `summarize_categorical`, `check_normality` |
| `bivariate` | `analyze_ordinal_outcome`, `analyze_numeric_categorical`, `bivar_analysis` (auto dispatcher), `pairwise_stats` |
| `ordinal_regression` | `fit_model`, `extract_results`, `plot_forest`, `plot_profile`, `fit_stratified`, `fit_interaction` |
| `linear_regression` | Same API as `ordinal_regression`, on the beta/coefficient scale |
| `experiment` | `run_ordinal_grid`, `run_linear_grid`, `run_stratified_grid`, `run_interaction_grid`, `save_results`, `load_results` |

`experiment.run_stratified_grid` and `run_interaction_grid` accept
`model_type="ordinal"` or `"linear"` to reuse the same grid logic for both
model families.

See each module's docstrings for full parameter documentation.

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels reliabilipy openpyxl
```
