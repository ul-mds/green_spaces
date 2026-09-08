"""
experiment.py

High-level grid runner and persistence utilities for ordinal logistic
regression experiments across multiple outcome × predictor combinations.

Supports three analysis modes, each with its own grid runner:

    run_linear_grid       — main exposure–outcome association (continuous outcome)
    run_ordinal_grid      — main exposure–outcome association (ordinal outcome)
    run_stratified_grid   — effect modification via stratified models
                             (model_type="ordinal" or "linear")
    run_interaction_grid  — formal interaction test (predictor × moderator)
                             (model_type="ordinal" or "linear")

Supports both categorical predictors (greenspace quartiles) and
continuous predictors (NDVI). Run them separately or in sequence —
each call produces a self-contained results_df and model_store.

Location: utils/experiment.py

──────────────────────────────────────────────────────────────
Typical usage — import and outcome_direction_map
──────────────────────────────────────────────────────────────
import utils.experiment as experiment
import config.columns as cols

outcome_direction_map = {
    "CES-D Ordinal":  {"direction": "higher_is_worse",  "label": "more severe depressive symptoms"},
    "GAD-7 Ordinal":  {"direction": "higher_is_worse",  "label": "more severe anxiety symptoms"},
    "SWLS Ordinal":   {"direction": "higher_is_better", "label": "better life satisfaction"},
    "IDS Ordinal":    {"direction": "higher_is_worse",  "label": "more severe depressive symptoms"},
    "GDS-15 Ordinal": {"direction": "higher_is_worse",  "label": "higher depression risk"},
}

──────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────
Typical usage — Linear Regression 
    continuous outcomes : Mental Scores
    continuous predictor: NDVI
──────────────────────────────────────────────────────────────
results_ndvi_df, model_store_ndvi = experiment.run_linear_grid(
    data                  = data,
    outcomes              = cols.mental_numeric,
    predictors            = cols.ndvi,
    predictor_type        = "continuous",
    scale_predictor       = 0.1,       # OR per 0.1-unit increase in NDVI
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex", "Socioeconomic Status (Tiers)"],
)

──────────────────────────────────────────────────────────────
Typical usage — Linear Regression 
    continuous outcomes : Mental Scores
    continuous predictor: Land-use greenspace quartiles
──────────────────────────────────────────────────────────────
results_df, model_store = experiment.run_linear_grid(
    data                  = data,
    outcomes              = cols.mental_numeric,      # CES-D Score, GAD-7 Score, etc.
    predictors            = cols.greenspace_quartiles,
    predictor_type        = "categorical",
    reference_category    = "1.0",
    label_mapping         = {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"},
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex", "Socioeconomic Status (Tiers)"],
)

──────────────────────────────────────────────────────────────
Typical usage — Ordinal Regression 
    continuous outcomes : Mental Ordinal
    continuous predictor: NDVI
──────────────────────────────────────────────────────────────
results_df, model_store = experiment.run_ordinal_grid(
    data                  = data,
    outcomes              = cols.mental_ordinal,      # CES-D Ordinal, GAD-7 Ordinal, etc.
    predictors            = cols.ndvi,
    predictor_type        = "continuous",
    scale_predictor       = 0.1,       # OR per 0.1-unit increase in NDVI
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex", "Socioeconomic Status (Tiers)"],
)

──────────────────────────────────────────────────────────────
Typical usage — Ordinal Regression 
    continuous outcomes : Mental Ordinal
    continuous predictor: Land-use greenspace quartiles
──────────────────────────────────────────────────────────────
results_df, model_store = experiment.run_ordinal_grid(
    data                  = data,
    outcomes              = cols.mental_ordinal,      # CES-D Ordinal, GAD-7 Ordinal, etc.
    predictors            = cols.greenspace_quartiles,
    predictor_type        = "categorical",
    reference_category    = "1.0",
    label_mapping         = {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"},
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex", "Socioeconomic Status (Tiers)"],
)

──────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────
Typical usage — Stratified Linear Regression (effect modification) 
    stratified by SES
    continuous outcomes : Mental Scores
    continuous predictor: NDVI
──────────────────────────────────────────────────────────────
strat_df, strat_store = experiment.run_stratified_grid(
    data                  = data,
    outcomes              = cols.mental_numeric,
    predictors            = cols.ndvi,
    stratify_by           = "Socioeconomic Status (Tiers)",
    strata                = [1, 2, 3],
    model_type            = "linear",
    predictor_type        = "continuous",
    scale_predictor       = 0.1,
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex"],
)

──────────────────────────────────────────────────────────────
Typical usage — Stratified Linear Regression (effect modification)
    stratified by SES
    continuous outcomes : Mental Scores
    ordinal predictor: Land-use greenspace quartiles
──────────────────────────────────────────────────────────────
strat_df, strat_store = experiment.run_stratified_grid(
    data                  = data,
    outcomes              = cols.mental_numeric,
    predictors            = cols.greenspace_quartiles,
    stratify_by           = "Socioeconomic Status (Tiers)",
    strata                = [1, 2, 3],
    model_type            = "linear",
    predictor_type        = "categorical",
    reference_category    = "1.0",
    label_mapping         = {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"},
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex"],
)

──────────────────────────────────────────────────────────────
Typical usage — Stratified Ordinal Regression (effect modification)
    stratified by SES
    ordinal outcomes : Mental Ordinal
    continuous predictor: NDVI
──────────────────────────────────────────────────────────────
strat_df, strat_store = experiment.run_stratified_grid(
    data                  = data,
    outcomes              = cols.mental_ordinal,
    predictors            = cols.ndvi,
    stratify_by           = "Socioeconomic Status (Tiers)",
    strata                = [1, 2, 3],
    model_type            = "ordinal",
    predictor_type        = "continuous",
    scale_predictor       = 0.1,
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex"],
)

──────────────────────────────────────────────────────────────
Typical usage — Stratified Ordinal Regression (effect modification)
    stratified by SES
    ordinal outcomes : Mental Ordinal
    ordinal predictor: Land-use greenspace quartiles
──────────────────────────────────────────────────────────────
strat_df, strat_store = experiment.run_stratified_grid(
    data                  = data,
    outcomes              = cols.mental_ordinal,
    predictors            = cols.greenspace_quartiles,
    stratify_by           = "Socioeconomic Status (Tiers)",
    strata                = [1, 2, 3],
    model_type            = "ordinal",
    predictor_type        = "categorical",
    reference_category    = "1.0",
    label_mapping         = {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"},
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex"],
)

──────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────
Typical usage — Interaction Linear Regression (effect modification)
    moderator: SES
    continuous outcomes : Mental Scores
    continuous predictor: NDVI
──────────────────────────────────────────────────────────────
inter_df, inter_store = experiment.run_interaction_grid(
    data                  = data,
    outcomes              = cols.mental_numeric,
    predictors            = cols.ndvi,
    moderator             = "Socioeconomic Status (Tiers)",
    moderator_type        = "categorical",
    moderator_reference   = "1.0",
    model_type            = "linear",
    predictor_type        = "continuous",
    scale_predictor       = 0.1,
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex"],
)

──────────────────────────────────────────────────────────────
Typical usage — Interaction Linear Regression (effect modification)
    moderator: SES
    continuous outcomes : Mental Scores
    ordinal predictor: Land-use greenspace quartiles
──────────────────────────────────────────────────────────────
inter_df, inter_store = experiment.run_interaction_grid(
    data                  = data,
    outcomes              = cols.mental_numeric,
    predictors            = cols.greenspace_quartiles,
    moderator             = "Socioeconomic Status (Tiers)",
    moderator_type        = "categorical",
    moderator_reference   = "1.0",
    model_type            = "linear",
    predictor_type        = "categorical",
    reference_category    = "1.0",
    label_mapping         = {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"},
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex"],
)

──────────────────────────────────────────────────────────────
Typical usage — Interaction Ordinal Regression (effect modification)
    moderator: SES
    ordinal outcomes : Mental Ordinal
    continuous predictor: NDVI
──────────────────────────────────────────────────────────────
inter_df, inter_store = experiment.run_interaction_grid(
    data                  = data,
    outcomes              = cols.mental_ordinal,
    predictors            = cols.ndvi,
    moderator             = "Socioeconomic Status (Tiers)",
    moderator_type        = "categorical",
    moderator_reference   = "1.0",
    model_type            = "ordinal",
    predictor_type        = "continuous",
    scale_predictor       = 0.1,
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex"],
)

──────────────────────────────────────────────────────────────
Typical usage — Interaction Ordinal Regression (effect modification)
    moderator: SES
    ordinal outcomes : Mental Ordinal
    ordinal predictor: Land-use greenspace quartiles
──────────────────────────────────────────────────────────────
inter_df, inter_store = experiment.run_interaction_grid(
    data                  = data,
    outcomes              = cols.mental_ordinal,
    predictors            = cols.greenspace_quartiles,
    moderator             = "Socioeconomic Status (Tiers)",
    moderator_type        = "categorical",
    moderator_reference   = "1.0",
    model_type            = "ordinal",
    predictor_type        = "categorical",
    reference_category    = "1.0",
    label_mapping         = {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"},
    outcome_direction_map = outcome_direction_map,
    covariates            = ["Age", "Sex"],
)

──────────────────────────────────────────────────────────────
Persistence (same save/load for all grid runners)
──────────────────────────────────────────────────────────────
experiment.save_results(results_df, model_store, path="outputs/ordinal_results")
results_df, model_store = experiment.load_results("outputs/ordinal_results")
"""

import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path

import utils.ordinal_regression as ordinal
import utils.linear_regression  as linear

__all__ = [
    "run_ordinal_grid",
    "run_linear_grid",
    "run_stratified_grid",
    "run_interaction_grid",
    "save_results",
    "load_results",
]


############################################################
## Internal helpers
############################################################
def _resolve_direction(outcome_direction_map, outcome):
    """
    Extract outcome_direction and interpretation_label from the map.
    Returns (outcome_direction, interpretation_label) or (None, None)
    if the outcome is missing or invalid.
    """
    entry = outcome_direction_map.get(outcome)
 
    if entry is None:
        return None, None
 
    if isinstance(entry, dict):
        direction = entry.get("direction")
        label     = entry.get("label", None)
    else:
        direction = entry
        label     = None
 
    if direction not in {"higher_is_worse", "higher_is_better"}:
        return None, None
 
    return direction, label
 
 
# Statsmodels result types that are not picklable
_SM_RESULT_MARKERS = ("Results", "Wrapper")

def _make_serializable(obj):
    """
    Recursively strip unpicklable statsmodels result objects (ordinal or
    linear) from a nested dict so it can be pickled.
    Raw result objects remain available in the in-memory store during
    the session but are excluded from saved files.
    """
    if isinstance(obj, dict):
        return {
            k: _make_serializable(v)
            for k, v in obj.items()
            if not (
                k == "result"
                and any(m in type(v).__name__ for m in _SM_RESULT_MARKERS)
            )
        }
    return obj
 
 
def _insert_identifiers(df, **kwargs):
    """Insert key=value pairs as leading columns in a DataFrame."""
    for i, (col, val) in enumerate(kwargs.items()):
        df.insert(i, col, val)
    return df


def _resolve_model_module(model_type):
    """
    Return the regression module matching model_type.
    "ordinal" → utils.ordinal_regression (fit_stratified/fit_interaction
                need outcome_direction; or_table/interaction OR columns)
    "linear"  → utils.linear_regression  (no outcome_direction; beta_table/
                interaction beta columns)
    """
    if model_type == "ordinal":
        return ordinal
    elif model_type == "linear":
        return linear
    else:
        raise ValueError("model_type must be 'ordinal' or 'linear'.")


############################################################
## Linear Regression Grid Runner
############################################################
def run_linear_grid(
    data,
    outcomes,
    predictors,
    outcome_direction_map,
    predictor_type="categorical",
    reference_category=None,
    covariates=None,
    label_mapping=None,
    scale_predictor=None
):
    """
    Fit OLS linear regression for all outcome × predictor pairs.

    Use this when outcomes are continuous numeric scores (e.g. CES-D Score,
    GAD-7 Score) rather than ordinal categories. Results are on the
    coefficient (beta) scale, not the OR scale.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis dataset (already renamed and transformed).

    outcomes : list[str] or str
        Continuous numeric outcome variables.
        Example: cols.mental_numeric

    predictors : list[str] or str
        Exposure variables.
        Categorical: cols.greenspace_quartiles
        Continuous:  cols.ndvi

    outcome_direction_map : dict
        Same format as run_ordinal_grid().
        Direction is used only for interpreting results in reporting —
        it does not affect the model itself.
        Outcomes missing from the map are skipped with a warning.

    predictor_type : str, default="categorical"

    reference_category : str, optional

    covariates : list[str] or str, optional

    label_mapping : dict, optional
        Default: {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"}

    scale_predictor : float, optional

    Returns
    -------
    results_df : pd.DataFrame
        Long-format — one row per predictor level per model.
        Columns:
            outcome, predictor, predictor_type, n, outcome_direction,
            reference_category (categorical) or scale_predictor (continuous),
            reference_level, level,
            beta, SE, CI_lower, CI_upper, beta_95CI,
            t_stat, p_value, p_value_fmt, covariates

    model_store : dict
        Keyed by (outcome, predictor).
        Each entry: "result", "metadata", "results_df".
    """
    if outcome_direction_map is None:
        outcome_direction_map = {}

    if isinstance(outcomes,   str): outcomes   = [outcomes]
    if isinstance(predictors, str): predictors = [predictors]

    model_store  = {}
    results_list = []
    n_total      = len(outcomes) * len(predictors)
    done         = 0

    for outcome in outcomes:

        direction, label = _resolve_direction(outcome_direction_map, outcome)
        if direction is None:
            print(f"  [warn] '{outcome}' — missing or invalid direction, skipping.")
            done += len(predictors)
            continue

        for predictor in predictors:
            done += 1
            print(f"  [{done}/{n_total}] {outcome} × {predictor}")

            try:
                result, meta = linear.fit_model(
                    data               = data,
                    outcome            = outcome,
                    predictor          = predictor,
                    predictor_type     = predictor_type,
                    reference_category = reference_category,
                    covariates         = covariates,
                    scale_predictor    = scale_predictor
                )

                res_df = linear.extract_results(
                    result, meta,
                    label_mapping = label_mapping
                ).copy()

                id_cols = dict(
                    outcome           = meta["outcome"],
                    predictor         = meta["predictor"],
                    predictor_type    = meta["predictor_type"],
                    n                 = meta["n"],
                    outcome_direction = direction,
                )
                if predictor_type == "categorical":
                    id_cols["reference_category"] = meta["reference_category"]
                else:
                    id_cols["scale_predictor"] = meta["scale_predictor"]

                res_df = _insert_identifiers(res_df, **id_cols)

                model_store[(outcome, predictor)] = {
                    "result":     result,
                    "metadata":   meta,
                    "results_df": res_df
                }
                results_list.append(res_df)

            except Exception as e:
                print(f"  [error] outcome='{outcome}', predictor='{predictor}': {e}")

    results_df = (
        pd.concat(results_list, axis=0, ignore_index=True)
        if results_list else pd.DataFrame()
    )

    n_fitted = len(model_store)
    print(
        f"\n  Linear grid complete: {n_fitted}/{n_total} models fitted"
        + (f", {n_total - n_fitted} failed or skipped." if n_total > n_fitted else ".")
    )
    return results_df, model_store


############################################################
## Ordinal Regression Grid Runner
############################################################
def run_ordinal_grid(
    data,
    outcomes,
    predictors,
    outcome_direction_map,
    predictor_type="categorical",
    reference_category=None,
    covariates=None,
    label_mapping=None,
    scale_predictor=None
):
    """
    Fit ordinal logistic regression for all outcome × predictor pairs.

    Use this when outcomes are ordinal valuse (e.g. CES-D Ordinal,
    GAD-7 Ordinal) rather than numeric scores. Results are on the OR scale.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis dataset (already renamed and transformed).

    outcomes : list[str] or str
        Ordinal mental-health outcome variables.
        Example: cols.mental_ordinal or "CES-D Ordinal"

    predictors : list[str] or str
        Exposure variables — all must share the same predictor_type.
        Categorical: cols.greenspace_quartiles
        Continuous:  cols.ndvi

    outcome_direction_map : dict
        Maps each outcome to its direction and optional interpretation label.

        Simple form (direction only):
            {"CES-D Ordinal": "higher_is_worse"}

        Extended form (direction + custom label):
            {
                "CES-D Ordinal":  {
                    "direction": "higher_is_worse",
                    "label":     "more severe depressive symptoms"
                },
                "SWLS Ordinal": {
                    "direction": "higher_is_better",
                    "label":     "better life satisfaction"
                }
            }

        Outcomes missing from this map are skipped with a warning.

    predictor_type : str, default="categorical"
        "categorical" or "continuous".
        All predictors in a single call must be the same type.
        Run separate calls for quartiles and NDVI.

    reference_category : str, optional
        Reference level for categorical predictors.
        Example: "1.0" for quartile Q1.
        Not used for continuous predictors.

    covariates : list[str] or str, optional
        Adjustment variables included in every model.
        Example: ["Age", "Sex", "Socioeconomic Status (Tiers)"]

    label_mapping : dict, optional
        Maps raw coefficient labels to display labels.
        Only used for categorical predictors.
        Default (quartiles):
            {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"}
        The reference level key (e.g. "1.0") determines reference_level
        in the output — no separate parameter needed.

    scale_predictor : float, optional
        Divide the continuous predictor by this value before fitting.
        Only relevant for continuous predictors.
        Example: 0.1 → OR per 0.1-unit increase in NDVI.

    Returns
    -------
    results_df : pd.DataFrame
        Long-format table — one row per predictor level per model.

        Columns (categorical):
            outcome, predictor, predictor_type, n, reference_category,
            outcome_direction, reference_level, level,
            beta, OR, CI_lower, CI_upper, OR_95CI,
            p_value, p_value_fmt, pct_change_odds, interpretation

        Columns (continuous):
            outcome, predictor, predictor_type, n, scale_predictor,
            outcome_direction, reference_level (None), level,
            beta, OR, CI_lower, CI_upper, OR_95CI,
            p_value, p_value_fmt, pct_change_odds, interpretation

    model_store : dict
        Keyed by (outcome, predictor) tuple.
        Each entry:
            - "result"    : OrderedResults — raw statsmodels object (in-memory only)
            - "metadata"  : dict           – from ordinal.fit_model()
            - "results_df": pd.DataFrame   — extracted OR table with identifiers
    """

    if outcome_direction_map is None:
        outcome_direction_map = {}
        print(
            f"[warn] 'outcome_direction_map' is None; "
            f"initialized as empty dictionary."
        )

    if isinstance(outcomes, str):   outcomes   = [outcomes]
    if isinstance(predictors, str): predictors = [predictors]
    
    model_store  = {}
    results_list = []

    n_total = len(outcomes) * len(predictors)
    done    = 0

    for outcome in outcomes:

        # --------------------------------------------------
        # Resolve direction and interpretation label
        # --------------------------------------------------
        direction, label = _resolve_direction(outcome_direction_map, outcome)
        if direction is None:
            print(f"  [warn] '{outcome}' — missing or invalid direction, skipping.")
            done += len(predictors)
            continue

        for predictor in predictors:

            done += 1
            print(f"  [{done}/{n_total}] {outcome} × {predictor}")

            try:
                # ------------------------------------------
                # Fit Ordinal Model
                # ------------------------------------------
                result, metadata = ordinal.fit_model(
                    data               = data,
                    outcome            = outcome,
                    predictor          = predictor,
                    predictor_type     = predictor_type,
                    reference_category = reference_category,
                    covariates         = covariates,
                    scale_predictor    = scale_predictor
                )

                # ------------------------------------------
                # Extract OR table
                # reference_level column comes from extract_results
                # (derived from label_mapping for categorical,
                #  None for continuous)
                # ------------------------------------------
                results_df = ordinal.extract_results(
                    result,
                    metadata,
                    outcome_direction     = direction,
                    interpretation_label  = label,
                    label_mapping         = label_mapping
                ).copy()

                # ------------------------------------------
                # Add model-level identifiers
                # ------------------------------------------
                id_cols = dict(
                    outcome           = metadata["outcome"],
                    predictor         = metadata["predictor"],
                    predictor_type    = metadata["predictor_type"],
                    n                 = metadata["n"],
                    outcome_direction = direction,
                )

                if predictor_type == "categorical":
                    id_cols["reference_category"] = metadata["reference_category"]
                else:
                    id_cols["scale_predictor"] = metadata["scale_predictor"]

                results_df = _insert_identifiers(results_df, **id_cols)
                
                model_store[(outcome, predictor)] = {
                    "result":     result,
                    "metadata":   metadata,
                    "results_df": results_df
                }

                results_list.append(results_df)

            except Exception as e:
                print(
                    f"  [error] outcome='{outcome}', "
                    f"predictor='{predictor}': {e}"
                )

    # the final results_df
    results_df = (
        pd.concat(results_list, axis=0, ignore_index=True)
        if results_list
        else pd.DataFrame()
    )

    n_fitted = len(model_store)
    print(
        f"\n  Grid complete: {n_fitted}/{n_total} models fitted"
        + (f", {n_total - n_fitted} failed or skipped." if n_total > n_fitted else ".")
    )

    return results_df, model_store


############################################################
## Stratified Grid Runner (Effect Modification)
############################################################
def run_stratified_grid(
    data,
    outcomes,
    predictors,
    stratify_by,
    strata=None,
    model_type="linear",
    predictor_type="categorical",
    scale_predictor=None,
    reference_category=None,
    label_mapping=None,
    outcome_direction_map=None,
    covariates=None,
    min_group_n=30,
):
    """
    Fit stratified ordinal regression models for all outcome × predictor
    pairs to assess effect modification.

    Works with either ordinal or linear models via model_type.
 
    For each pair, a separate model is fitted within each stratum of
    `stratify_by`. Comparing effect estimates across strata reveals 
    whether the exposure–outcome association is more pronounced in 
    specific subgroups (e.g. low-SES participants).
 
    Pair with run_interaction_grid() for the formal statistical test.
 
    Parameters
    ----------
    data : pd.DataFrame
 
    outcomes : list[str] or str
        Ordinal outcomes (model_type="ordinal") or continuous numeric
        outcomes (model_type="linear").
 
    predictors : list[str] or str
 
    stratify_by : str
        Grouping variable defining the strata.
        Example: "Socioeconomic Status (Tiers)"  (values: 1, 2, 3)
 
    strata : list, optional
        Specific stratum values to include.
        If None, all unique values of stratify_by are used.

    model_type : {"linear", "ordinal"}, default="linear"
        Which regression module to use for every model in the grid.
 
    predictor_type : str, default="categorical"
 
    scale_predictor : float, optional
    
    reference_category : str, optional
 
    label_mapping : dict, optional
    
    outcome_direction_map : dict
        For model_type="linear" this is used only for the
        outcome_direction identifier column in the output — it does
        not affect the model itself.
    
    covariates : list[str], optional
        Adjustment variables other than stratify_by.
        stratify_by is never passed as a covariate — the data is
        already split by it.
 
    min_group_n : int, default=30
        Minimum sample size required within a stratum to fit a model.
        Passed through to fit_stratified().
 
    Returns
    -------
    results_df : pd.DataFrame
        Long-format, stacked across strata and pairs.

        Ordinal columns:
            outcome, predictor, predictor_type, n, stratum_var, stratum,
            outcome_direction, reference_level, level,
            beta, OR, CI_lower, CI_upper, OR_95CI,
            p_value, p_value_fmt, pct_change_odds, interpretation, covariates

        Linear columns:
            outcome, predictor, predictor_type, n, stratum_var, stratum,
            outcome_direction, reference_level, level,
            beta, SE, CI_lower, CI_upper, beta_95CI,
            t_stat, p_value, p_value_fmt, covariates

    model_store : dict
        Keyed by (outcome, predictor).
        Each entry:
            - "strata_results" : dict keyed by stratum value
                  each with "result", "metadata", "n", and
                  "beta_table" (linear) or "or_table" (ordinal)
            - "summary"        : pd.DataFrame — stacked tables
    """
    if outcome_direction_map is None:
        outcome_direction_map = {}

    model = _resolve_model_module(model_type)
    table_key = "or_table" if model_type == "ordinal" else "beta_table"
 
    if isinstance(outcomes,   str): outcomes   = [outcomes]
    if isinstance(predictors, str): predictors = [predictors]
 
    model_store  = {}
    results_list = []
    n_total      = len(outcomes) * len(predictors)
    done         = 0
 
    for outcome in outcomes:
 
        direction, label = _resolve_direction(outcome_direction_map, outcome)
        if direction is None:
            print(f"  [warn] '{outcome}' — missing or invalid direction, skipping.")
            done += len(predictors)
            continue
 
        for predictor in predictors:
            done += 1
            print(f"  [{done}/{n_total}] {outcome} × {predictor}  [stratified by {stratify_by}, {model_type}]")
 
            try:
                # fit_stratified handles the subsetting per stratum
                strata_results, _ = model.fit_stratified(
                    data               = data,
                    outcome            = outcome,
                    predictor          = predictor,
                    stratify_by        = stratify_by,
                    strata             = strata,
                    predictor_type     = predictor_type,
                    reference_category = reference_category,
                    covariates         = covariates,
                    scale_predictor    = scale_predictor,
                    min_group_n        = min_group_n
                )
 
                # Re-extract tables with correct outcome direction / label
                # mapping (fit_stratified uses defaults internally).
                # Ordinal extract_results needs outcome_direction and
                # interpretation_label; linear's does not.
                extract_kwargs = dict(label_mapping=label_mapping)
                if model_type == "ordinal":
                    extract_kwargs["outcome_direction"]    = direction
                    extract_kwargs["interpretation_label"] = label
                
                stratum_tables = []
                for stratum, res in strata_results.items():
                    stratum_table = model.extract_results(
                        res["result"], res["metadata"], **extract_kwargs
                    ).copy()
 
                    res[table_key] = stratum_table
 
                    id_cols = dict(
                        outcome           = outcome,
                        predictor         = predictor,
                        predictor_type    = predictor_type,
                        n                 = res["n"],
                        stratum_var       = stratify_by,
                        stratum           = stratum,
                        outcome_direction = direction,
                    )
                    if predictor_type == "categorical":
                        id_cols["reference_category"] = res["metadata"]["reference_category"]
                    else:
                        id_cols["scale_predictor"] = res["metadata"]["scale_predictor"]
 
                    stratum_table = _insert_identifiers(stratum_table, **id_cols)
                    stratum_tables.append(stratum_table)
 
                summary = (
                    pd.concat(stratum_tables, axis=0, ignore_index=True)
                    if stratum_tables else pd.DataFrame()
                )
 
                model_store[(outcome, predictor)] = {
                    "strata_results": strata_results,
                    "summary":        summary
                }
                results_list.append(summary)
 
            except Exception as e:
                print(f"  [error] outcome='{outcome}', predictor='{predictor}': {e}")
 
    results_df = (
        pd.concat(results_list, axis=0, ignore_index=True)
        if results_list else pd.DataFrame()
    )
 
    n_fitted = len(model_store)
    print(
        f"\n  Stratified grid complete: {n_fitted}/{n_total} pairs fitted"
        + (f", {n_total - n_fitted} failed or skipped." if n_total > n_fitted else ".")
    )
    return results_df, model_store
 
 
############################################################
## Interaction Grid Runner (Effect Modification)
############################################################
def run_interaction_grid(
    data,
    outcomes,
    predictors,
    moderator,
    moderator_type="categorical",
    moderator_reference=None,
    model_type="ordinal",
    predictor_type="categorical",
    scale_predictor=None,
    reference_category=None,
    label_mapping=None,
    outcome_direction_map=None,
    covariates=None,
):
    """
    Formally test effect modification for all outcome × predictor pairs
    by fitting a model with a predictor × moderator interaction term.

    Works with either ordinal or linear models via model_type.
 
    A significant interaction p-value means the exposure–outcome
    association statistically differs across moderator levels.
 
    Run run_stratified_grid() alongside this to inspect the direction
    and magnitude of the effect in each subgroup.
 
    Parameters
    ----------
    data : pd.DataFrame
 
    outcomes : list[str] or str
        Ordinal outcomes (model_type="ordinal") or continuous numeric
        outcomes (model_type="linear").
 
    predictors : list[str] or str
 
    moderator : str
        Effect modifier variable.
        Example: "Socioeconomic Status (Tiers)"

    moderator_type : str, default="categorical"
        "categorical" or "continuous".
 
    moderator_reference : str/int, optional
        Reference level for the moderator.
        Example: "1" for low SES as reference.

    model_type : {"linear", "ordinal"}, default="linear"
        Which regression module to use for every model in the grid.
 
    predictor_type : str, default="categorical"

    scale_predictor : float, optional
 
    reference_category : str, optional

    label_mapping : dict, optional
        Maps raw coefficient labels to display labels.
        Default: {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"}
 
    outcome_direction_map : dict
        Used only for the outcome_direction identifier column in the
        output — interaction models themselves do not depend on it.

    covariates : list[str], optional
        Additional adjustment variables (not interacted).

    Returns
    -------
    results_df : pd.DataFrame
        Long-format with one row per interaction term per model.
 
        Ordinal columns:
            outcome, predictor, predictor_type, n, moderator,
            outcome_direction, reference_level, level,
            beta, OR, CI_lower, CI_upper, OR_95CI,
            p_value, p_value_fmt, pct_change_odds, covariates, significant
 
        Linear columns:
            outcome, predictor, predictor_type, n, moderator,
            outcome_direction, reference_level, level,
            beta, SE, CI_lower, CI_upper, beta_95CI,
            p_value, p_value_fmt, covariates, significant
 
    model_store : dict
        Keyed by (outcome, predictor).
        Each entry:
            - "result"            : result object (in-memory only)
            - "metadata"          : dict
            - "interaction_table" : pd.DataFrame
    """
    if outcome_direction_map is None:
        outcome_direction_map = {}

    model = _resolve_model_module(model_type)
 
    if isinstance(outcomes,   str): outcomes   = [outcomes]
    if isinstance(predictors, str): predictors = [predictors]
 
    model_store  = {}
    results_list = []
    n_total      = len(outcomes) * len(predictors)
    done         = 0
 
    for outcome in outcomes:
 
        direction, _ = _resolve_direction(outcome_direction_map, outcome)
        if direction is None:
            print(f"  [warn] '{outcome}' — missing or invalid direction, skipping.")
            done += len(predictors)
            continue
 
        for predictor in predictors:
            done += 1
            print(f"  [{done}/{n_total}] {outcome} × {predictor}  [× {moderator}, {model_type}]")
 
            try:
                result, meta, interaction_table = model.fit_interaction(
                    data                = data,
                    outcome             = outcome,
                    predictor           = predictor,
                    moderator           = moderator,
                    predictor_type      = predictor_type,
                    reference_category  = reference_category,
                    moderator_type      = moderator_type,
                    moderator_reference = moderator_reference,
                    covariates          = covariates,
                    scale_predictor     = scale_predictor,
                    label_mapping       = label_mapping
                )
 
                interaction_table = interaction_table.copy()
                
                id_cols = dict(
                    outcome           = outcome,
                    predictor         = predictor,
                    predictor_type    = predictor_type,
                    n                 = meta["n"],
                    moderator         = moderator,
                    outcome_direction = direction,
                )
                
                # Only insert identifiers that are not already present 
                # (in case a module's output changes independently).
                id_cols = {
                    k: v for k, v in id_cols.items()
                    if k not in interaction_table.columns
                }
                interaction_table = _insert_identifiers(interaction_table, **id_cols)
 
                model_store[(outcome, predictor)] = {
                    "result":            result,
                    "metadata":          meta,
                    "interaction_table": interaction_table
                }
                results_list.append(interaction_table)
 
            except Exception as e:
                print(f"  [error] outcome='{outcome}', predictor='{predictor}': {e}")
 
    results_df = (
        pd.concat(results_list, axis=0, ignore_index=True)
        if results_list else pd.DataFrame()
    )
 
    n_fitted = len(model_store)
    print(
        f"\n  Interaction grid complete: {n_fitted}/{n_total} models fitted"
        + (f", {n_total - n_fitted} failed or skipped." if n_total > n_fitted else ".")
    )
    return results_df, model_store


############################################################
## Save Results
############################################################
def save_results(
    results_df,
    model_store,
    path,
    save_csv=True
):
    """
    Persist the grid results to disk. Works with all three grid runners.

    Saves:
        <path>.pkl  — results_df + model_store (metadata and OR tables,
                      without raw statsmodels objects which are not
                      picklable due to patsy internals)
        <path>.csv  — results_df only, human-readable (optional)

    Note
    ----
    The raw statsmodels result objects in model_store["result"] are
    excluded from the pickle. All information needed for reporting —
    OR tables, CIs, p-values, metadata — is preserved. The raw objects
    remain available in the in-memory model_store during the session.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from any run_*_grid() function.

    model_store : dict
        Output from any run_*_grid() function.

    path : str or Path
        File path without extension. 
        Example: "outputs/ordinal_quartiles"

    save_csv : bool, default=True
        If True, also save results_df as a CSV alongside the pickle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "results_df":  results_df,
        "model_store": _make_serializable(model_store),
        "saved_at":    datetime.now().isoformat()
    }

    pkl_path = path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Saved pickle : {pkl_path}")

    if save_csv and not results_df.empty:
        csv_path = path.with_suffix(".csv")
        results_df.to_csv(csv_path, index=False)
        print(f"  Saved CSV    : {csv_path}")


############################################################
## Load Results
############################################################
def load_results(path):
    """
    Load previously saved grid results.
    Works with output from any run_*_grid() function.

    Parameters
    ----------
    path : str or Path
        File path with or without the .pkl extension.

    Returns
    -------
    results_df : pd.DataFrame
        Identical to what run_ordinal_grid() returned.

    model_store : dict
        Contains "metadata" and "results_df" per (outcome, predictor) key.
        Does not contain raw statsmodels result objects (not picklable).
    """
    path = Path(path).with_suffix(".pkl")

    if not path.exists():
        raise FileNotFoundError(f"No results file found at: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    print(f"  Loaded results saved at: {payload.get('saved_at', 'unknown')}")

    return payload["results_df"], payload["model_store"]
