"""
linear_regression.py

Utility functions for fitting, extracting, and visualizing OLS linear
regression models via statsmodels.

Mirrors the API of ordinal_regression.py so both can be called
interchangeably from experiment.py with model_type="ordinal" or
model_type="linear".

Supports both categorical predictors (e.g. greenspace quartiles) and
continuous predictors (e.g. NDVI).

Location: utils/linear_regression.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats as scipy_stats

__all__ = [
    "fit_model",
    "extract_results",
    "plot_forest",
    "fit_stratified",
    "fit_interaction",
    "check_residuals",
]


############################################################
## Fit OLS Linear Regression
############################################################
def fit_model(
    data,
    outcome,
    predictor,
    predictor_type="categorical",
    reference_category=None,
    covariates=None,
    scale_predictor=None,
    cov_type="HC3"
):
    """
    Fit an OLS linear regression model.

    Mirrors the signature of ordinal_regression.fit_model() so both
    can be called from experiment.py with model_type="linear".

    Use this when the outcome is a continuous numeric score (e.g.
    CES-D Score, GAD-7 Score) rather than an ordinal category.

    Parameters
    ----------
    data : pd.DataFrame

    outcome : str
        Continuous numeric outcome variable.
        Example: "CES-D Score"

    predictor : str
        Exposure variable.
        - Categorical: greenspace quartiles, tertiles, etc.
        - Continuous:  NDVI, raw greenspace coverage, etc.

    predictor_type : str, default="categorical"
        "categorical" or "continuous".

    reference_category : str/int, optional
        Reference level for categorical predictors.
        Defaults to the numerically (or alphabetically) first level.
        Example: "1.0" for quartile Q1.
        Not used for continuous predictors.

    covariates : list[str] or str, optional
        Adjustment variables included as additive terms.
        Example: ["Age", "Sex", "Socioeconomic Status (Tiers)"]

    scale_predictor : float, optional
        Divide the continuous predictor by this value before fitting.
        Only relevant for continuous predictors.
        Example:
            scale_predictor=0.1 → coefficient is effect per
            0.1-unit increase in NDVI.

    cov_type : str, default="HC3"
        Covariance estimator passed to statsmodels .fit(cov_type=...).
        "HC3" gives heteroskedasticity-robust standard errors (recommended
        default — robust to non-constant residual variance across
        predictor/covariate levels, at negligible cost when errors are in
        fact homoskedastic). Set to "nonrobust" for classical OLS standard
        errors (assumes homoskedasticity), or another statsmodels-supported
        type ("HC0", "HC1", "HC2", "HAC", "cluster", ...).
        Note: Changing cov_type affects only SE, CI, and p-values — 
        point estimates (beta) are identical regardless of cov_type.

    Returns
    -------
    result : RegressionResultsWrapper
        Fitted statsmodels OLS result object.

    metadata : dict
        - n                 : int
        - outcome           : str
        - predictor         : str
        - predictor_type    : str
        - reference_category: str or None
        - covariates        : list[str]
        - scale_predictor   : float or None
        - predictor_levels  : list or None (categorical only)
        - cov_type          : str — covariance estimator used
    """

    if covariates is None:
        covariates = []
    elif isinstance(covariates, str):
        covariates = [covariates]
    else:
        covariates = list(covariates)

    model_variables = [outcome, predictor] + covariates
    subset = data[model_variables].dropna().copy()

    if subset.empty:
        raise ValueError(
            f"No complete cases for outcome='{outcome}', "
            f"predictor='{predictor}'."
        )

    # --------------------------------------------------
    # Predictor preparation
    # --------------------------------------------------
    if predictor_type == "categorical":
        subset[predictor] = subset[predictor].astype(str)

        raw_levels = subset[predictor].unique()
        try:
            predictor_levels = sorted(raw_levels, key=lambda v: float(v))
        except ValueError:
            predictor_levels = sorted(raw_levels)

        if len(predictor_levels) < 2:
            raise ValueError(
                "Categorical predictor must have at least two levels."
            )

        if reference_category is None:
            reference_category = predictor_levels[0]
        else:
            reference_category = str(reference_category)

    elif predictor_type == "continuous":
        subset[predictor]  = pd.to_numeric(subset[predictor], errors="coerce")
        subset             = subset.dropna(subset=[predictor])
        predictor_levels   = None
        reference_category = None

        if scale_predictor is not None:
            subset[predictor] = subset[predictor] / scale_predictor

    else:
        raise ValueError(
            "predictor_type must be 'categorical' or 'continuous'."
        )

    # --------------------------------------------------
    # Rename variables for formula
    # --------------------------------------------------
    rename_dict = {outcome: "outcome_y", predictor: "predictor_x"}
    working_df  = subset.rename(columns=rename_dict)

    # --------------------------------------------------
    # Build formula
    # --------------------------------------------------
    if predictor_type == "categorical":
        ref     = reference_category.replace('"', '\\"')
        formula = f'outcome_y ~ C(predictor_x, Treatment(reference="{ref}"))'
    else:
        formula = "outcome_y ~ predictor_x"

    if covariates:
        covariate_terms = " + ".join(f'Q("{cov}")' for cov in covariates)
        formula += " + " + covariate_terms

    # --------------------------------------------------
    # Fit model
    # --------------------------------------------------
    model  = smf.ols(formula, data=working_df)
    result = model.fit(cov_type=cov_type)

    metadata = {
        "n":                  len(working_df),
        "outcome":            outcome,
        "predictor":          predictor,
        "predictor_type":     predictor_type,
        "reference_category": reference_category,
        "covariates":         covariates,
        "predictor_levels":   predictor_levels,
        "scale_predictor":    scale_predictor,
        "cov_type":           cov_type
    }

    return result, metadata


############################################################
## Extract Coefficients, CIs, p-values
############################################################
def extract_results(
    result,
    metadata,
    label_mapping=None,
    round_digits=3
):
    """
    Extract regression coefficients from a fitted OLS model.

    Mirrors the output structure of ordinal_regression.extract_results()
    but on the coefficient (beta) scale instead of the OR scale.

    Parameters
    ----------
    result : RegressionResultsWrapper
        Fitted OLS result from fit_model().

    metadata : dict
        Model information returned by fit_model().

    label_mapping : dict, optional
        Maps raw coefficient labels to display labels.
        Default (quartiles):
            {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"}

    round_digits : int, default=3

    Returns
    -------
    pd.DataFrame
        One row per non-reference predictor level (categorical) or
        one row for the predictor effect (continuous).

        Columns:
        - reference_level : display label of the reference category
        - level           : predictor level label
        - beta            : OLS regression coefficient
        - SE              : standard error
        - CI_lower        : lower 95% CI (beta scale)
        - CI_upper        : upper 95% CI (beta scale)
        - beta_95CI       : formatted "beta (lower, upper)" string
        - t_stat          : t-statistic
        - p_value         : raw p-value
        - p_value_fmt     : formatted ("<0.001" or "0.023")
        - covariates      : comma-separated list of adjustment variables
    """

    if label_mapping is None:
        label_mapping = {
            "1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"
        }

    params   = result.params
    conf_int = result.conf_int()
    p_values = result.pvalues
    t_stats  = result.tvalues

    # --------------------------------------------------
    # Keep only predictor coefficients (exclude intercept and covariates)
    # --------------------------------------------------
    coef_mask = params.index.str.contains("predictor_x", regex=False)

    betas    = params.loc[coef_mask]
    lower_ci = conf_int.loc[coef_mask, 0]
    upper_ci = conf_int.loc[coef_mask, 1]
    pvals    = p_values.loc[coef_mask]
    tstats   = t_stats.loc[coef_mask]

    results = pd.DataFrame({
        "term":     betas.index,
        "beta":     betas.values,
        "SE":       result.bse.loc[coef_mask].values,
        "CI_lower": lower_ci.values,
        "CI_upper": upper_ci.values,
        "t_stat":   tstats.values,
        "p_value":  pvals.values
    })

    # --------------------------------------------------
    # Clean labels and set reference_level
    # --------------------------------------------------
    if metadata["predictor_type"] == "categorical":
        raw_labels = (
            results["term"]
            .astype(str)
            .str.split("T.")
            .str[-1]
            .str.replace("]", "", regex=False)
        )
        results["level"]           = raw_labels.map(lambda x: label_mapping.get(x, x))
        results["reference_level"] = label_mapping.get(
            str(metadata["reference_category"]),
            str(metadata["reference_category"])
        )
    else:
        results["level"]           = metadata["predictor"]
        results["reference_level"] = None

    # --------------------------------------------------
    # Formatted CI string
    # --------------------------------------------------
    results["beta_95CI"] = results.apply(
        lambda row: (
            f"{row['beta']:.{round_digits}f} "
            f"({row['CI_lower']:.{round_digits}f}, "
            f"{row['CI_upper']:.{round_digits}f})"
        ),
        axis=1
    )

    # --------------------------------------------------
    # Formatted p-values
    # --------------------------------------------------
    results["p_value_fmt"] = results["p_value"].map(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )

    # --------------------------------------------------
    # Covariates used and SE estimator (for methods reporting)
    # --------------------------------------------------
    results["covariates"] = (
        ", ".join(metadata["covariates"]) if metadata.get("covariates") else "unadjusted"
    )
    results["cov_type"] = metadata.get("cov_type", "nonrobust")

    # --------------------------------------------------
    # Round numeric columns
    # --------------------------------------------------
    results["beta"]     = results["beta"].round(round_digits)
    results["SE"]       = results["SE"].round(round_digits)
    results["CI_lower"] = results["CI_lower"].round(round_digits)
    results["CI_upper"] = results["CI_upper"].round(round_digits)
    results["t_stat"]   = results["t_stat"].round(round_digits)

    results.attrs["predictor_type"] = metadata["predictor_type"]

    return results[[
        "reference_level",
        "level",
        "beta",
        "SE",
        "CI_lower",
        "CI_upper",
        "beta_95CI",
        "t_stat",
        "p_value",
        "p_value_fmt",
        "covariates",
        "cov_type",
    ]]


############################################################
## Forest Plot (Coefficient Plot)
############################################################
def plot_forest(
    results_df,
    title=None,
    xlabel="Coefficient (95% CI)",
    ylabel="Predictor",
    figsize=(6, 3),
    dpi=300,
    predictor_type=None,
    reference_level=None,
    save_path=None
):
    """
    Coefficient plot for OLS regression results.

    Plots on the beta scale — null reference line is at x=0 (not x=1).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from extract_results().
        Required columns: beta, CI_lower, CI_upper, p_value, level.

    title : str, optional

    xlabel : str, default="Coefficient (95% CI)"

    ylabel : str, default="Predictor"

    figsize : tuple, default=(6, 3)

    dpi : int, default=300

    predictor_type : str, optional
        "categorical" or "continuous".
        If None, inferred from results_df.attrs["predictor_type"].

    reference_level : str, optional
        Label for the reference row (categorical only).
        If None, read from results_df["reference_level"].

    save_path : str, optional
    """

    if predictor_type is None:
        predictor_type = results_df.attrs.get("predictor_type", "categorical")
    is_continuous = predictor_type == "continuous"

    if not is_continuous:
        if reference_level is None:
            reference_level = (
                results_df["reference_level"].iloc[0]
                if "reference_level" in results_df.columns
                else "Ref."
            )

    # --------------------------------------------------
    # Build arrays
    # --------------------------------------------------
    if is_continuous:
        labels   = results_df["level"].astype(str).tolist()
        betas    = results_df["beta"].to_numpy()
        ci_lower = results_df["CI_lower"].to_numpy()
        ci_upper = results_df["CI_upper"].to_numpy()
        p_values = results_df["p_value"].to_numpy()
        start_idx = 0
    else:
        labels   = [reference_level] + results_df["level"].astype(str).tolist()
        betas    = np.r_[0.0, results_df["beta"].to_numpy()]
        ci_lower = np.r_[0.0, results_df["CI_lower"].to_numpy()]
        ci_upper = np.r_[0.0, results_df["CI_upper"].to_numpy()]
        p_values = np.r_[1.0, results_df["p_value"].to_numpy()]
        start_idx = 1

    y_pos = np.arange(len(labels))

    # --------------------------------------------------
    # Figure — null line at 0 (not 1 as in ordinal)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.axvline(x=0.0, color="tab:red", linestyle="--", linewidth=2)

    xerr = np.vstack([betas - ci_lower, ci_upper - betas])

    ax.errorbar(
        betas[start_idx:], y_pos[start_idx:],
        xerr=xerr[:, start_idx:],
        fmt="o", color="steelblue", ecolor="navy",
        elinewidth=2, capsize=4, markersize=8,
        label="Coefficient (95% CI)"
    )

    if not is_continuous:
        ax.plot(0, y_pos[0], marker="s", color="#666666", markersize=7)

    ax.tick_params(axis="x", labelsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=14, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

    # Adaptive x-limits centred on range of CIs
    max_abs = max(abs(np.nanmin(ci_lower)), abs(np.nanmax(ci_upper)))
    margin  = max_abs * 1.3
    ax.set_xlim(-margin, margin)

    # --------------------------------------------------
    # Inline annotations
    # --------------------------------------------------
    for ii in range(len(labels)):
        if not is_continuous and ii == 0:
            ax.text(
                margin * 0.55, ii, "0.00 (Reference)",
                va="center", ha="left", fontsize=11, color="#666666"
            )
        else:
            pval   = p_values[ii]
            p_text = (
                "***" if pval < 0.001
                else "**" if pval < 0.01
                else "*"  if pval < 0.05
                else ""
            )
            ax.text(
                margin * 0.55, ii,
                f"{betas[ii]:.2f} ({ci_lower[ii]:.2f}, {ci_upper[ii]:.2f}){p_text}",
                va="center", ha="left", fontsize=11
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
        print(f"Figure saved to: {save_path}")

    plt.show()


############################################################
## Stratified Models (Effect Modification)
############################################################
def fit_stratified(
    data,
    outcome,
    predictor,
    stratify_by,
    strata=None,
    predictor_type="categorical",
    reference_category=None,
    covariates=None,
    scale_predictor=None,
    min_group_n=30,
    cov_type="HC3"
):
    """
    Fit separate OLS linear regression models within each stratum of a
    grouping variable to assess effect modification (interaction).

    Use this to answer: "Is the association between green exposure and
    mental health score stronger or weaker for people with low SES?"

    Each stratum receives its own model. Comparing coefficients across
    strata reveals whether the exposure–outcome association is more
    pronounced in specific subgroups.

    For a formal statistical test of interaction, use fit_interaction().

    Parameters
    ----------
    data : pd.DataFrame

    outcome : str
        Continuous numeric outcome (e.g. "CES-D Score").

    predictor : str
        Exposure variable (e.g. "Green Space Quartile (100m Buffer)").

    stratify_by : str
        Grouping variable defining the strata.
        Example: "Socioeconomic Status (Tiers)"
            0 = low SES, 1 = mid SES, 2 = high SES

    strata : list, optional
        Specific stratum values to include.
        If None, all unique values of stratify_by are used.
        Example: [0, 1, 2]

    predictor_type : str, default="categorical"
        Passed to fit_model().

    reference_category : str, optional
        Passed to fit_model().

    covariates : list[str], optional
        Adjustment variables other than stratify_by.
        stratify_by is never included as a covariate in stratified
        models — the data is already split by it.

    scale_predictor : float, optional
        Passed to fit_model() for continuous predictors.

    min_group_n : int, default=30
        Minimum sample size required within a stratum to fit a model.
        Strata with fewer observations are skipped with a message
        rather than raising an error.

    cov_type : str, default="HC3"
        Covariance estimator passed to fit_model() for every stratum.
        See fit_model()'s docstring for details.

    Returns
    -------
    results : dict
        Keyed by stratum value. Each entry:
            - "result"    : RegressionResultsWrapper
            - "metadata"  : dict  (includes stratum info)
            - "beta_table": pd.DataFrame from extract_results()
            - "n"         : int

    summary : pd.DataFrame
        Coefficient tables from all strata stacked, with a "stratum"
        column added. Ready for comparison or plotting.
    """

    subset_data = data.dropna(subset=[stratify_by])

    if strata is None:
        try:
            strata = sorted(subset_data[stratify_by].unique())
        except TypeError:
            strata = list(subset_data[stratify_by].unique())

    # Ensure stratify_by is not also in covariates
    covariates = list(covariates) if covariates else []
    if stratify_by in covariates:
        covariates = [c for c in covariates if c != stratify_by]

    results     = {}
    beta_tables = []

    for stratum in strata:

        stratum_df = subset_data[subset_data[stratify_by] == stratum].copy()

        n_stratum = len(stratum_df)
        print(f"  Stratum {stratify_by} = {stratum}  (n = {n_stratum})")

        if n_stratum < min_group_n:
            print(
                f"    [skip] n={n_stratum} is below min_group_n={min_group_n} "
                "— insufficient data to fit model."
            )
            continue

        try:
            result, metadata = fit_model(
                data               = stratum_df,
                outcome            = outcome,
                predictor          = predictor,
                predictor_type     = predictor_type,
                reference_category = reference_category,
                covariates         = covariates,
                scale_predictor    = scale_predictor,
                cov_type           = cov_type
            )

            # Add stratum info to metadata
            metadata["stratum_var"]   = stratify_by
            metadata["stratum_value"] = stratum

            beta_table = extract_results(result, metadata)
            beta_table.insert(0, "stratum", stratum)

            results[stratum] = {
                "result":     result,
                "metadata":   metadata,
                "beta_table": beta_table,
                "n":          metadata["n"]
            }

            beta_tables.append(beta_table)

        except Exception as e:
            print(f"    [error] stratum {stratum}: {e}")

    summary = (
        pd.concat(beta_tables, axis=0, ignore_index=True)
        if beta_tables else pd.DataFrame()
    )

    return results, summary


############################################################
## Interaction Term (Effect Modification)
############################################################
def fit_interaction(
    data,
    outcome,
    predictor,
    moderator,
    moderator_type="categorical",
    moderator_reference=None,
    predictor_type="categorical",
    scale_predictor=None,
    reference_category=None,
    label_mapping=None,
    covariates=None,
    cov_type="HC3"
):
    """
    Fit an OLS linear regression model with a predictor × moderator
    interaction term to formally test effect modification.

    Use this to answer: "Does SES statistically modify the association
    between green exposure and mental health score?"

    A significant interaction p-value means the exposure–outcome
    association differs across moderator levels — i.e. effect
    modification is present. Pair this with fit_stratified() to
    visualize the direction and magnitude in each stratum.

    Parameters
    ----------
    data : pd.DataFrame

    outcome : str
        Continuous numeric outcome (e.g. "CES-D Score").

    predictor : str
        Primary exposure (e.g. "Green Space Quartile (500m Buffer)").

    moderator : str
        Effect modifier to test (e.g. "Socioeconomic Status (Tiers)").

    moderator_type : str, default="categorical"
        "categorical" or "continuous".
        For ordinal moderators like SES (1/2/3), "categorical" uses
        dummy coding; "continuous" assumes a linear moderating effect.

    moderator_reference : str/int, optional
        Reference level for a categorical moderator.
        Example: 1 (low SES as reference).

    predictor_type : str, default="categorical"
        "categorical" or "continuous".

    scale_predictor : float, optional
        Scale factor for continuous predictor (e.g. 0.1 for NDVI).

    reference_category : str, optional
        Reference level for a categorical predictor.

    label_mapping : dict, optional
        Maps raw coefficient labels to display labels.
        Default (quartiles):
            {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"}

    covariates : list[str], optional
        Additional adjustment variables (not interacted).

    cov_type : str, default="HC3"
        Covariance estimator passed to statsmodels .fit(cov_type=...).
        Matches fit_model()'s default — see its docstring for details.

    Returns
    -------
    result : RegressionResultsWrapper
        Fitted model with interaction term.

    metadata : dict
        Standard metadata plus:
            - "moderator"           : str
            - "moderator_type"      : str
            - "moderator_reference" : str or None
            - "formula"             : str — full patsy formula used

    interaction_table : pd.DataFrame
        Coefficients, CIs, and p-values for the interaction terms only
        — the key output for assessing effect modification.

        Columns:
            outcome, predictor, predictor_type, moderator, scale_predictor,
            reference_level, level, moderator_level,
            beta, SE, CI_lower, CI_upper, beta_95CI,
            p_value, p_value_fmt, covariates, significant

        `level` and `moderator_level` together identify each row: with
        a categorical moderator that has more than two levels (e.g.
        SES tiers 1 and 2 both vs reference 0), `level` alone repeats
        across rows — `moderator_level` distinguishes them.

    Notes
    -----
    Interpretation of interaction terms:
        Categorical predictor × categorical moderator:
            Each interaction coefficient is the *additional* change
            in outcome for that predictor level in that moderator
            group, relative to the reference group.

        Continuous predictor × categorical moderator:
            The interaction coefficient is the *additional* slope
            per unit of predictor for that moderator group.

    A significant interaction p-value (< 0.05) supports effect
    modification. Use fit_stratified() alongside this to inspect
    stratum-specific coefficients.
    """

    covariates = list(covariates) if covariates else []

    # Exclude moderator from covariates — it enters via interaction
    covariates = [c for c in covariates if c != moderator]

    model_cols = [outcome, predictor, moderator] + covariates
    subset     = data[model_cols].dropna().copy()

    if subset.empty:
        raise ValueError(
            f"No complete cases for outcome='{outcome}', "
            f"predictor='{predictor}', moderator='{moderator}'."
        )

    # --------------------------------------------------
    # Predictor
    # --------------------------------------------------
    if predictor_type == "categorical":
        subset[predictor] = subset[predictor].astype(str)
        raw_levels = subset[predictor].unique()
        try:
            predictor_levels = sorted(raw_levels, key=lambda v: float(v))
        except ValueError:
            predictor_levels = sorted(raw_levels)
        
        if reference_category is None:
            reference_category = predictor_levels[0]
        else:
            reference_category = str(reference_category)

    elif predictor_type == "continuous":
        subset[predictor] = pd.to_numeric(subset[predictor], errors="coerce")
        subset            = subset.dropna(subset=[predictor])
        predictor_levels  = None
        reference_category = None
        if scale_predictor is not None:
            subset[predictor] = subset[predictor] / scale_predictor
    else:
        raise ValueError("predictor_type must be 'categorical' or 'continuous'.")

    # --------------------------------------------------
    # Moderator
    # --------------------------------------------------
    if moderator_type == "categorical":
        subset[moderator] = subset[moderator].astype(str)
        mod_levels = subset[moderator].unique()
        try:
            mod_levels = sorted(mod_levels, key=lambda v: float(v))
        except ValueError:
            mod_levels = sorted(mod_levels)
        
        if moderator_reference is None:
            moderator_reference = mod_levels[0]
        else:
            moderator_reference = str(moderator_reference)
    else:
        subset[moderator] = pd.to_numeric(subset[moderator], errors="coerce")
        subset            = subset.dropna(subset=[moderator])
        mod_levels        = None

    # --------------------------------------------------
    # Rename for formula
    # --------------------------------------------------
    rename_dict = {
        outcome:   "outcome_y",
        predictor: "predictor_x",
        moderator: "moderator_z"
    }
    working_df = subset.rename(columns=rename_dict)

    # --------------------------------------------------
    # Build formula with interaction
    # --------------------------------------------------
    if predictor_type == "categorical":
        ref = reference_category.replace('"', '\\"')
        pred_term = f'C(predictor_x, Treatment(reference="{ref}"))'
    else:
        pred_term = "predictor_x"

    if moderator_type == "categorical":
        mod_ref  = str(moderator_reference).replace('"', '\\"')
        mod_term = f'C(moderator_z, Treatment(reference="{mod_ref}"))'
    else:
        mod_term = "moderator_z"

    # Main effects + interaction
    formula = f"outcome_y ~ {pred_term} + {mod_term} + {pred_term}:{mod_term}"

    if covariates:
        covariate_terms = " + ".join(f'Q("{c}")' for c in covariates)
        formula += " + " + covariate_terms

    # --------------------------------------------------
    # Fit model
    # --------------------------------------------------
    model  = smf.ols(formula, data=working_df)
    result = model.fit(cov_type=cov_type)

    # --------------------------------------------------
    # Extract interaction terms only
    # --------------------------------------------------
    params   = result.params
    conf_int = result.conf_int()
    pvals    = result.pvalues
    ses      = result.bse

    interaction_mask = params.index.str.contains(
        "predictor_x.*moderator_z|moderator_z.*predictor_x", regex=True
    )

    if not interaction_mask.any():
        # Fallback: any term containing ":"
        interaction_mask = params.index.str.contains(":", regex=False)

    betas    = params.loc[interaction_mask]
    lower_ci = conf_int.loc[interaction_mask, 0]
    upper_ci = conf_int.loc[interaction_mask, 1]
    p_values = pvals.loc[interaction_mask]
    se_vals  = ses.loc[interaction_mask]

    interaction_table = pd.DataFrame({
        "term":        betas.index,
        "beta":        betas.values.round(3),
        "SE":          se_vals.values.round(3),
        "CI_lower":    lower_ci.values.round(3),
        "CI_upper":    upper_ci.values.round(3),
        "beta_95CI":   [
            f"{b:.3f} ({l:.3f}, {u:.3f})"
            for b, l, u in zip(betas.values, lower_ci.values, upper_ci.values)
        ],
        "p_value":     p_values.values,
        "p_value_fmt": [
            "<0.001" if p < 0.001 else f"{p:.3f}" for p in p_values.values
        ],
        "significant": p_values.values < 0.05
    })

    # --------------------------------------------------
    # Clean coefficient labels and set reference_level
    # Categorical: predictor_x[T.2.0]:moderator_z[T.1] → Q2
    # Continuous:  predictor_x → predictor name
    # --------------------------------------------------
    if label_mapping is None:
        label_mapping = {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"}

    if predictor_type == "categorical":
        raw_labels = (
            interaction_table["term"]
            .astype(str)
            .str.split(":").str[0]
            .str.split("T.")
            .str[-1]
            .str.replace("]", "", regex=False)
        )
        interaction_table["level"]           = raw_labels.map(lambda x: label_mapping.get(x, x))
        interaction_table["reference_level"] = label_mapping.get(
            str(reference_category), str(reference_category)
        )
    else:
        # For continuous variables, use the feature name
        interaction_table["level"]           = predictor
        interaction_table["reference_level"] = None

    # --------------------------------------------------
    # Extract moderator_level — the term's SECOND (moderator) component.
    # --------------------------------------------------
    if moderator_type == "categorical":
        interaction_table["moderator_level"] = (
            interaction_table["term"]
            .astype(str)
            .str.split(":").str[-1]
            .str.split("T.")
            .str[-1]
            .str.replace("]", "", regex=False)
        )
    else:
        interaction_table["moderator_level"] = moderator

    metadata = {
        "n":                   len(working_df),
        "outcome":             outcome,
        "predictor":           predictor,
        "predictor_type":      predictor_type,
        "reference_category":  reference_category,
        "predictor_levels":    predictor_levels,
        "moderator":           moderator,
        "moderator_type":      moderator_type,
        "moderator_reference": moderator_reference,
        "scale_predictor":     scale_predictor,
        "covariates":          covariates,
        "cov_type":            cov_type,
        "formula":             formula
    }

    interaction_table["outcome"]         = outcome
    interaction_table["predictor"]       = predictor
    interaction_table["predictor_type"]  = predictor_type
    interaction_table["moderator"]       = moderator
    interaction_table["scale_predictor"] = scale_predictor

    interaction_table["covariates"] = (
        ", ".join(covariates) if covariates else "unadjusted"
    )
    interaction_table["cov_type"] = cov_type

    interaction_table = interaction_table[[
        "outcome",
        "predictor",
        "predictor_type",
        "moderator",
        "scale_predictor",
        "reference_level",
        "level",
        "moderator_level",
        "beta",
        "SE",
        "CI_lower",
        "CI_upper",
        "beta_95CI",
        "p_value",
        "p_value_fmt",
        "covariates",
        "cov_type",
        "significant",
    ]]

    n_sig = interaction_table["significant"].sum()
    print(
        f"  Interaction fitted | n = {len(working_df)} | "
        f"{n_sig}/{len(interaction_table)} interaction terms p < 0.05"
    )

    return result, metadata, interaction_table


############################################################
## Residual Diagnostics
############################################################
def check_residuals(
    result,
    metadata=None,
    alpha=0.05,
    figsize=(12, 4),
    dpi=300,
    save_path=None
):
    """
    Diagnostic checks for OLS linear regression residuals.

    Reports:
        1) Breusch-Pagan test for heteroskedasticity
           (motivates the choice of a robust covariance estimator,
           e.g. cov_type="HC3" in fit_model())
        2) Shapiro-Wilk test for residual normality
        3) Durbin-Watson statistic for autocorrelation of residuals
           in the order the data was fitted (NOT a spatial test —
           see Notes)
        4) Residuals-vs-fitted plot (detects non-linearity, heteroskedasticity)
        5) Q-Q plot of residuals (detects non-normality)

    Parameters
    ----------
    result : RegressionResultsWrapper
        Fitted OLS result from fit_model().

    metadata : dict, optional
        If provided, used to label the diagnostic plot title with the
        outcome and predictor names.

    alpha : float, default=0.05
        Significance threshold for the Breusch-Pagan and Shapiro-Wilk
        verdicts.

    figsize : tuple, default=(12, 4)

    dpi : int, default=300

    save_path : str, optional
        If provided, save the diagnostic figure to this path as PNG.

    Returns
    -------
    dict
        - breusch_pagan  : dict — {"statistic", "p_value", "heteroskedastic"}
        - shapiro        : dict — {"statistic", "p_value", "normal"}
        - durbin_watson  : float — ~2.0 indicates no autocorrelation;
                           <1.5 or >2.5 suggests positive/negative
                           autocorrelation in fit order
        - fig            : matplotlib.figure.Figure

    Notes
    -----
    Durbin-Watson tests autocorrelation in the ORDER the data appears in
    the design matrix (e.g. time order, or arbitrary row order) — it is
    NOT a test of spatial autocorrelation. To assess whether residuals
    are spatially clustered (e.g. neighbouring participants having
    correlated residuals), compute Moran's I on the residuals joined to
    participant coordinates, using a spatial weights matrix (e.g. via
    `libpysal`/`esda`). This requires participant-level geometry and is
    not implemented here — flag as a limitation if not performed, or
    add a separate spatial-diagnostics utility if coordinates are
    available.
    """

    resid  = result.resid
    fitted = result.fittedvalues
    exog   = result.model.exog

    # --------------------------------------------------
    # 1. Breusch-Pagan test for heteroskedasticity
    # --------------------------------------------------
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)
    breusch_pagan = {
        "statistic":       round(bp_stat, 4),
        "p_value":         round(bp_p, 4),
        "heteroskedastic": bp_p < alpha
    }

    # --------------------------------------------------
    # 2. Shapiro-Wilk test for residual normality
    # --------------------------------------------------
    # Shapiro-Wilk is unreliable for very large n; subsample if needed
    resid_for_test = (
        resid if len(resid) <= 5000
        else resid.sample(5000, random_state=0)
    )
    sw_stat, sw_p = scipy_stats.shapiro(resid_for_test)
    shapiro = {
        "statistic": round(sw_stat, 4),
        "p_value":   round(sw_p, 4),
        "normal":    sw_p >= alpha
    }

    # --------------------------------------------------
    # 3. Durbin-Watson (fit-order autocorrelation, NOT spatial)
    # --------------------------------------------------
    dw_stat = round(durbin_watson(resid), 4)

    # --------------------------------------------------
    # 4-5. Diagnostic plots
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    axes[0].scatter(fitted, resid, alpha=0.4, s=12, edgecolor="none")
    axes[0].axhline(0, color="tab:red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    sm.qqplot(resid, line="s", ax=axes[1])
    axes[1].set_title("Normal Q-Q")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    if metadata:
        fig.suptitle(
            f"Residual diagnostics: {metadata.get('outcome', '')} ~ "
            f"{metadata.get('predictor', '')}",
            fontweight="bold"
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
        print(f"Figure saved to: {save_path}")

    plt.show()

    print(
        f"  Breusch-Pagan: stat={breusch_pagan['statistic']}, "
        f"p={breusch_pagan['p_value']} "
        f"({'heteroskedastic' if breusch_pagan['heteroskedastic'] else 'homoskedastic'})"
    )
    print(
        f"  Shapiro-Wilk:  stat={shapiro['statistic']}, "
        f"p={shapiro['p_value']} "
        f"({'non-normal' if not shapiro['normal'] else 'normal'} residuals)"
    )
    print(f"  Durbin-Watson: {dw_stat} (fit-order autocorrelation; ~2.0 = none)")

    return {
        "breusch_pagan": breusch_pagan,
        "shapiro":       shapiro,
        "durbin_watson": dw_stat,
        "fig":           fig
    }
