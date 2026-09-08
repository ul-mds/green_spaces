"""
ordinal_regression.py

Utility functions for fitting, extracting, and visualizing proportional-odds
ordinal logistic regression models via statsmodels.

Supports both categorical exposures (e.g. greenspace quartiles) and
continuous exposures (e.g. NDVI), including per-unit scaling for continuous
predictors.

Location: utils/ordinal_regression.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel

__all__ = [
    "fit_model",
    "extract_results",
    "plot_forest",
    "plot_profile",
    "fit_stratified",
    "fit_interaction",
]


############################################################
## Fit Ordinal Logistic Regression
############################################################
def fit_model(
    data,
    outcome,
    predictor,
    predictor_type="categorical",
    reference_category=None,
    covariates=None,
    distribution="logit",
    method="bfgs",
    scale_predictor=None
):
    """
    Fit a proportional-odds ordinal regression model.

    Supports categorical predictors (e.g. greenspace quartiles coded as
    1.0, 2.0, 3.0, 4.0) and continuous predictors (e.g. NDVI).

    Parameters
    ----------
    data : pd.DataFrame

    outcome : str
        Ordered categorical outcome variable.

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
        Adjustment variables included as additive terms in the formula.
        Example: ["Age", "Sex", "Socioeconomic Status (Tiers)"]

    distribution : str, default="logit"
        Link function for OrderedModel: "logit" or "probit".

    method : str, default="bfgs"
        Optimization method passed to statsmodels .fit().

    scale_predictor : float, optional
        Divide the continuous predictor by this value before fitting.
        Only relevant for continuous predictors.
        Example:
            scale_predictor=0.1 → OR is interpretable as effect per
            0.1-unit increase in NDVI.

    Returns
    -------
    result : OrderedResults
        Fitted statsmodels result object.

    metadata : dict
        - n                 : int   — analytic sample size
        - outcome           : str
        - predictor         : str
        - predictor_type    : str   — "categorical" or "continuous"
        - reference_category: str or None
        - covariates        : list[str]
        - outcome_levels    : list  — ordered outcome categories
        - predictor_levels  : list or None — sorted levels (categorical only)
        - distribution      : str
        - scale_predictor   : float or None
    """

    if covariates is None:
        covariates = []
    elif isinstance(covariates, str):
        covariates = [covariates]
    else:
        covariates = list(covariates)

    model_variables = [outcome, predictor] + covariates
    subset = data[model_variables].dropna().copy()

    # --------------------------------------------------
    # Outcome preparation
    # --------------------------------------------------
    subset[outcome] = pd.Categorical(subset[outcome], ordered=True)
    outcome_levels  = list(subset[outcome].cat.categories)

    if len(outcome_levels) < 3:
        raise ValueError(
            "Ordinal logistic regression requires at least three "
            "ordered outcome levels."
        )

    # --------------------------------------------------
    # Predictor preparation
    # --------------------------------------------------
    if predictor_type == "categorical":
        subset[predictor] = subset[predictor].astype(str)

        # Sort numerically where possible, otherwise alphabetically
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
        subset[predictor] = pd.to_numeric(subset[predictor], errors="coerce")
        subset = subset.dropna(subset=[predictor])

        if subset[predictor].nunique() < 2:
            raise ValueError("Continuous predictor must have variation.")

        predictor_levels   = None
        reference_category = None

        if scale_predictor is not None:
            subset[predictor] = subset[predictor] / scale_predictor

    else:
        raise ValueError(
            "predictor_type must be 'continuous' or 'categorical'."
        )

    # --------------------------------------------------
    # Rename variables for formula
    # --------------------------------------------------
    rename_dict = {outcome: "outcome_y", predictor: "predictor_x"}
    working_df  = subset.rename(columns=rename_dict)

    # --------------------------------------------------
    # Build formula
    # Double-quote reference value to handle strings with spaces/special chars
    # --------------------------------------------------
    if predictor_type == "categorical":
        ref     = reference_category.replace('"', '\\"')
        formula = f'outcome_y ~ C(predictor_x, Treatment(reference="{ref}"))'
    else:
        formula = f"outcome_y ~ predictor_x"

    if covariates:
        covariate_terms = " + ".join(f'Q("{cov}")' for cov in covariates)
        formula += " + " + covariate_terms

    # --------------------------------------------------
    # Fit model
    # --------------------------------------------------
    model = OrderedModel.from_formula(
        formula,
        data=working_df,
        distr=distribution
    )

    result = model.fit(method=method, disp=False)

    metadata = {
        "n":                  len(working_df),
        "outcome":            outcome,
        "predictor":          predictor,
        "predictor_type":     predictor_type,
        "reference_category": reference_category,
        "covariates":         covariates,
        "outcome_levels":     outcome_levels,
        "predictor_levels":   predictor_levels,
        "distribution":       distribution,
        "scale_predictor":    scale_predictor
    }

    return result, metadata


############################################################
## Extract Odds Ratios, CIs, p-values, Interpretation
############################################################
def extract_results(
    result,
    metadata,
    outcome_direction=None,
    interpretation_label=None,
    label_mapping=None,
    round_digits=3
):
    """
    Extract odds ratios from a fitted ordinal regression model.

    Parameters
    ----------
    result : OrderedResults
        Fitted statsmodels result object from fit_model().

    metadata : dict
        Model information returned by fit_model().

    outcome_direction : str, optional
        "higher_is_worse" or "higher_is_better".
        If None, auto-inferred:
            "SWLS" in outcome name → "higher_is_better"
            all others             → "higher_is_worse"

    interpretation_label : str, optional
        Custom phrase for the interpretation column.
        If None, auto-generated from outcome_direction and outcome name.
        Example: "more severe depressive symptoms"

    label_mapping : dict, optional
        Maps raw statsmodels coefficient labels to display labels.
        Only relevant for categorical predictors.
        Default (quartiles):
            {"1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"}
        Override for tertiles, quintiles, etc.:
            {"1.0": "T1", "2.0": "T2", "3.0": "T3"}

    round_digits : int, default=3
        Decimal places for rounded output columns.

    Returns
    -------
    pd.DataFrame
        One row per non-reference predictor level (categorical) or
        one row for the predictor effect (continuous).

        Columns:
        - reference_level  : human-readable reference label (categorical only)
        - level            : predictor level label
        - beta             : log-odds coefficient
        - OR               : odds ratio
        - CI_lower         : lower 95% CI (OR scale)
        - CI_upper         : upper 95% CI (OR scale)
        - OR_95CI          : formatted "OR (lower, upper)" string
        - p_value          : raw p-value
        - p_value_fmt      : formatted p-value ("<0.001" or "0.023")
        - pct_change_odds  : absolute % change in odds vs reference
        - interpretation   : plain-language interpretation of the OR

    Note
    ----
    The `reference_level` column is None for continuous predictors
    (no reference category applies).
    The DataFrame `.attrs["predictor_type"]` is set for use by
    plot_forest(); pass predictor_type explicitly to plot_forest()
    when calling on a concatenated DataFrame (attrs are not preserved
    after pd.concat).
    """

    # --------------------------------------------------
    # Outcome direction
    # --------------------------------------------------
    if outcome_direction is None:
        outcome_direction = (
            "higher_is_better"
            if "SWLS" in metadata["outcome"].upper()
            else "higher_is_worse"
        )

    if outcome_direction not in {"higher_is_worse", "higher_is_better"}:
        raise ValueError(
            "outcome_direction must be 'higher_is_worse' or 'higher_is_better'."
        )

    if interpretation_label is None:
        interpretation_label = (
            f"poorer outcome ({metadata['outcome']})"
            if outcome_direction == "higher_is_worse"
            else f"better outcome ({metadata['outcome']})"
        )

    # --------------------------------------------------
    # Default label mapping (quartiles, including reference)
    # --------------------------------------------------
    if label_mapping is None:
        label_mapping = {
            "1.0": "Q1", "2.0": "Q2", "3.0": "Q3", "4.0": "Q4"
        }

    params   = result.params
    conf_int = result.conf_int()
    p_values = result.pvalues

    # --------------------------------------------------
    # Keep only predictor coefficients (exclude thresholds)
    # --------------------------------------------------
    coef_mask = params.index.str.contains("predictor_x", regex=False)

    betas    = params.loc[coef_mask]
    lower_ci = conf_int.loc[coef_mask, 0]
    upper_ci = conf_int.loc[coef_mask, 1]
    pvals    = p_values.loc[coef_mask]

    results = pd.DataFrame({
        "term":     betas.index,
        "beta":     betas.values,
        "OR":       np.exp(betas.values),
        "CI_lower": np.exp(lower_ci.values),
        "CI_upper": np.exp(upper_ci.values),
        "p_value":  pvals.values
    })

    # --------------------------------------------------
    # Clean coefficient labels and set reference_level
    # Categorical: C(predictor_x, Treatment(...))[T.2.0] → Q2
    # Continuous:  predictor_x → predictor name
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
        # For continuous variables, use the feature name
        results["level"]           = metadata["predictor"]
        results["reference_level"] = None

    # --------------------------------------------------
    # Percent change in odds vs reference (OR = 1)
    # OR < 1 → reduction
    # OR > 1 → increase
    # --------------------------------------------------
    results["pct_change_odds"] = np.where(
        results["OR"] < 1,
        (1 - results["OR"]) * 100,
        (results["OR"] - 1) * 100
    )

    # --------------------------------------------------
    # Plain-language interpretation
    # --------------------------------------------------
    def _interpret_or(or_value):
        if np.isclose(or_value, 1.0):
            return "No meaningful difference in odds"
        change_pct = abs(1 - or_value) * 100
        direction  = "lower" if or_value < 1 else "higher"
        if metadata["predictor_type"] == "categorical":
            return (
                f"{change_pct:.1f}% {direction} odds of {interpretation_label} "
                f"compared to reference group"
            )
        else:
            unit = (
                f"per {metadata['scale_predictor']}-unit increase"
                if metadata["scale_predictor"]
                else f"per 1-unit increase"
            )
            return (
                f"{change_pct:.1f}% {direction} odds of {interpretation_label} "
                f"{unit} in {metadata['predictor']}"
            )

    results["interpretation"] = results["OR"].map(_interpret_or)

    # --------------------------------------------------
    # Formatted p-values
    # --------------------------------------------------
    results["p_value_fmt"] = results["p_value"].map(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )

    # --------------------------------------------------
    # Formatted OR (95% CI) for manuscript tables
    # --------------------------------------------------
    results["OR_95CI"] = results.apply(
        lambda row: (
            f"{row['OR']:.{round_digits}f} "
            f"({row['CI_lower']:.{round_digits}f}, "
            f"{row['CI_upper']:.{round_digits}f})"
        ),
        axis=1
    )

    # --------------------------------------------------
    # Round numeric columns (keep raw p_value for sorting)
    # --------------------------------------------------
    results["beta"]            = results["beta"].round(round_digits)
    results["OR"]              = results["OR"].round(round_digits)
    results["CI_lower"]        = results["CI_lower"].round(round_digits)
    results["CI_upper"]        = results["CI_upper"].round(round_digits)
    results["pct_change_odds"] = results["pct_change_odds"].round(1)

    # Store predictor_type in attrs for plot_forest() when called directly.
    # Note: attrs are not preserved after pd.concat — pass predictor_type
    # explicitly to plot_forest() when working with combined results.
    results.attrs["predictor_type"] = metadata["predictor_type"]

    # Covariates used in this model (empty string = unadjusted)
    results["covariates"] = (
        ", ".join(metadata["covariates"]) if metadata.get("covariates") else "unadjusted"
    )

    return results[[
        "reference_level",
        "level",
        "beta",
        "OR",
        "CI_lower",
        "CI_upper",
        "OR_95CI",
        "p_value",
        "p_value_fmt",
        "pct_change_odds",
        "interpretation",
        "covariates",
    ]]


############################################################
## Forest Plot
############################################################
def plot_forest(
    results_df,
    title=None,
    xlabel="Odds Ratio (95% CI)",
    ylabel="Green Exposure",
    figsize=(6, 3),
    dpi=300,
    predictor_type=None,
    reference_level=None,
    save_path=None
):
    """
    Create a forest plot of odds ratios from an ordinal regression model.

    Works for both categorical and continuous predictors.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from extract_results().
        Required columns: OR, CI_lower, CI_upper, p_value, level.
        Optional column: reference_level.

    title : str, optional

    xlabel : str, default="Odds Ratio (95% CI)"

    ylabel : str, default="Green Exposure"

    figsize : tuple, default=(6, 3)

    dpi : int, default=300

    predictor_type : str, optional
        "categorical" or "continuous".
        If None, inferred from results_df.attrs["predictor_type"].
        Pass explicitly when results_df comes from pd.concat (attrs lost).

    reference_level : str, optional
        Label for the reference row (categorical only).
        If None, uses the "reference_level" column from results_df.
        Override with e.g. "T1" for tertiles if label_mapping was custom.

    save_path : str, optional
        File path to save the figure as PNG.
    """

    # --------------------------------------------------
    # Resolve predictor type
    # --------------------------------------------------
    if predictor_type is None:
        predictor_type = results_df.attrs.get("predictor_type", "categorical")
    is_continuous = predictor_type == "continuous"

    # --------------------------------------------------
    # Resolve reference label for categorical plots
    # --------------------------------------------------
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
        labels    = results_df["level"].astype(str).tolist()
        or_values = results_df["OR"].to_numpy()
        ci_lower  = results_df["CI_lower"].to_numpy()
        ci_upper  = results_df["CI_upper"].to_numpy()
        p_values  = results_df["p_value"].to_numpy()
        start_idx = 0
    else:
        # Categorical workflows require injecting a reference line at the top
        labels    = [reference_level] + results_df["level"].astype(str).tolist()
        or_values = np.r_[1.0, results_df["OR"].to_numpy()]
        ci_lower  = np.r_[1.0, results_df["CI_lower"].to_numpy()]
        ci_upper  = np.r_[1.0, results_df["CI_upper"].to_numpy()]
        p_values  = np.r_[1.0, results_df["p_value"].to_numpy()]
        start_idx = 1

    y_pos = np.arange(len(labels))

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.axvline(x=1.0, color="tab:red", linestyle="--", linewidth=2)

    xerr = np.vstack([or_values - ci_lower, ci_upper - or_values])

    ax.errorbar(
        or_values[start_idx:], y_pos[start_idx:],
        xerr=xerr[:, start_idx:],
        fmt="o", color="forestgreen", ecolor="darkgreen",
        elinewidth=2, capsize=4, markersize=8, 
        label="Odds Ratio (95% CI)"
    )

    # Plot reference point if categorical
    if not is_continuous:
        ax.plot(1, y_pos[0], marker="s", color="#666666", markersize=7)

    # --------------------------------------------------
    # Axes formatting
    # --------------------------------------------------
    ax.tick_params(axis="x", labelsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    
    ax.set_xlabel(xlabel, fontsize=14, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

    # --------------------------------------------------
    # X-axis limits — adaptive with margin
    # --------------------------------------------------
    max_ci_val = np.nanmax(ci_upper)
    x_limit    = max(2.0, int(np.ceil(max_ci_val * 1.2)))
    ax.set_xlim(0, x_limit)

    # --------------------------------------------------
    # Text annotations (OR and CI inline)
    # --------------------------------------------------
    for ii in range(len(labels)):
        if not is_continuous and ii == 0:
            ax.text(
                x_limit * 0.72, ii, "1.00 (Reference)",
                va="center", ha="left", fontsize=11, color="#666666"
            )
        else:
            pval   = p_values[ii]
            p_text = (
                "***" if pval < 0.001     # Extremely significant
                else "**" if pval < 0.01  # Highly significant
                else "*"  if pval < 0.05  # Statistically significant
                else ""                   # Not statistically significant
            )
            ax.text(
                x_limit * 0.72, ii,
                f"{or_values[ii]:.2f} ({ci_lower[ii]:.2f}, "
                f"{ci_upper[ii]:.2f}){p_text}",
                va="center", ha="left", fontsize=11
            )

    # --------------------------------------------------
    # Spines formatting
    # --------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)

    plt.tight_layout()

    # --------------------------------------------------
    # Save the figure if a path is provided
    # --------------------------------------------------
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
        print(f"Figure saved to: {save_path}")

    plt.show()


############################################################
## Ordinal Profile Plot
############################################################
def plot_profile(
    data,
    outcome,
    predictor,
    predictor_type="categorical",
    continuous_bins=4,
    outcome_order=None,
    predictor_order=None,
    outcome_labels=None,
    predictor_labels=None,
    normalize=True,
    percent=False,
    title=None,
    xlabel=None,
    legend_title=None,
    figsize=(6, 3),
    dpi=300,
    save_path=None
):
    """
    Plot the distribution of an ordinal outcome across ordered exposure levels.

    Intended as a descriptive companion to ordinal logistic regression.
    For continuous predictors, the variable is auto-binned into quantile
    groups before cross-tabulation.

    Parameters
    ----------
    data : pd.DataFrame

    outcome : str
        Ordinal outcome variable.

    predictor : str
        Exposure variable (categorical or continuous).

    predictor_type : str, default="categorical"
        "categorical" or "continuous".
        For "continuous", the predictor is binned into `continuous_bins`
        quantile groups before plotting.

    continuous_bins : int, default=4
        Number of quantile bins for continuous predictors (e.g. 4 = quartiles).

    outcome_order : list, optional
        Explicit order of outcome categories, e.g. [0, 1, 2].

    predictor_order : list, optional
        Explicit order of predictor categories, e.g. [1, 2, 3, 4].
        Auto-set from quantile bins when predictor_type="continuous".

    outcome_labels : dict, optional
        Mapping from outcome codes to display labels.
        Example: {0: "High", 1: "Moderate", 2: "Low"}

    predictor_labels : dict or list, optional
        Mapping from predictor codes/bins to display labels.
        Example: {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}

    normalize : bool, default=True
        If True, plot within-predictor proportions; if False, raw counts.

    percent : bool, default=False
        If True and normalize=True, scale proportions to 0–100.

    title : str, optional

    xlabel : str, optional
        Defaults to the predictor column name.

    legend_title : str, optional
        Defaults to the outcome column name.

    figsize : tuple, default=(6, 3)

    dpi : int, default=300

    save_path : str, optional
        If provided, save figure to this path as PNG.

    Returns
    -------
    table : pd.DataFrame
        Cross-tabulation used for plotting.

    fig : matplotlib.figure.Figure

    ax : matplotlib.axes.Axes
    """

    subset = data[[outcome, predictor]].dropna().copy()

    # --------------------------------------------------
    # Bin continuous predictor into quantile groups
    # --------------------------------------------------
    if predictor_type == "continuous":
        bin_labels = (
            [f"Bin {i+1}" for i in range(continuous_bins)]
            if not predictor_labels
            else None
        )
        subset[predictor] = pd.qcut(
            subset[predictor],
            q=continuous_bins,
            labels=bin_labels,
            duplicates="drop"
        )
        if predictor_order is None:
            predictor_order = list(subset[predictor].cat.categories)

    # --------------------------------------------------
    # Apply explicit ordering
    # --------------------------------------------------
    if outcome_order is not None:
        subset[outcome] = pd.Categorical(
            subset[outcome], categories=outcome_order, ordered=True
        )

    if predictor_order is not None:
        subset[predictor] = pd.Categorical(
            subset[predictor], categories=predictor_order, ordered=True
        )

    # --------------------------------------------------
    # Cross-tabulation
    # --------------------------------------------------
    table = pd.crosstab(
        subset[predictor],
        subset[outcome],
        normalize="index" if normalize else False
    )

    if outcome_order is not None:
        table = table.reindex(columns=outcome_order)

    if predictor_order is not None:
        table = table.reindex(predictor_order)

    if normalize and percent:
        table = table * 100

    # --------------------------------------------------
    # Labels for plotting
    # --------------------------------------------------
    x_values = list(table.index)
    x_labels = [
        predictor_labels.get(x, x) if isinstance(predictor_labels, dict)
        else x
        for x in x_values
    ]

    outcome_columns = list(table.columns)
    outcome_display = [
        outcome_labels.get(y, y) if outcome_labels else y
        for y in outcome_columns
    ]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for col, label in zip(outcome_columns, outcome_display):
        ax.plot(
            x_labels, table[col].values,
            marker="o", linewidth=2, markersize=6, label=str(label)
        )

    # --------------------------------------------------
    # Axis labels and title
    # --------------------------------------------------
    ax.set_xlabel(
        xlabel if xlabel is not None else predictor,
        fontsize=14, labelpad=8
    )

    if normalize and percent:
        y_label = "Percentage"
    elif normalize:
        y_label = "Proportion"
    else:
        y_label = "Count"
    
    ax.set_ylabel(y_label, fontsize=14, labelpad=8)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

    # --------------------------------------------------
    # Axis styling
    # --------------------------------------------------
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)

    ax.legend(
        title=legend_title if legend_title is not None else outcome,
        frameon=False, fontsize=11, title_fontsize=11, loc="best"
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Save the figure if a path is provided
    # --------------------------------------------------
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format="png")
        print(f"Figure saved to: {save_path}")

    return table, fig, ax


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
    distribution="logit",
    method="bfgs",
    scale_predictor=None,
    min_group_n=30
):
    """
    Fit separate ordinal regression models within each stratum of a
    grouping variable to assess effect modification (interaction).
 
    Use this to answer: "Is the association between green exposure and
    mental health stronger or weaker for people with low SES?"
 
    Each stratum receives its own model. Comparing ORs across strata
    reveals whether the exposure–outcome association is more pronounced
    in specific subgroups.
 
    For a formal statistical test of interaction, use fit_interaction().
 
    Parameters
    ----------
    data : pd.DataFrame
 
    outcome : str
        Ordinal mental-health outcome (e.g. "CES-D Ordinal").
 
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
 
    distribution : str, default="logit"
 
    method : str, default="bfgs"
 
    scale_predictor : float, optional
        Passed to fit_model() for continuous predictors.

    min_group_n : int, default 30
        Minimum sample size required within a stratum to fit a model.
        Strata with fewer observations are skipped with a message
        rather than raising an error.
 
    Returns
    -------
    results : dict
        Keyed by stratum value. Each entry:
            - "result"    : OrderedResults
            - "metadata"  : dict  (includes stratum info)
            - "or_table"  : pd.DataFrame from extract_results()
            - "n"         : int
 
    summary : pd.DataFrame
        OR tables from all strata stacked, with a "stratum" column added.
        Ready for comparison or plotting.
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
 
    results  = {}
    or_tables = []
 
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
                distribution       = distribution,
                method             = method,
                scale_predictor    = scale_predictor
            )
 
            # Add stratum info to metadata
            metadata["stratum_var"]   = stratify_by
            metadata["stratum_value"] = stratum
 
            or_table = extract_results(result, metadata)
            or_table.insert(0, "stratum", stratum)
 
            results[stratum] = {
                "result":   result,
                "metadata": metadata,
                "or_table": or_table,
                "n":        metadata["n"]
            }
 
            or_tables.append(or_table)
 
        except Exception as e:
            print(f"    [error] stratum {stratum}: {e}")
 
    summary = (
        pd.concat(or_tables, axis=0, ignore_index=True)
        if or_tables else pd.DataFrame()
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
    predictor_type="categorical",
    reference_category=None,
    moderator_type="categorical",
    moderator_reference=None,
    covariates=None,
    distribution="logit",
    method="bfgs",
    scale_predictor=None,
    label_mapping=None
):
    """
    Fit an ordinal regression model with a predictor × moderator
    interaction term to formally test effect modification.
 
    Use this to answer: "Does SES statistically modify the association
    between green exposure and mental health?"
 
    A significant interaction p-value means the exposure–outcome
    association differs across moderator levels — i.e. effect
    modification is present. Pair this with fit_stratified() to
    visualize the direction and magnitude in each stratum.
 
    Parameters
    ----------
    data : pd.DataFrame
 
    outcome : str
        Ordinal mental-health outcome (e.g. "CES-D Ordinal").
 
    predictor : str
        Primary exposure (e.g. "Green Space Quartile (100m Buffer)").
 
    moderator : str
        Effect modifier to test (e.g. "Socioeconomic Status (Tiers)").
 
    predictor_type : str, default="categorical"
        "categorical" or "continuous".
 
    reference_category : str, optional
        Reference level for a categorical predictor.
 
    moderator_type : str, default="categorical"
        "categorical" or "continuous".
        For ordinal moderators like SES (0/1/2), "categorical" uses
        dummy coding; "continuous" assumes a linear moderating effect.
 
    moderator_reference : str/int, optional
        Reference level for a categorical moderator.
        Example: 0 (low SES as reference).
 
    covariates : list[str], optional
        Additional adjustment variables (not interacted).
 
    distribution : str, default="logit"
 
    method : str, default="bfgs"
 
    scale_predictor : float, optional
        Scale factor for continuous predictor (e.g. 0.1 for NDVI).
 
    Returns
    -------
    result : OrderedResults
        Fitted model with interaction term.
 
    metadata : dict
        Standard metadata plus:
            - "moderator"           : str
            - "moderator_type"      : str
            - "moderator_reference" : str or None
            - "formula"             : str — full patsy formula used
 
    interaction_table : pd.DataFrame
        Coefficients, ORs, CIs, and p-values for the interaction terms
        only — the key output for assessing effect modification.

        Columns:
            outcome, predictor, predictor_type, moderator, scale_predictor,
            reference_level, level, moderator_level,
            beta, OR, CI_lower, CI_upper, OR_95CI,
            p_value, p_value_fmt, pct_change_odds, covariates, significant

        `level` and `moderator_level` together identify each row: with
        a categorical moderator that has more than two levels (e.g.
        SES tiers 1 and 2 both vs reference 0), `level` alone repeats
        across rows — `moderator_level` distinguishes them.
 
    Notes
    -----
    Interpretation of interaction terms:
        Categorical predictor × categorical moderator:
            Each interaction coefficient is the *additional* log-OR
            for that predictor level in that moderator group, relative
            to the reference group.
 
        Continuous predictor × categorical moderator:
            The interaction coefficient is the *additional* slope
            per unit of predictor for that moderator group.
 
    A significant interaction p-value (< 0.05) supports effect
    modification. Use fit_stratified() alongside this to inspect
    stratum-specific ORs.
    """
 
    covariates = list(covariates) if covariates else []
 
    # Exclude moderator from covariates — it enters via interaction
    covariates = [c for c in covariates if c != moderator]
 
    model_cols = [outcome, predictor, moderator] + covariates
    subset     = data[model_cols].dropna().copy()
 
    # --------------------------------------------------
    # Outcome
    # --------------------------------------------------
    subset[outcome] = pd.Categorical(subset[outcome], ordered=True)
    outcome_levels  = list(subset[outcome].cat.categories)
 
    if len(outcome_levels) < 3:
        raise ValueError(
            "Ordinal logistic regression requires at least three outcome levels."
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
    # Fit Model
    # --------------------------------------------------
    model  = OrderedModel.from_formula(formula, data=working_df, distr=distribution)
    result = model.fit(method=method, disp=False)
 
    # --------------------------------------------------
    # Extract interaction terms only
    # --------------------------------------------------
    params   = result.params
    conf_int = result.conf_int()
    pvals    = result.pvalues
 
    interaction_mask = params.index.str.contains("predictor_x.*moderator_z|moderator_z.*predictor_x", regex=True)
 
    if not interaction_mask.any():
        # Fallback: any term containing ":"
        interaction_mask = params.index.str.contains(":", regex=False)
 
    betas    = params.loc[interaction_mask]
    lower_ci = conf_int.loc[interaction_mask, 0]
    upper_ci = conf_int.loc[interaction_mask, 1]
    p_values = pvals.loc[interaction_mask]
 
    or_vals   = np.exp(betas.values)
    ci_lo     = np.exp(lower_ci.values)
    ci_hi     = np.exp(upper_ci.values)

    interaction_table = pd.DataFrame({
        "term":            betas.index,
        "beta":            betas.values.round(3),
        "OR":              or_vals.round(3),
        "CI_lower":        ci_lo.round(3),
        "CI_upper":        ci_hi.round(3),
        "OR_95CI":         [
            f"{b:.3f} ({l:.3f}, {u:.3f})"
            for b, l, u in zip(or_vals, ci_lo, ci_hi)
        ],
        "p_value":         p_values.values,
        "p_value_fmt":     [
            "<0.001" if p < 0.001 else f"{p:.3f}" for p in p_values.values
        ],
        "pct_change_odds": np.where(
            or_vals < 1,
            (1 - or_vals) * 100,
            (or_vals - 1) * 100
        ).round(1),
        "significant":     p_values.values < 0.05
    })

    # --------------------------------------------------
    # Clean coefficient labels and set reference_level
    # Categorical: C(predictor_x, Treatment(...))[T.2.0] → Q2
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
        "covariates":          covariates,
        "outcome_levels":      outcome_levels,
        "distribution":        distribution,
        "scale_predictor":     scale_predictor,
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
        "OR",
        "CI_lower",
        "CI_upper",
        "OR_95CI",
        "p_value",
        "p_value_fmt",
        "pct_change_odds",
        "covariates",
        "significant",
    ]]

    n_sig = interaction_table["significant"].sum()
    print(
        f"  Interaction fitted | n = {len(working_df)} | "
        f"{n_sig}/{len(interaction_table)} interaction terms p < 0.05"
    )

    return result, metadata, interaction_table

