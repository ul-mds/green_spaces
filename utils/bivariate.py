"""
bivariate.py

Utility functions for bivariate exploratory data analysis:
pairwise association tests and visualizations for numeric, categorical,
and ordinal variable combinations.

Location: utils/bivariate.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import utils.ordinal_regression as ordinal

__all__ = [
    "analyze_numeric_numeric",
    "plot_numeric_numeric",
    "analyze_numeric_categorical",
    "plot_numeric_categorical",
    "analyze_categorical_categorical",
    "plot_categorical_categorical",
    "analyze_ordinal_outcome",
    "analyze_ordinal_ordinal",
    "plot_ordinal_ordinal",
    "bivar_analysis",
    "group_differences",
    "pairwise_stats",
]


############################################################
## Numeric vs Numeric – Analysis
############################################################
def analyze_numeric_numeric(
    data,
    x,
    y,
    method="spearman"
):
    """
    Analyze the association between two numeric variables.

    Parameters
    ----------
    data : pd.DataFrame

    x, y : str
        Column names.

    method : {"pearson", "spearman"}, default="spearman"
        Pearson for normally distributed data; Spearman otherwise.

    Returns
    -------
    dict
        - x, y, method, n
        - correlation : float — Pearson r or Spearman rho
        - p_value     : float
        - effect_size : float — r² (Pearson) or rho (Spearman; reported as-is)
        - effect_label: str
    """
    subset = data[[x, y]].dropna()

    if method == "pearson":
        corr, p = stats.pearsonr(subset[x], subset[y])
        effect_size  = corr ** 2
        effect_label = "r² (variance explained)"

    elif method == "spearman":
        corr, p = stats.spearmanr(subset[x], subset[y])
        effect_size  = corr
        effect_label = "rho (Spearman)"

    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    return {
        "x":            x,
        "y":            y,
        "method":       method,
        "n":            len(subset),
        "correlation":  corr,
        "p_value":      p,
        "effect_size":  effect_size,
        "effect_label": effect_label
    }


############################################################
## Numeric vs Numeric – Visualization
############################################################
def plot_numeric_numeric(
    data,
    x,
    y,
    method="spearman",
    figsize=(6, 4)
):
    """
    Scatter plot with regression line, annotated with correlation and p-value.
    """
    subset = data[[x, y]].dropna()

    if method == "pearson":
        corr, p = stats.pearsonr(subset[x], subset[y])
        label = "r"
    else:
        corr, p = stats.spearmanr(subset[x], subset[y])
        label = "ρ"

    p_text = "<0.001" if p < 0.001 else f"{p:.3f}"
    annotation = f"{label} = {corr:.3f}, p = {p_text} (n = {len(subset)})"

    fig, ax = plt.subplots(figsize=figsize)

    sns.regplot(data=data, x=x, y=y, ax=ax)

    ax.set_title(f"{x} vs {y}")
    ax.annotate(
        annotation,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

    plt.tight_layout()

    return fig


############################################################
## Numeric vs Categorical – Analysis
############################################################
def analyze_numeric_categorical(
    data,
    numeric_var,
    categorical_var,
    nonparametric=True
):
    """
    Compare a numeric outcome across groups.

    Kruskal-Wallis (nonparametric) or one-way ANOVA (parametric),
    plus per-group descriptives and effect size.

    Parameters
    ----------
    nonparametric : bool, default=True
        If True, use Kruskal-Wallis + eta²_H effect size.
        If False, use one-way ANOVA + eta² effect size.

    Returns
    -------
    dict
        - method, statistic, p_value, n
        - effect_size  : float
        - effect_label : str
        - group_stats  : pd.DataFrame — median, IQR, mean, SD, n per group
    """
    subset = data[[numeric_var, categorical_var]].dropna()

    grouped = subset.groupby(categorical_var)[numeric_var]

    group_stats = grouped.agg(
        n     = "count",
        mean  = "mean",
        sd    = "std",
        median = "median",
        q25   = lambda s: s.quantile(0.25),
        q75   = lambda s: s.quantile(0.75)
    ).reset_index()

    group_stats["IQR"] = group_stats["q75"] - group_stats["q25"]

    groups = [g.values for _, g in grouped]
    n_total = len(subset)
    k = len(groups)

    if nonparametric:
        stat, p = stats.kruskal(*groups)
        method = "Kruskal-Wallis"
        # eta²_H = H / (n - 1)
        effect_size  = stat / (n_total - 1)
        effect_label = "eta²_H (Kruskal-Wallis effect size)"

    else:
        stat, p = stats.f_oneway(*groups)
        method = "ANOVA"
        # eta² = (F * df_between) / (F * df_between + df_within)
        df_between = k - 1
        df_within  = n_total - k
        effect_size  = (stat * df_between) / (stat * df_between + df_within)
        effect_label = "eta² (ANOVA effect size)"

    return {
        "method":       method,
        "statistic":    stat,
        "p_value":      p,
        "n":            n_total,
        "effect_size":  effect_size,
        "effect_label": effect_label,
        "group_stats":  group_stats
    }


############################################################
## Numeric vs Categorical – Visualization
############################################################
def plot_numeric_categorical(
    data,
    numeric_var,
    categorical_var,
    figsize=(6, 4)
):
    """
    Boxplots with strip overlay by category.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=data, x=categorical_var, y=numeric_var, ax=ax
    )

    sns.stripplot(
        data=data, x=categorical_var, y=numeric_var,
        color="black", alpha=0.3, ax=ax
    )

    ax.set_title(f"{numeric_var} by {categorical_var}")

    plt.tight_layout()

    return fig


############################################################
## Categorical vs Categorical – Analysis
############################################################
def analyze_categorical_categorical(
    data,
    var1,
    var2,
    alpha=0.05
):
    """
    Chi-square test (or Fisher's exact for 2×2 tables with low expected counts)
    and Cramér's V effect size.

    Parameters
    ----------
    alpha : float, default=0.05
        Threshold for flagging small expected counts.

    Returns
    -------
    dict
        - table      : pd.DataFrame — observed counts
        - test       : str          — "chi2" or "fisher"
        - statistic  : float
        - p_value    : float
        - dof        : int or None
        - cramers_v  : float
        - n          : int
        - small_expected : bool — True if any expected count < 5
    """
    subset = data[[var1, var2]].dropna()
    table  = pd.crosstab(subset[var1], subset[var2])

    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)

    small_expected = bool((expected < 5).any())

    n         = table.values.sum()
    min_dim   = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

    # For 2x2 tables with small expected counts, prefer Fisher's exact
    if table.shape == (2, 2) and small_expected:
        fisher_or, p_fisher = stats.fisher_exact(table)
        return {
            "table":          table,
            "test":           "fisher",
            "statistic":      fisher_or,
            "p_value":        p_fisher,
            "dof":            None,
            "cramers_v":      cramers_v,
            "n":              n,
            "small_expected": small_expected
        }

    return {
        "table":          table,
        "test":           "chi2",
        "statistic":      chi2,
        "p_value":        p_chi2,
        "dof":            dof,
        "cramers_v":      cramers_v,
        "n":              n,
        "small_expected": small_expected
    }


############################################################
## Categorical vs Categorical – Visualization
############################################################
def plot_categorical_categorical(
    data,
    var1,
    var2,
    normalize=False,
    figsize=(6, 4)
):
    """
    Heatmap of a contingency table.
    """
    table = pd.crosstab(data[var1], data[var2])

    if normalize:
        table = table.div(table.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        table, annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues", ax=ax
    )

    ax.set_title(f"{var1} vs {var2}")

    return fig


############################################################
## Ordinal Outcome vs Ordered Predictor – Comprehensive
############################################################
def analyze_ordinal_outcome(
    data,
    outcome,
    predictor,
    outcome_order=None,
    predictor_order=None,
    fit_ordinal=False,
    outcome_direction="higher_is_worse",
    interpretation_label=None
):
    """
    Analyze the association between an ordinal outcome and an ordered
    exposure variable.

    Provides:
        1) Descriptive profile table (row-normalized cross-tab)
        2) Rank-based association: Spearman's rho, Kendall's tau
        3) Chi-square test and Cramér's V
        4) Q1 vs Q4 extreme-group comparison:
             Fisher OR and relative risk (2×2 collapse if needed)
        5) Optional proportional-odds ordinal regression

    Parameters
    ----------
    data : pd.DataFrame

    outcome : str
        Ordinal outcome variable.

    predictor : str
        Ordered exposure variable (e.g. greenspace quartiles).

    outcome_order : list, optional
        Explicit order of outcome categories, e.g. [0, 1, 2].

    predictor_order : list, optional
        Explicit order of predictor categories, e.g. [1, 2, 3, 4].

    fit_ordinal : bool, default=False
        If True, fit a proportional-odds ordinal logistic regression.

    outcome_direction : {"higher_is_worse", "higher_is_better"},
        default="higher_is_worse"

    interpretation_label : str, optional
        Custom label for OR interpretation column.
        If None, auto-generated from outcome_direction and outcome name.

    Returns
    -------
    dict
        - "summary_df"         : pd.DataFrame  — transposed one-row summary
        - "profile_table"      : pd.DataFrame  — row-normalized cross-tab
        - "count_table"        : pd.DataFrame  — raw count cross-tab
        - "q1_q4_table"        : pd.DataFrame  — first vs last predictor level
        - "q1_q4_binary_table" : pd.DataFrame  — collapsed 2×2 for OR/RR
        - "or_table"           : pd.DataFrame or None
        - "model_result"       : OrderedResults or None
        - "reference_category" : str or None
    """

    # --------------------------------------------------
    # 1. Prepare subset
    # --------------------------------------------------
    subset = data[[outcome, predictor]].dropna().copy()

    if subset.empty:
        raise ValueError(
            f"No complete cases for outcome='{outcome}', predictor='{predictor}'."
        )

    if outcome_order is not None:
        subset[outcome] = pd.Categorical(
            subset[outcome], categories=outcome_order, ordered=True
        )

    if predictor_order is not None:
        subset[predictor] = pd.Categorical(
            subset[predictor], categories=predictor_order, ordered=True
        )

    # --------------------------------------------------
    # 2. Rank-based association
    # --------------------------------------------------
    x_num = subset[predictor].cat.codes if hasattr(subset[predictor], "cat") else subset[predictor]
    y_num = subset[outcome].cat.codes   if hasattr(subset[outcome],   "cat") else subset[outcome]

    spearman_rho, spearman_p = stats.spearmanr(x_num, y_num)
    kendall_tau,  kendall_p  = stats.kendalltau(x_num, y_num)

    # --------------------------------------------------
    # 3. Cross-tabulations
    # --------------------------------------------------
    count_table   = pd.crosstab(subset[predictor], subset[outcome])
    profile_table = pd.crosstab(subset[predictor], subset[outcome], normalize="index")

    if predictor_order is not None:
        count_table   = count_table.reindex(predictor_order)
        profile_table = profile_table.reindex(predictor_order)

    if outcome_order is not None:
        count_table   = count_table.reindex(columns=outcome_order)
        profile_table = profile_table.reindex(columns=outcome_order)

    # --------------------------------------------------
    # 4. Chi-square and Cramér's V
    # --------------------------------------------------
    chi2, chi2_p, dof, _ = stats.chi2_contingency(count_table)

    n         = count_table.to_numpy().sum()
    min_dim   = min(count_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

    # --------------------------------------------------
    # 5. Q1 vs Q4 extreme-group comparison
    # --------------------------------------------------
    q1_q4_table = count_table.iloc[[0, -1]].copy()

    fisher_or = fisher_p = np.nan
    risk_q1 = risk_q4 = relative_risk = np.nan
    relative_risk_reduction = odds_ratio_reduction = np.nan

    # Collapse to 2×2 if outcome has >2 levels
    # event = highest/worst category; non-event = all others
    if q1_q4_table.shape[1] == 2:
        sub_tab = q1_q4_table.copy()
    else:
        worst_col = q1_q4_table.columns[-1]
        sub_tab = pd.DataFrame({
            "non_event": q1_q4_table.drop(columns=worst_col).sum(axis=1),
            "event":     q1_q4_table[worst_col]
        }, index=q1_q4_table.index)

    if sub_tab.shape == (2, 2):
        a, b = sub_tab.iloc[0, 0], sub_tab.iloc[0, 1]
        c, d = sub_tab.iloc[1, 0], sub_tab.iloc[1, 1]

        fisher_or, fisher_p = stats.fisher_exact(sub_tab)

        risk_q1 = b / (a + b) if (a + b) > 0 else np.nan
        risk_q4 = d / (c + d) if (c + d) > 0 else np.nan

        relative_risk = (
            risk_q4 / risk_q1
            if pd.notna(risk_q1) and risk_q1 > 0 else np.nan
        )
        relative_risk_reduction = 1 - relative_risk if pd.notna(relative_risk) else np.nan
        odds_ratio_reduction    = 1 - fisher_or     if pd.notna(fisher_or)     else np.nan

    # --------------------------------------------------
    # 6. Optional ordinal regression
    # --------------------------------------------------
    model_result       = None
    reference_category = None
    or_table           = None

    if fit_ordinal:
        model_result, metadata = ordinal.fit_model(
            data=subset,
            outcome=outcome,
            predictor=predictor
        )

        or_table = ordinal.extract_results(
            model_result,
            metadata,
            outcome_direction=outcome_direction,
            interpretation_label=interpretation_label
        )

        reference_category = metadata["reference_category"]

    # --------------------------------------------------
    # 7. Summary
    # --------------------------------------------------
    summary_df = pd.DataFrame([{
        "outcome":   outcome,
        "predictor": predictor,
        "n":         n,

        "spearman_rho": round(spearman_rho, 4),
        "spearman_p":   round(spearman_p,   4),
        "kendall_tau":  round(kendall_tau,  4),
        "kendall_p":    round(kendall_p,    4),

        "chi2":      round(chi2,      4),
        "chi2_p":    round(chi2_p,    4),
        "chi2_dof":  dof,
        "cramers_v": round(cramers_v, 4),

        "q1_q4_odds_ratio":        round(fisher_or,               4) if pd.notna(fisher_or)               else np.nan,
        "q1_q4_fisher_p":          round(fisher_p,                4) if pd.notna(fisher_p)                else np.nan,
        "q1_risk":                 round(risk_q1,                 4) if pd.notna(risk_q1)                 else np.nan,
        "q4_risk":                 round(risk_q4,                 4) if pd.notna(risk_q4)                 else np.nan,
        "q4_vs_q1_relative_risk":  round(relative_risk,           4) if pd.notna(relative_risk)           else np.nan,
        "relative_risk_reduction": round(relative_risk_reduction, 4) if pd.notna(relative_risk_reduction) else np.nan,
        "odds_ratio_reduction":    round(odds_ratio_reduction,    4) if pd.notna(odds_ratio_reduction)    else np.nan
    }])

    return {
        "summary_df":         summary_df.T,
        "profile_table":      profile_table,
        "count_table":        count_table,
        "q1_q4_table":        q1_q4_table,
        "q1_q4_binary_table": sub_tab,
        "or_table":           or_table,
        "model_result":       model_result,
        "reference_category": reference_category
    }


############################################################
## Ordinal vs Ordinal – Lightweight Analysis
############################################################
def analyze_ordinal_ordinal(
    data,
    outcome,
    predictor
):
    """
    Lightweight bivariate analysis for two ordered categorical variables.

    Combines rank-based association and proportional-odds ordinal regression.
    For a full analysis including chi-square, Q1 vs Q4 comparison, and
    descriptive tables, use analyze_ordinal_outcome() with fit_ordinal=True.

    Parameters
    ----------
    data : pd.DataFrame

    outcome : str
        Ordinal outcome (0 = best, higher = worse).

    predictor : str
        Ordered exposure variable.

    Returns
    -------
    dict
        - spearman_rho, spearman_p
        - kendall_tau, kendall_p
        - n
        - ordinal_model : OrderedResults
        - metadata      : dict
        - or_table      : pd.DataFrame
    """
    subset = data[[outcome, predictor]].dropna()

    rho, rho_p = stats.spearmanr(subset[predictor], subset[outcome])
    tau, tau_p = stats.kendalltau(subset[predictor], subset[outcome])

    result, metadata = ordinal.fit_model(subset, outcome, predictor)
    or_table = ordinal.extract_results(result, metadata)

    return {
        "spearman_rho":  rho,
        "spearman_p":    rho_p,
        "kendall_tau":   tau,
        "kendall_p":     tau_p,
        "n":             len(subset),
        "ordinal_model": result,
        "metadata":      metadata,
        "or_table":      or_table
    }

############################################################
## Ordinal vs Ordinal – Visualization
############################################################
def plot_ordinal_ordinal(
    data,
    outcome,
    predictor,
    figsize=(6, 4)
):
    """
    Profile plot: proportion of each outcome category across predictor levels.
    """
    table = pd.crosstab(
        data[predictor], data[outcome], normalize="index"
    )

    fig, ax = plt.subplots(figsize=figsize)

    for col in table.columns:
        ax.plot(table.index, table[col], marker="o", label=str(col))

    ax.set_ylabel("Proportion")
    ax.set_xlabel(predictor)
    ax.set_title(f"{outcome} by {predictor}")
    ax.legend(title=outcome)

    plt.tight_layout()

    return fig

############################################################
## Bivariate Analysis – Dispatcher
############################################################
def bivar_analysis(
    data,
    x_data,
    y_data,
    x_type,
    y_type
):
    """
    Automatically select and run the appropriate bivariate analysis
    and visualization based on variable types.

    Parameters
    ----------
    data : pd.DataFrame

    x_data : str or list[str]
        Column name(s). For numeric vs categorical, x = numeric.
        For ordinal vs ordinal, x is treated as the predictor.
 
    y_data : str or list[str]
        Column name(s). For numeric vs categorical, y = categorical.
        For ordinal vs ordinal, y is treated as the outcome.

    x_type, y_type : str
        Variable type: "numeric", "categorical", or "ordinal".

    Returns
    -------
    dict
        Keyed by (x, y) tuple. Each value:
            - "results" : dict — statistical results
            - "fig"     : matplotlib Figure

    Notes
    -----
    Type routing:
        numeric     × numeric      → Spearman/Pearson correlation
        numeric     × categorical  → Kruskal-Wallis / ANOVA
        categorical × numeric      → Kruskal-Wallis / ANOVA (x/y swapped)
        ordinal     × ordinal      → rank-based + ordinal regression
        ordinal     × numeric      → Kruskal-Wallis (ordinal as grouping)
        numeric     × ordinal      → Kruskal-Wallis
        ordinal     × categorical  → chi-square + Cramér's V
        categorical × ordinal      → chi-square + Cramér's V
        categorical × categorical  → chi-square + Cramér's V
    """
    if isinstance(x_data, str): x_data = [x_data]
    if isinstance(y_data, str): y_data = [y_data]

    results_list = {}
    
    for x in x_data:
        for y in y_data:
            # --------------------------------------------------
            # Select analysis and plot function
            # --------------------------------------------------
            if x_type == "numeric" and y_type == "numeric":
                results = analyze_numeric_numeric(data, x, y)
                fig = plot_numeric_numeric(data, x, y)
        
            elif x_type == "numeric" and y_type == "categorical":
                results = analyze_numeric_categorical(data, x, y)
                fig = plot_numeric_categorical(data, x, y)
        
            elif x_type == "categorical" and y_type == "numeric":
                results = analyze_numeric_categorical(data, y, x)
                fig = plot_numeric_categorical(data, y, x)
        
            elif x_type == "ordinal" and y_type == "ordinal":
                results = analyze_ordinal_ordinal(data, y, x)
                fig = plot_ordinal_ordinal(data, y, x)
        
            elif x_type == "ordinal" and y_type == "numeric":
                results = analyze_numeric_categorical(data, y, x)
                fig = plot_numeric_categorical(data, y, x)
        
            elif x_type == "numeric" and y_type == "ordinal":
                results = analyze_numeric_categorical(data, x, y)
                fig = plot_numeric_categorical(data, x, y)
        
            elif x_type == "ordinal" and y_type == "categorical":
                results = analyze_categorical_categorical(data, x, y)
                fig = plot_categorical_categorical(data, x, y)
        
            elif x_type == "categorical" and y_type == "ordinal":
                results = analyze_categorical_categorical(data, x, y)
                fig = plot_categorical_categorical(data, x, y)
        
            elif x_type == "categorical" and y_type == "categorical":
                results = analyze_categorical_categorical(data, x, y)
                fig = plot_categorical_categorical(data, x, y)
            else:
                raise ValueError(
                    f"Unsupported type combination: x_type='{x_type}', "
                    f"y_type='{y_type}'. Expected 'numeric', 'categorical', "
                    f"or 'ordinal'."
                )
            
            # --------------------------------------------------
            # Per-pair output: header → results → figure
            # --------------------------------------------------
            print(f"\n{'='*60}")
            print(f"  {x}  ×  {y}")
            print(f"{'='*60}")
 
            for key, val in results.items():
                if isinstance(val, pd.DataFrame):
                    print(f"\n{key}:")
                    print(val.to_string())
                elif val is not None:
                    print(f"  {key}: {val}")
 
            plt.show()
            
            results_list[(x, y)] = {"results": results, "fig": fig}

    return results_list


############################################################
## Pairwise Grid Runner
############################################################
def run_pairwise(
    data,
    outcome_cols,
    predictor_cols,
    analysis_fn,
    **kwargs
):
    """
    Run a bivariate analysis function for every outcome × predictor
    combination from two column lists.
 
    Replaces infinite values with NaN and drops incomplete rows for
    each pair independently, so one problematic column does not
    affect unrelated pairs.
 
    Parameters
    ----------
    data : pd.DataFrame
 
    outcome_cols : list[str]
        Outcome variables (rows of the grid).
        Example: cols.mental_numeric or cols.mental_ordinal
 
    predictor_cols : list[str]
        Predictor/exposure variables (columns of the grid).
        Example: cols.ndvi or cols.greenspace_quartiles
 
    analysis_fn : callable
        Any bivariate analysis function with signature:
            analysis_fn(data, outcome, predictor, **kwargs)
        Examples:
            bivariate.analyze_numeric_categorical
            bivariate.analyze_ordinal_outcome
            bivariate.bivar_analysis   (requires x_type/y_type in kwargs)
 
    **kwargs
        Additional keyword arguments forwarded to analysis_fn on
        every call (e.g. outcome_direction, fit_ordinal, nonparametric).
 
    Returns
    -------
    dict
        Nested dict keyed by (outcome, predictor) tuple.
        Each value is the return value of analysis_fn.
 
        Example access:
            results[("CES-D Ordinal", "NDVI (100m Buffer)")]
 
    Example
    -------
    results = bivariate.run_pairwise(
        df,
        outcome_cols   = cols.mental_ordinal,
        predictor_cols = cols.greenspace_quartiles,
        analysis_fn    = bivariate.analyze_ordinal_outcome,
        outcome_order  = [0, 1, 2],
        fit_ordinal    = True,
        outcome_direction = "higher_is_worse"
    )
    """
    results = {}
    n_pairs = len(outcome_cols) * len(predictor_cols)
    done    = 0
 
    for outcome in outcome_cols:
        for predictor in predictor_cols:
 
            # Clean this pair independently
            pair = (
                data[[outcome, predictor]]
                .where(lambda df: ~df.isin([np.inf, -np.inf]))
                .dropna()
            )
 
            if pair.empty:
                print(f"  [skip] {outcome} × {predictor} — no complete cases")
                done += 1
                continue
 
            done += 1
            print(f"  [{done}/{n_pairs}] {outcome} × {predictor}")
 
            results[(outcome, predictor)] = analysis_fn(
                pair, outcome, predictor, **kwargs
            )
 
    return results
 
############################################################
## Group Differences – Detailed Analysis
############################################################
def group_differences(
    data,
    outcome,
    predictor,
    min_group_n=3,
    alpha=0.05
):
    """
    Detailed distribution analysis of a numeric outcome across
    categories of a categorical predictor.
 
    Runs per-group descriptives and normality tests, then selects and
    runs the appropriate statistical test via a decision tree:
 
        Any group non-normal?
        ├── Yes → Mann-Whitney U (2 groups) / Kruskal-Wallis (>2)
        └── No  → Levene's test for equal variances
                  ├── Unequal → Welch's t-test (2) / Welch's ANOVA (>2)
                  └── Equal   → Student's t-test (2) / One-way ANOVA (>2)
 
    All tests are computed regardless of the recommendation so results
    are always available for comparison.
 
    Parameters
    ----------
    data : pd.DataFrame
 
    outcome : str
        Numeric outcome variable (e.g. "CES-D Score").
 
    predictor : str
        Categorical grouping variable (e.g. "Sex").
 
    min_group_n : int, default=3
        Minimum observations required per group.
        Groups below this threshold are excluded; if any group is
        excluded the function returns early with only descriptives.
 
    alpha : float, default=0.05
        Significance level for normality and variance tests.
 
    Returns
    -------
    dict
        - outcome         : str
        - predictor       : str
        - n               : int   — total analytic n
        - group_stats     : pd.DataFrame — mean, SD, median, IQR, min, max, n per group
        - normality       : pd.DataFrame — Shapiro-Wilk (D'Agostino-Pearson) per group
        - all_normal      : bool
        - levene          : dict or None — statistic, p_value (None if non-normal)
        - equal_variance  : bool or None
        - recommended_test: str
        - tests           : dict — all computed test results
        - skipped_groups  : list — groups excluded for low n
    """
    # Clean data and group extraction
    df_clean   = data[[outcome, predictor]].dropna()
    groups_all = df_clean[predictor].unique()
    group_data = {g: df_clean.loc[df_clean[predictor] == g, outcome] for g in groups_all}
 
    # --------------------------------------------------
    # Filter by minimum group size check
    # --------------------------------------------------
    valid   = {g: s for g, s in group_data.items() if len(s) >= min_group_n}
    skipped = [g for g, s in group_data.items()    if len(s) <  min_group_n]
 
    if skipped:
        print(
            f"  [warn] {predictor}: groups {skipped} have < {min_group_n} observations "
            f"and are excluded."
        )
 
    analytic_df  = df_clean[df_clean[predictor].isin(valid.keys())]
    
    if len(valid) < 2:
        print(
            f"  [skip] {outcome} × {predictor} — fewer than 2 valid groups after "
            f"minimum-n filter."
        )
        return {
            "outcome": outcome, "predictor": predictor,
            "n": len(analytic_df), "skipped_groups": skipped,
            "group_stats": _group_descriptives(analytic_df, outcome, predictor),
            "normality": None, "all_normal": None,
            "levene": None, "equal_variance": None,
            "recommended_test": "insufficient data",
            "tests": {}
        }
 
    valid_series = list(valid.values())
    k            = len(valid)

    # --------------------------------------------------
    # Per-group descriptives & normality checks
    # --------------------------------------------------
    group_stats = _group_descriptives(analytic_df, outcome, predictor)
    norm_rows = []

    for group, series in valid.items():
        n = len(series)
        if n <= 5000:
            stat, p = stats.shapiro(series)
            test = "Shapiro-Wilk"
        else:
            stat, p = stats.normaltest(series)
            test = "D'Agostino-Pearson"
            print(f"\n[Warning] N = {n} > 5000: Shapiro-Wilk may be inaccurate, "
                  f"D'Agostino-Pearson test used instead")
        norm_rows.append({
            "group":     group,
            "n":         n,
            "test":      test,
            "statistic": round(stat, 4),
            "p_value":   round(p, 4),
            "normal":    p >= alpha
        })
 
    normality    = pd.DataFrame(norm_rows).set_index("group")
    all_normal   = bool(normality["normal"].all())
 
    # --------------------------------------------------
    # Levene's test – homogeneity of variance 
    # (only when all groups are normal)
    # --------------------------------------------------
    levene_result  = None
    equal_variance = None
 
    if all_normal:
        lev_stat, lev_p = stats.levene(*valid_series)
        levene_result  = {"statistic": round(lev_stat, 4), "p_value": round(lev_p, 4)}
        equal_variance = lev_p >= alpha
 
    # --------------------------------------------------
    # Statistical tests
    # --------------------------------------------------
    tests = {}
 
    if k == 2:
        s0, s1 = valid_series
 
        u_stat, p_mw = stats.mannwhitneyu(s0, s1, alternative="two-sided")
        tests["mann_whitney"] = {
            "statistic": round(u_stat, 4),
            "p_value":   round(p_mw, 4)
        }
 
        t_w, p_w = stats.ttest_ind(s0, s1, equal_var=False)
        tests["welch_t"] = {
            "statistic": round(t_w, 4),
            "p_value":   round(p_w, 4)
        }
 
        t_s, p_s = stats.ttest_ind(s0, s1, equal_var=True)
        tests["student_t"] = {
            "statistic": round(t_s, 4),
            "p_value":   round(p_s, 4)
        }
 
        if not all_normal:
            recommended_test = "mann_whitney"
        else:
            recommended_test = "student_t" if equal_variance else "welch_t"
 
    else:  # k > 2
        h_stat, p_kw = stats.kruskal(*valid_series)
        tests["kruskal_wallis"] = {
            "statistic": round(h_stat, 4),
            "p_value":   round(p_kw, 4)
        }
 
        f_stat, p_anova = stats.f_oneway(*valid_series)
        tests["anova"] = {
            "statistic": round(f_stat, 4),
            "p_value":   round(p_anova, 4)
        }
 
        # Welch's ANOVA Pure Implementation
        ns = np.array([len(x) for x in valid_series])
        means = np.array([np.mean(x) for x in valid_series])
        vars_ = np.array([np.var(x, ddof=1) for x in valid_series])
        weights = ns / vars_
        
        w_sum = np.sum(weights)
        mean_w = np.sum(weights * means) / w_sum
        
        num = np.sum(weights * (means - mean_w) ** 2) / (k - 1)
        lambdas = weights / w_sum
        den_term = np.sum((1.0 / (ns - 1)) * ((1.0 - lambdas) ** 2))
        den = 1.0 + (2.0 * (k - 2) / (k ** 2 - 1)) * den_term
        
        f_welch = num / den
        df2 = 1.0 / (3.0 / (k ** 2 - 1) * den_term)
        p_welch = stats.f.sf(f_welch, k - 1, df2)

        tests["welch_anova"] = {
            "statistic": round(f_welch, 4), "p_value": round(p_welch, 4),
            "ddof1": k - 1, "ddof2": round(df2, 4)
        }

        if not all_normal:
            recommended_test = "kruskal_wallis"
        else:
            recommended_test = "anova" if equal_variance else "welch_anova"
 
    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------
    rec = tests.get(recommended_test, {})
    summary_str = (
        f" n={len(analytic_df)} | "
        f"{'Normal' if all_normal else 'Non-normal'} | "
        f"Recommended: {recommended_test} | "
        f"p = {rec.get('p_value', 'n/a')}"
    )
    print(f"  {summary_str}")

    return {
        "outcome":          outcome,
        "predictor":        predictor,
        "summary":          summary_str,
        "n":                len(analytic_df),
        "group_stats":      group_stats,
        "normality":        normality,
        "all_normal":       all_normal,
        "levene":           levene_result,
        "equal_variance":   equal_variance,
        "recommended_test": recommended_test,
        "tests":            tests,
        "skipped_groups":   skipped
    }
 
 
def _group_descriptives(data, outcome, predictor):
    """Per-group descriptive statistics for outcome across predictor groups."""
    desc = data.groupby(predictor, observed=True)[outcome].agg(
        n      = "count",
        mean   = "mean",
        sd     = "std",
        median = "median",
        q25    = lambda s: s.quantile(0.25),
        q75    = lambda s: s.quantile(0.75),
        min    = "min",
        max    = "max"
        )
    desc["IQR"] = desc["q75"] - desc["q25"]
    
    return desc.round(3).reset_index()


############################################################
## Pairwise Group Differences – Batch Runner
############################################################
def pairwise_stats(
    data,
    outcome_cols,
    predictor_cols,
    min_group_n=3,
    alpha=0.05
):
    """
    Run analyze_group_differences() for every outcome × predictor
    combination from two column lists.
 
    Intended for numeric outcomes across categorical predictors,
    e.g. mental health scores by sex or by greenspace quartile.
 
    Replaces infinite values with NaN and drops incomplete rows for
    each pair independently.
 
    Parameters
    ----------
    data : pd.DataFrame
 
    outcome_cols : list[str]
        Numeric outcome variables.
        Example: cols.mental_numeric
 
    predictor_cols : list[str]
        Categorical grouping variables.
        Example: cols.baseline_categorical or cols.greenspace_quartiles
 
    min_group_n : int, default=3
        Forwarded to analyze_group_differences().
 
    alpha : float, default=0.05
        Forwarded to analyze_group_differences().
 
    Returns
    -------
    dict
        Keyed by (outcome, predictor) tuple.
        Each value is the dict returned by analyze_group_differences().
 
    Example
    -------
    results = bivariate.pairwise_stats(
        df,
        outcome_cols   = cols.mental_numeric,
        predictor_cols = cols.baseline_categorical
    )
 
    # Access one result
    results[("CES-D Score", "Sex")]["group_stats"]
    results[("CES-D Score", "Sex")]["tests"]["mann_whitney"]
    results[("CES-D Score", "Sex")]["recommended_test"]
    """
    if isinstance(outcome_cols, str):   outcome_cols   = [outcome_cols]
    if isinstance(predictor_cols, str): predictor_cols = [predictor_cols]
    
    results = {}
    n_pairs = len(outcome_cols) * len(predictor_cols)
    done    = 0
 
    for outcome in outcome_cols:
        for predictor in predictor_cols:
            done += 1

            # Clean infinite strings/values and extract targeted columns safely
            pair = (
                data[[outcome, predictor]]
                .where(lambda df: ~df.isin([np.inf, -np.inf]))
                .dropna()
            )

            if pair.empty:
                print(f"\n[{done}/{n_pairs}] {outcome} × {predictor}")
                print(f"  [skip] — No complete cases found.")
                print("-" * 60)
                continue
 
            print(f"[{done}/{n_pairs}] Processing... {outcome} × {predictor}")
 
            results[(outcome, predictor)] = group_differences(
                pair, outcome, predictor,
                min_group_n=min_group_n,
                alpha=alpha
            )
            print("-" * 60)

    return results
