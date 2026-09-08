"""
univariate.py

Utility functions for univariate exploratory data analysis:
descriptive statistics, missing data, normality testing,
and visualizations for numeric, ordinal, and categorical variables.

Location: utils/univariate.py

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

__all__ = [
    "summarize_missing",
    "check_normality",
    "summarize_normality",
    "summarize_numeric",
    "plot_numeric_distribution",
    "summarize_categorical",
    "plot_categorical_distribution",
    "univar_analysis",
]


############################################################
## Missing Data Summary
############################################################
def summarize_missing(
    data,
    columns=None
):
    """
    Summarize missing values across columns.

    Parameters
    ----------
    data : pd.DataFrame

    columns : list[str], optional
        Columns to inspect. Defaults to all columns.

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - missing_n   : int   — count of missing values
        - missing_pct : float — percentage missing
        - present_n   : int   — count of non-missing values
        Sorted descending by missing_pct.
        Only columns with at least one missing value are returned.
    """
    if columns is None:
        columns = list(data.columns)

    n_total = len(data)

    missing_n   = data[columns].isna().sum()
    missing_pct = (missing_n / n_total * 100).round(2)
    present_n   = n_total - missing_n

    result = pd.DataFrame({
        "missing_n":   missing_n,
        "missing_pct": missing_pct,
        "present_n":   present_n
    })

    result = result[result["missing_n"] > 0].sort_values(
        "missing_pct", ascending=False
    )

    return result


############################################################
## Normality Check
############################################################
def check_normality(
    data,
    column,
    alpha=0.05
):
    """
    Test normality of a single numeric variable.

    Uses Shapiro-Wilk for n <= 5000 (more powerful for small samples),
    and D'Agostino-Pearson for n > 5000.

    Parameters
    ----------
    data : pd.DataFrame

    column : str

    alpha : float, default=0.05
        Significance level for the normality decision.

    Returns
    -------
    dict
        - column    : str
        - test      : str   — test name
        - statistic : float
        - p_value   : float
        - normal    : bool  — True if p >= alpha (fail to reject normality)
        - n         : int
    """
    series = data[column].dropna()
    n = len(series)

    if n < 3:
        raise ValueError(
            f"Column '{column}' has fewer than 3 non-null values; "
            "normality cannot be tested."
        )

    if n <= 5000:
        stat, p = stats.shapiro(series)
        test = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(series)
        test = "D'Agostino-Pearson"

    return {
        "column":    column,
        "n":         n,
        "test":      test,
        "statistic": round(stat, 4),
        "p_value":   round(p, 4),
        "normal":    p >= alpha
    }


############################################################
## Summarize Normality
############################################################
def summarize_normality(
    data,
    columns,
    alpha=0.05
):
    """
    Run normality tests across multiple columns and return a summary table.

    Parameters
    ----------
    data : pd.DataFrame

    columns : list[str]

    alpha : float, default=0.05

    Returns
    -------
    pd.DataFrame
        One row per column with:
        - test, statistic, p_value, normal, n
    """
    records = []

    for col in columns:
        try:
            records.append(check_normality(data, col, alpha=alpha))
        except ValueError as e:
            records.append({
                "column":    col,
                "test":      None,
                "statistic": None,
                "p_value":   None,
                "normal":    None,
                "n":         data[col].notna().sum(),
                "note":      str(e)
            })

    return pd.DataFrame(records).set_index("column")


############################################################
## Numeric Summary
############################################################
def summarize_numeric(
    data,
    columns,
    round_digits=3,
    ci=0.95
):
    """
    Generate descriptive statistics for numeric variables.

    Statistics include:
    - n (non-missing count)
    - missing_n, missing_pct
    - mean, SD
    - CI lower/upper for the mean
    - min, Q1, median, Q3, max
    - IQR
    - mode (first if tied)
    - skewness
    - kurtosis (excess / Fisher definition)

    Parameters
    ----------
    data : pd.DataFrame

    columns : list[str]
        Numeric variables to summarize.

    round_digits : int, default=3
        Decimal places for all numeric output.

    ci : float, default=0.95
        Confidence level for the mean CI.

    Returns
    -------
    pd.DataFrame
        Summary statistics table (one row per column).
    """
    z = stats.norm.ppf((1 + ci) / 2)

    rows = {}

    for col in columns:
        series = data[col].dropna()
        n      = len(series)
        n_miss = data[col].isna().sum()
        mean   = series.mean()
        sd     = series.std()
        se     = sd / np.sqrt(n) if n > 0 else np.nan

        rows[col] = {
            "n":            n,
            "missing_n":    n_miss,
            "missing_pct":  round(n_miss / len(data) * 100, round_digits),
            "mean":         round(mean, round_digits),
            "sd":           round(sd, round_digits),
            f"ci{int(ci*100)}_lower": round(mean - z * se, round_digits),
            f"ci{int(ci*100)}_upper": round(mean + z * se, round_digits),
            "min":          round(series.min(), round_digits),
            "Q1":           round(series.quantile(0.25), round_digits),
            "median":       round(series.median(), round_digits),
            "Q3":           round(series.quantile(0.75), round_digits),
            "max":          round(series.max(), round_digits),
            "IQR":          round(series.quantile(0.75) - series.quantile(0.25), round_digits),
            # First mode only — ties are not uncommon in clinical scales
            "mode":         round(series.mode().iloc[0], round_digits) if not series.mode().empty else np.nan,
            "skewness":     round(series.skew(), round_digits),
            "kurtosis":     round(series.kurt(), round_digits),
        }

    return pd.DataFrame(rows).T.astype({"n": "Int64", "missing_n": "Int64"})


############################################################
## Numeric Visualization
############################################################
def plot_numeric_distribution(
    data,
    column,
    bins=30,
    show_normal_ref=True
):
    """
    Visualize a numeric variable with three panels:

    1. Histogram — with mean (dashed) and median (dotted) reference lines
    2. Density (KDE) — with normality test result annotated
    3. Boxplot

    Parameters
    ----------
    data : pd.DataFrame

    column : str

    bins : int, default=30

    show_normal_ref : bool, default=True
        If True, overlay a normal reference curve on the density panel
        and annotate with the normality test result.
    """
    series = data[column].dropna()

    if series.empty:
        raise ValueError(f"Column '{column}' has no non-null values to plot.")

    mean   = series.mean()
    median = series.median()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # --------------------------------------------------
    # Panel 1: Histogram with mean and median lines
    # --------------------------------------------------
    sns.histplot(series, bins=bins, ax=axes[0])

    axes[0].axvline(
        mean,   color="tab:red",   linestyle="--", linewidth=1.5, label=f"Mean {mean:.2f}"
    )
    axes[0].axvline(
        median, color="tab:blue",  linestyle=":",  linewidth=1.5, label=f"Median {median:.2f}"
    )
    axes[0].legend(fontsize=9)
    axes[0].set_title("Histogram")

    # --------------------------------------------------
    # Panel 2: Density with optional normality annotation
    # --------------------------------------------------
    sns.kdeplot(series, fill=True, ax=axes[1])

    if show_normal_ref:
        try:
            norm_result = check_normality(data, column)
            p_fmt  = "<0.001" if norm_result["p_value"] < 0.001 else f'{norm_result["p_value"]:.3f}'
            label  = (
                f'{norm_result["test"]}\n'
                f'p = {p_fmt}\n'
                f'{"Normal" if norm_result["normal"] else "Non-normal"}'
            )
            axes[1].annotate(
                label,
                xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        except ValueError:
            pass

    axes[1].set_title("Density")

    # --------------------------------------------------
    # Panel 3: Boxplot
    # --------------------------------------------------
    sns.boxplot(y=series, ax=axes[2])
    axes[2].set_title("Boxplot")

    fig.suptitle(column, fontweight="bold")
    plt.tight_layout()

    return fig


############################################################
## Categorical / Ordinal Summary
############################################################
def summarize_categorical(
    data,
    columns,
    normalize=True,
    cumulative=False
):
    """
    Generate frequency tables for categorical or ordinal variables.

    Parameters
    ----------
    data : pd.DataFrame

    columns : list[str]

    normalize : bool, default=True
        Include proportions alongside counts.

    cumulative : bool, default=False
        Include cumulative proportion column.
        Useful for ordinal variables to show how responses accumulate
        across ordered categories.

    Returns
    -------
    dict
        Frequency tables keyed by column name.
        Each table contains:
        - count
        - proportion     (if normalize=True)
        - cumulative_pct (if cumulative=True)

        Missing values (NaN) are included as a separate row.
    """
    results = {}

    for col in columns:

        counts = data[col].value_counts(dropna=False)

        result = pd.DataFrame({"count": counts.astype("Int64")})

        if normalize:
            result["proportion"] = (counts / counts.sum()).round(4)

        if cumulative and normalize:
            # Cumulative % excluding NaN row — only meaningful for ordered categories
            valid_mask    = result.index.notna()
            valid_props   = result.loc[valid_mask, "proportion"]
            result.loc[valid_mask, "cumulative_pct"] = (
                valid_props.cumsum() * 100
            ).round(2)

        results[col] = result

    return results


############################################################
## Categorical / Ordinal Visualization
############################################################
def plot_categorical_distribution(
    data,
    column,
    show_percentages=True,
    sort=False
):
    """
    Bar chart of category frequencies.

    Parameters
    ----------
    data : pd.DataFrame

    column : str

    show_percentages : bool, default=True
        Annotate bars with percentage labels.

    sort : bool, default=False
        If True, sort bars by descending frequency.
        If False, preserve the natural category order.
        For ordinal variables, leave as False to keep meaningful order.
    """
    counts = data[column].value_counts(sort=sort, dropna=False)

    if counts.empty:
        raise ValueError(f"Column '{column}' has no values to plot.")

    total = counts.sum()

    fig, ax = plt.subplots(figsize=(6, 3))

    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=sns.color_palette("Blues_d", len(counts))
    )

    ax.set_title(column, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel(column)

    if show_percentages:
        for bar in bars:
            pct = bar.get_height() / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9
            )

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig


############################################################
## Univariate Analysis – Dispatcher
############################################################
def univar_analysis(
    data,
    columns,
    variable_type="numeric"
):
    """
    Run univariate analysis for a list of columns of the same type.

    Prints summary tables and displays distribution plots for each variable.

    Parameters
    ----------
    data : pd.DataFrame

    columns : list[str]

    variable_type : str, default="numeric"
        One of:
        - "numeric"     : descriptive stats + normality + histogram/density/boxplot
        - "categorical" : frequency table + bar chart
        - "ordinal"     : frequency table with cumulative % + bar chart
                          (preserves category order; does not sort bars)

    Returns
    -------
    dict
        - "summary"   : pd.DataFrame (numeric) or dict of DataFrames (categorical/ordinal)
        - "normality" : pd.DataFrame or None — normality results (numeric only)
    """

    if variable_type == "numeric":

        # Missing overview
        missing = summarize_missing(data, columns)
        if not missing.empty:
            print("── Missing Values ──────────────────────────")
            print(missing.to_string())
            print()

        # Descriptive statistics
        summary = summarize_numeric(data, columns)
        print("── Descriptive Statistics ───────────────────")
        print(summary.to_string())
        print()

        # Normality
        normality = summarize_normality(data, columns)
        print("── Normality Tests ──────────────────────────")
        print(normality.to_string())
        print()

        # Plots
        for col in columns:
            plot_numeric_distribution(data, col)
            plt.show()

        return {"summary": summary, "normality": normality}

    elif variable_type in ("categorical", "ordinal"):

        is_ordinal = variable_type == "ordinal"

        summary = summarize_categorical(
            data,
            columns,
            normalize=True,
            cumulative=is_ordinal
        )

        for col, table in summary.items():
            print(f"── {col} {'─' * max(0, 44 - len(col))}")
            print(table.to_string())
            print()

            plot_categorical_distribution(
                data,
                col,
                sort=False  # preserve order for both categorical and ordinal
            )
            plt.show()

        return {"summary": summary, "normality": None}

    else:
        raise ValueError(
            "variable_type must be 'numeric', 'categorical', or 'ordinal'."
        )

