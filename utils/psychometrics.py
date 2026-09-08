"""
psychometrics.py

Utility functions for psychometric scale handling:
person-mean imputation and internal consistency reliability testing 
for questionnaire data.

Location: utils/psychometrics.py
"""

import warnings
import numpy as np
import pandas as pd

from reliabilipy import reliability_analysis

__all__ = [
    "person_mean_imputation",
    "reliability_test",
]

############################################################
## Person-mean Imputation
############################################################
def person_mean_imputation(
    df: pd.DataFrame,
    subset_columns: list,
    threshold: float = 0.20,
    warn_item_missing: float = 0.05,
):
    """
    Person-mean imputation for psychometric scales.

    Participants with more than `threshold` proportion of missing items are
    excluded. Remaining missing items are replaced by the participant's own
    mean across completed items.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing questionnaire items.

    subset_columns : list
        Item columns belonging to one questionnaire.

    threshold : float, default=0.20
        Maximum proportion of missing items allowed.

        Examples
        --------
        CES-D  (20 items): 1/5 ≈ 0.20  (<=4 missing)
        GAD-7  (7  items): 1/7 ≈ 0.143 (<=1 missing)
        SWLS   (5  items): 1/5 ≈ 0.20  (<=1 missing)
        IDS    (30 items): 1/5 ≈ 0.20  (<=6 missing)
        GDS-15 (15 items): 1/5 ≈ 0.20  (<=3 missing)

    warn_item_missing : float, default=0.05
        Warn if more than this proportion of participants are missing an item.

    Returns
    -------
    pd.DataFrame
        DataFrame after person-mean imputation.
    """

    df = df.copy()

    n_total = df[subset_columns].shape[0]
    complete_count = df[subset_columns].notna().all(axis=1).sum()
    
    # --------------------------------------------------
    # Exclude participants with too many missing items
    # --------------------------------------------------
    missing_prop = df[subset_columns].isna().mean(axis=1)

    n_removed = (missing_prop > threshold).sum()

    print("\nMissing Imputation")
    print("-" * 40)
    
    if n_removed > 0:
        print(
            f"Of {n_total} participants, {n_removed} "
            f"({100*n_removed/n_total:.1f}%) excluded for "
            f"missing >{threshold:.0%} of questionnaire items,\n"
            f"leaving a final sample of {n_total - n_removed} cases "
            f"with {complete_count} complete and "
            f"{n_total - complete_count - n_removed} imputed."
        )
    else:
        print(
            f"Of {n_total} participants, 0 were excluded: "
            f"no missing values."
        )

    df = df.loc[missing_prop <= threshold].copy()

    # --------------------------------------------------
    # Warn about item-level missingness
    # --------------------------------------------------
    warning_block_started = True
    for col in subset_columns:

        pct_missing = df[col].isna().mean()

        if pct_missing > warn_item_missing:
            if warning_block_started:
                print("\n", end="")
                warning_block_started = False
            
            warnings.warn(
                f"\n{col}: {pct_missing:.1%} missing responses."
            )

    # --------------------------------------------------
    # Person-mean imputation
    # --------------------------------------------------
    person_means = df[subset_columns].mean(axis=1)

    df[subset_columns] = df[subset_columns].T.fillna(person_means).T

    return df

############################################################
## Internal Consistency Reliability
############################################################
def reliability_test(
    df: pd.DataFrame,
    items: list[str],
    omega_threshold: float = 0.70,
    alpha_threshold: float = 0.70,
):
    """
    Compute internal consistency reliability for a psychometric scale.

    The function assumes any desired missing-data handling (e.g.,
    person-mean imputation) has already been performed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing questionnaire items.

    items : list[str]
        Item columns belonging to the scale.

    omega_threshold : float, default=0.70
        Reference threshold for McDonald's Omega.

    alpha_threshold : float, default=0.70
        Reference threshold for Cronbach's alpha.

    Returns
    -------
    reliability_analysis
        Fitted reliability analysis object.
    """

    data = df[items].copy()

    # Warn if missing values remain
    if data.isna().any().any():
        n_missing = int(data.isna().sum().sum())
        warnings.warn(
            f"{n_missing} missing values remain in the selected items. "
            "Reliability estimates may be affected."
        )

    # Suppress a benign numerical warning from reliabilipy that may occur
    # during factor analysis when a uniqueness is estimated slightly below
    # zero due to numerical precision.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar power",
            category=RuntimeWarning,
            module="reliabilipy.*",
        )
    
        # Compute reliability
        ra = reliability_analysis(
            raw_dataset=data,
            is_corr_matrix=False,
            impute='mean'      # missing data already handled upstream
        )
    
        ra.fit()

    print("\nInternal consistency")
    print("-" * 40)
    print(f"Number of participants : {len(data):,}")
    print(f"Number of items        : {len(items)}")
    print(f"McDonald's Omega       : {ra.omega_total:.3f}")
    print(f"Cronbach's Alpha       : {ra.alpha_cronbach:.3f}")
    print(f"\n")

    if ra.omega_total < omega_threshold:
        warnings.warn(
            f"Omega ({ra.omega_total:.3f}) is below the recommended "
            f"threshold ({omega_threshold:.2f})."
        )

    if ra.alpha_cronbach < alpha_threshold:
        warnings.warn(
            f"Cronbach's alpha ({ra.alpha_cronbach:.3f}) is below the "
            f"recommended threshold ({alpha_threshold:.2f})."
        )

    return ra
