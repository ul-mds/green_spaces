"""
transform.py

Generic utility functions for transforming variables in a pandas
DataFrame: categorizing continuous variables into bins, binarizing
variables at a threshold, and renaming raw/coded columns to
standardized, readable names.

Location: utils/transform.py
"""

import numpy as np
import pandas as pd

__all__ = [
    "categorize",
    "binarize",
    "rename_columns",
]


############################################################
## Categorize a Variable
############################################################
def categorize(
    df,
    column,
    bins=None,
    labels=None,
    category_column=None,
    include_lowest=True,
    right=True,
    return_df=False,
):
    """
    Categorize a continuous variable into ordered categories.

    The variable is divided into intervals defined by the supplied
    bin boundaries using pandas.cut.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    column : str
        Variable to categorize.

    bins : sequence
        Bin boundaries.

        Example
        -------
        bins = [0, 5, 10, 20]

        produces the intervals

            (0,5], (5,10], (10,20]

        when right=True.

    labels : sequence, optional
        Labels assigned to each interval.

        If None, integer labels
        0, 1, ..., n_intervals-1 are used.

    category_column : str, optional
        Name of the output variable.
        Defaults to "<column>_category".

    include_lowest : bool, default=True
        Include the minimum value in the first interval.

    right : bool, default=True
        Defines interval closure.

        True
            (a, b]

        False
            [a, b)

    return_df: bool, default=False
        Return a copy of the input DataFrame with the new variable
        appended, if True. Otherwise, return only the new variable as
        a pandas.Series (default).

    Returns
    -------
    pandas.Series or pandas.DataFrame
    If return_df=False (default), returns the newly created categorized
    variable as a pandas.Series.

    If return_df=True, returns a copy of the input DataFrame with the
    categorized variable added as a new category column.
    """
    if bins is None:
        raise ValueError("'bins' must be provided.")

    if isinstance(bins, str) or not hasattr(bins, "__len__"):
        raise TypeError(
            "'bins' must be a sequence (e.g. list, tuple, or array) "
            "of bin boundaries."
        )

    bins = list(bins)

    if len(bins) < 2:
        raise ValueError(
            "'bins' must contain at least two values to define one interval."
        )

    if labels is None:
        labels = list(range(len(bins) - 1))

    if len(labels) != len(bins) - 1:
        raise ValueError(
            f"Length of labels must equal len(bins) - 1. "
            f"Expected {len(bins) - 1}, but got {len(labels)} labels."
        )

    if category_column is None:
        category_column = f"{column}_category"

    categorized = pd.cut(
        df[column],
        bins=bins,
        labels=labels,
        include_lowest=include_lowest,
        right=right
    )
    ##categorized = categorized.cat.codes.replace(-1, pd.NA).astype("Int64")
    ##categorized = categorized.astype(object)
    
    categorized.name = category_column

    if return_df:
        df = df.copy()
        df[category_column] = categorized
        return df

    return categorized


############################################################
## Binarize a Variable
############################################################
def binarize(
    df,
    column,
    threshold=None,
    labels=(0, 1),
    binary_column=None,
    inclusive="right",
    return_df=False,
):
    """
    Convert a continuous or ordinal variable into a binary variable.

    Values below (or below/equal to) a specified threshold are assigned
    to the first category, while values above the threshold are assigned
    to the second category.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    column : str
        Variable to binarize.

    threshold : numeric
        Cut-off value separating the two groups.

    labels : tuple, default=(0, 1)
        Labels assigned to the lower and upper groups. Need not be
        numeric (e.g. ("low", "high") is valid).

    binary_column : str, optional
        Name of the output variable.
        Defaults to "<column>_binary".

    inclusive : {"left", "right"}, default="left"
        Defines which side includes the threshold.

        - "left"  : value <= threshold → labels[0]
        - "right" : value < threshold  → labels[0]

    return_df: bool, default=False
        Return a copy of the input DataFrame with the new variable
        appended, if True. Otherwise, return only the new variable as
        a pandas.Series (default).

    Returns
    -------
    pandas.Series or pandas.DataFrame
    If return_df=False (default), returns the newly created binarized
    variable as a pandas.Series.

    If return_df=True, returns a copy of the input DataFrame with the
    binarized variable added as a new binary column.
    """
    if threshold is None:
        raise ValueError(
            f'Set threshold for {column}: '
            f'{df[column].min()} < threshold < {df[column].max()}'
        )

    if len(labels) != 2:
        raise ValueError("Exactly two labels are required.")

    if binary_column is None:
        binary_column = f"{column}_binary"

    if inclusive == "left":
        mask = df[column] <= threshold
    elif inclusive == "right":
        mask = df[column] < threshold
    else:
        raise ValueError("inclusive must be 'left' or 'right'.")

    # Infer dtype from the labels, so non-numeric
    # labels (e.g., strings) work too.
    label_dtype = pd.Series(list(labels)).dtype

    binarized = pd.Series(
        np.where(mask, labels[0], labels[1]),
        index=df.index,
        name=binary_column,
        dtype=label_dtype
    ).mask(df[column].isna())

    if return_df:
        df = df.copy()
        df[binary_column] = binarized
        return df

    return binarized


############################################################
## Rename Columns
############################################################
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames dataframe columns from raw codes/German to standardized English."""
    return df.rename(columns={
        # Demographics & Socioeconomics
        'ADULT_PROB_AGE': 'Age',
        'TEILNEHMER_GESCHLECHT': 'Sex',
        'BASIS_EXAMINATION_YEAR': 'Baseline Examination Year',
        'BASIS_EXAMINATION_SEASON': 'Baseline Examination Season',
        'city_dweller': 'City Dweller',
        'SES2_SES5': 'Socioeconomic Status (Quintiles)',
        'SES2_SES3': 'Socioeconomic Status (Tiers)',  # Bottom 20%, Mid 60%, Top 20%

        # Clinical Scale Scores – Numeric
        'CES_D_SUM': 'CES-D Score',
        'GAD7_SUM': 'GAD-7 Score',
        'SWLS_SUM': 'SWLS Score',
        'IDS_SUM': 'IDS Score',
        'GDS15_SUM': 'GDS-15 Score',

        # Clinical Severities – Ordinal Categories
        'CESD_depression_severity': 'CES-D Ordinal',
        'GAD7_anxiety_severity': 'GAD-7 Ordinal',
        'SWLS_satisfaction_grade': 'SWLS Ordinal',
        'IDS_depression_severity': 'IDS Ordinal',
        'GDS15_depression_severity': 'GDS-15 Ordinal',

        # Clinical Classifications – Binary Indicators
        'CESD_clinical_depression': 'CES-D Binary',
        'GAD7_anxiety': 'GAD-7 Binary',
        'SWLS_satisfaction': 'SWLS Binary',
        'IDS_clinical_depression': 'IDS Binary',
        'GDS15_clinical_depression_risk': 'GDS-15 Binary',

        # Environmental Green Space Quartiles
        'green_100_quartile': 'Green Space Quartile (100m Buffer)',
        'green_200_quartile': 'Green Space Quartile (200m Buffer)',
        'green_500_quartile': 'Green Space Quartile (500m Buffer)',
        'green_1000_quartile': 'Green Space Quartile (1000m Buffer)',

        # Environmental NDVI Metrics
        'spring_2011_w_avg_ndvi_100': 'NDVI (100m Buffer)',
        'spring_2011_w_avg_ndvi_200': 'NDVI (200m Buffer)',
        'spring_2011_w_avg_ndvi_500': 'NDVI (500m Buffer)',
        'spring_2011_w_avg_ndvi_1000': 'NDVI (1000m Buffer)'
    })
