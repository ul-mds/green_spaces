"""
columns.py

Dataset-specific column name groupings used throughout the analysis.
These constants assume columns have already been renamed via
transform.rename_columns() where applicable (note: a few constants
below still reference raw/un-renamed source column names, e.g. the
greenspace/grey-space buffer variables that have not yet been passed
through a renaming step).

Location: utils/columns.py
"""

__all__ = [
    "baseline_categorical", "baseline_numeric",
    "ndvi",
    "greenspace_raw", "greenspace_no_outliers",
    "greenspace_tertiles", "greenspace_quartiles",
    "greenspace_quintiles", "greenspace_bins_3", "greenspace_bins_5",
    "greyspace_raw", "tree_canopy_no_outliers", "green_to_grey_ratio",
    "cesd", "ids", "gds", "gad", "swls",
    "mental_numeric", "mental_ordinal", "mental_binary",
    "mental_numeric_2", "mental_ordinal_2", "mental_binary_2",
]

########################################
# Baseline Demographics & Characteristics
baseline_categorical = [
    'Baseline Examination Year',
    'Baseline Examination Season',
    'City Dweller'
]

sex = ['Sex']

age = ['Age']

# Environmental Buffers (NDVI & Greenspace)
ndvi = [
    'NDVI (100m Buffer)',
    'NDVI (200m Buffer)',
    'NDVI (500m Buffer)',
    'NDVI (1000m Buffer)'
]

greenspace_raw = [
    'green_100', 
    'green_200', 
    'green_500', 
    'green_1000'
]

greenspace_no_outliers = [
    'green_100_rm_outl',
    'green_200_rm_outl',
    'green_500_rm_outl',
    'green_1000_rm_outl'
]

# Environmental Categorical Bins
greenspace_tertiles = [
    'green_100_tertial',
    'green_200_tertial',
    'green_500_tertial',
    'green_1000_tertial'
]

greenspace_quartiles = [
    'Green Space Quartile (100m Buffer)',
    'Green Space Quartile (200m Buffer)',
    'Green Space Quartile (500m Buffer)',
    'Green Space Quartile (1000m Buffer)'
]

greenspace_quintiles = [
    'green_100_quintile',
    'green_200_quintile',
    'green_500_quintile',
    'green_1000_quintile'
]

greenspace_bins_3 = [
    'green_100equal_bin3',
    'green_200equal_bin3',
    'green_500equal_bin3',
    'green_1000equal_bin3'
]

greenspace_bins_5 = [
    'green_100equal_bin5',
    'green_200equal_bin5',
    'green_500equal_bin5',
    'green_1000equal_bin5'
]

# Grey Infrastructure, Canopy & Ratios
greyspace_raw = [
    'grey_100', 
    'grey_200', 
    'grey_500', 
    'grey_1000'
]

tree_canopy_no_outliers = [
    'area_5_100_rm_outl',
    'area_5_200_rm_outl',
    'area_5_500_rm_outl',
    'area_5_1000_rm_outl'
]

green_to_grey_ratio = [
    'green2grey_100',
    'green2grey_200',
    'green2grey_500',
    'green2grey_1000'
]

# Clinical Scale Subsets
cesd = [
    'CES-D Score',
    'CES-D Ordinal',
    'CES-D Binary'
]

ids = [
    'IDS Score',
    'IDS Ordinal',
    'IDS Binary'
]

gds = [
    'GDS-15 Score',
    'GDS-15 Ordinal',
    'GDS-15 Binary'
]

gad = [
    'GAD-7 Score',
    'GAD-7 Ordinal',
    'GAD-7 Binary'
]

swls = [
    'SWLS Score',
    'SWLS Ordinal',
    'SWLS Binary'
]

# Aggregated Clinical Subsets
mental_numeric = [
    'CES-D Score',
    'GAD-7 Score',
    'SWLS Score'
]

mental_ordinal = [
    'CES-D Ordinal',
    'GAD-7 Ordinal',
    'SWLS Ordinal'
]

mental_binary = [
    'CES-D Binary',
    'GAD-7 Binary',
    'SWLS Binary'
]

mental_numeric_2 = [
    'IDS Score',
    'GDS-15 Score',
]

mental_ordinal_2 = [
    'IDS Ordinal',
    'GDS-15 Ordinal',
]

mental_binary_2 = [
    'IDS Binary',
    'GDS-15 Binary',
]
