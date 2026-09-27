import os
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "ml_dataset",
    "train.csv"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "feature_analysis"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

if not os.path.exists(TRAIN_FILE):

    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

df = pd.read_csv(TRAIN_FILE)

print(
    f"Training rows: {len(df):,}"
)

print(
    f"Training columns: {len(df.columns)}"
)


# ============================================================
# COLUMNS TO EXCLUDE
# ============================================================

EXCLUDE_COLUMNS = [
    "Patient_ID",
    "Dataset",
    "Split",
    "Sepsis_12h"
]

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDE_COLUMNS
]


# ============================================================
# BASIC FEATURE INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE TYPES")
print("=" * 70)

for column in FEATURE_COLUMNS:

    print(
        f"{column:20s} | "
        f"{str(df[column].dtype):10s} | "
        f"Unique: {df[column].nunique(dropna=True):,}"
    )


# ============================================================
# NUMERIC FEATURE STATISTICS
# ============================================================

numeric_features = df[
    FEATURE_COLUMNS
].select_dtypes(
    include=[np.number]
).columns.tolist()


statistics = []

for feature in numeric_features:

    series = df[feature]

    statistics.append({

        "Feature": feature,

        "Data_Type": str(series.dtype),

        "Missing_Count":
            int(series.isna().sum()),

        "Missing_Percentage":
            float(series.isna().mean() * 100),

        "Unique_Values":
            int(series.nunique(dropna=True)),

        "Minimum":
            series.min(),

        "Maximum":
            series.max(),

        "Mean":
            series.mean(),

        "Median":
            series.median(),

        "Std_Dev":
            series.std(),

        "Q1":
            series.quantile(0.25),

        "Q3":
            series.quantile(0.75)
    })


statistics_df = pd.DataFrame(statistics)


# ============================================================
# INTERQUARTILE RANGE AND OUTLIERS
# ============================================================

statistics_df["IQR"] = (
    statistics_df["Q3"]
    - statistics_df["Q1"]
)

statistics_df["Lower_IQR_Bound"] = (
    statistics_df["Q1"]
    - 1.5 * statistics_df["IQR"]
)

statistics_df["Upper_IQR_Bound"] = (
    statistics_df["Q3"]
    + 1.5 * statistics_df["IQR"]
)


# ============================================================
# COUNT EXTREME VALUES
# ============================================================

outlier_counts = []

for feature in numeric_features:

    series = df[feature].dropna()

    if len(series) == 0:

        outlier_counts.append(0)
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (series < lower_bound) |
        (series > upper_bound)
    )

    outlier_counts.append(
        int(outliers.sum())
    )


statistics_df["IQR_Outlier_Count"] = (
    outlier_counts
)

statistics_df["IQR_Outlier_Percentage"] = (
    statistics_df["IQR_Outlier_Count"]
    /
    (
        len(df)
        - statistics_df["Missing_Count"]
    )
    * 100
)


# ============================================================
# CONSTANT FEATURES
# ============================================================

statistics_df["Constant_Feature"] = (
    statistics_df["Unique_Values"] <= 1
)


# ============================================================
# SAVE STATISTICS
# ============================================================

statistics_file = os.path.join(
    OUTPUT_FOLDER,
    "feature_statistics.csv"
)

statistics_df.to_csv(
    statistics_file,
    index=False
)


# ============================================================
# PRINT NUMERIC STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("NUMERIC FEATURE STATISTICS")
print("=" * 70)

display_columns = [
    "Feature",
    "Missing_Percentage",
    "Unique_Values",
    "Minimum",
    "Maximum",
    "Mean",
    "Median",
    "Std_Dev",
    "IQR_Outlier_Percentage"
]

print(
    statistics_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# CATEGORICAL / LOW-CARDINALITY FEATURES
# ============================================================

print("\n")
print("=" * 70)
print("LOW-CARDINALITY FEATURES")
print("=" * 70)

low_cardinality_features = []

for feature in FEATURE_COLUMNS:

    unique_count = df[
        feature
    ].nunique(
        dropna=True
    )

    if unique_count <= 10:

        low_cardinality_features.append(
            feature
        )

        print(
            f"\n{feature}"
        )

        print(
            df[feature]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


# ============================================================
# SAVE LOW-CARDINALITY REPORT
# ============================================================

low_cardinality_data = []

for feature in low_cardinality_features:

    counts = (
        df[feature]
        .value_counts(
            dropna=False
        )
    )

    for value, count in counts.items():

        low_cardinality_data.append({

            "Feature": feature,

            "Value": value,

            "Count": int(count),

            "Percentage":
                float(
                    count /
                    len(df)
                    * 100
                )
        })


low_cardinality_df = pd.DataFrame(
    low_cardinality_data
)

low_cardinality_file = os.path.join(
    OUTPUT_FOLDER,
    "low_cardinality_features.csv"
)

low_cardinality_df.to_csv(
    low_cardinality_file,
    index=False
)


# ============================================================
# CONSTANT FEATURES REPORT
# ============================================================

constant_features = statistics_df[
    statistics_df["Constant_Feature"]
].copy()

constant_file = os.path.join(
    OUTPUT_FOLDER,
    "constant_features.csv"
)

constant_features.to_csv(
    constant_file,
    index=False
)


# ============================================================
# HIGH OUTLIER FEATURES
# ============================================================

high_outlier_features = statistics_df[
    statistics_df["IQR_Outlier_Percentage"] > 10
].copy()

high_outlier_features = (
    high_outlier_features
    .sort_values(
        "IQR_Outlier_Percentage",
        ascending=False
    )
)

high_outlier_file = os.path.join(
    OUTPUT_FOLDER,
    "high_outlier_features.csv"
)

high_outlier_features.to_csv(
    high_outlier_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE DISTRIBUTION ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Numeric features analyzed: "
    f"{len(numeric_features)}"
)

print(
    f"Low-cardinality features: "
    f"{len(low_cardinality_features)}"
)

print(
    f"Constant features: "
    f"{len(constant_features)}"
)

print(
    f"Features with >10% IQR outliers: "
    f"{len(high_outlier_features)}"
)

print("\nOutput files:")

print(
    f"\nFeature statistics:\n"
    f"{statistics_file}"
)

print(
    f"\nLow-cardinality features:\n"
    f"{low_cardinality_file}"
)

print(
    f"\nConstant features:\n"
    f"{constant_file}"
)

print(
    f"\nHigh-outlier features:\n"
    f"{high_outlier_file}"
)