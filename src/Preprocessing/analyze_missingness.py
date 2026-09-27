import os
import pandas as pd


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
    "missingness"
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
    f"Training rows: {len(df)}"
)

print(
    f"Training columns: {len(df.columns)}"
)


# ============================================================
# IDENTIFY FEATURE COLUMNS
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
# CALCULATE MISSINGNESS
# ============================================================

missing_count = df[FEATURE_COLUMNS].isna().sum()

total_rows = len(df)

missing_percentage = (
    missing_count
    / total_rows
    * 100
)


missingness_df = pd.DataFrame({

    "Feature": FEATURE_COLUMNS,

    "Missing_Count": missing_count.values,

    "Total_Rows": total_rows,

    "Missing_Percentage":
        missing_percentage.values,

    "Available_Count":
        total_rows - missing_count.values,

    "Available_Percentage":
        100 - missing_percentage.values
})


# ============================================================
# SORT BY MISSINGNESS
# ============================================================

missingness_df = missingness_df.sort_values(
    "Missing_Percentage",
    ascending=False
).reset_index(drop=True)


# ============================================================
# ADD MISSINGNESS CATEGORY
# ============================================================

def classify_missingness(percent):

    if percent == 0:
        return "No missing values"

    elif percent < 10:
        return "Low (<10%)"

    elif percent < 30:
        return "Moderate (10-30%)"

    elif percent < 60:
        return "High (30-60%)"

    elif percent < 90:
        return "Very high (60-90%)"

    else:
        return "Extremely high (>=90%)"


missingness_df["Missingness_Category"] = (
    missingness_df["Missing_Percentage"]
    .apply(classify_missingness)
)


# ============================================================
# SAVE FULL REPORT
# ============================================================

full_report_file = os.path.join(
    OUTPUT_FOLDER,
    "train_missingness.csv"
)

missingness_df.to_csv(
    full_report_file,
    index=False
)


# ============================================================
# SUMMARY COUNTS
# ============================================================

no_missing = (
    missingness_df["Missing_Percentage"] == 0
).sum()

low_missing = (
    (missingness_df["Missing_Percentage"] > 0) &
    (missingness_df["Missing_Percentage"] < 10)
).sum()

moderate_missing = (
    (missingness_df["Missing_Percentage"] >= 10) &
    (missingness_df["Missing_Percentage"] < 30)
).sum()

high_missing = (
    (missingness_df["Missing_Percentage"] >= 30) &
    (missingness_df["Missing_Percentage"] < 60)
).sum()

very_high_missing = (
    (missingness_df["Missing_Percentage"] >= 60) &
    (missingness_df["Missing_Percentage"] < 90)
).sum()

extreme_missing = (
    missingness_df["Missing_Percentage"] >= 90
).sum()


# ============================================================
# PRINT REPORT
# ============================================================

print("\n")
print("=" * 70)
print("MISSINGNESS ANALYSIS")
print("=" * 70)

print(
    f"Total features analyzed: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"No missing values: "
    f"{no_missing}"
)

print(
    f"Low missingness (<10%): "
    f"{low_missing}"
)

print(
    f"Moderate missingness (10-30%): "
    f"{moderate_missing}"
)

print(
    f"High missingness (30-60%): "
    f"{high_missing}"
)

print(
    f"Very high missingness (60-90%): "
    f"{very_high_missing}"
)

print(
    f"Extremely high missingness (>=90%): "
    f"{extreme_missing}"
)


# ============================================================
# PRINT FULL FEATURE TABLE
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE MISSINGNESS")
print("=" * 70)

for _, row in missingness_df.iterrows():

    print(
        f"{row['Feature']:20s}"
        f" | Missing: "
        f"{row['Missing_Count']:>8,}"
        f" | "
        f"{row['Missing_Percentage']:>6.2f}%"
        f" | "
        f"{row['Missingness_Category']}"
    )


# ============================================================
# SAVE TOP SPARSE FEATURES
# ============================================================

sparse_features = missingness_df[
    missingness_df["Missing_Percentage"] >= 90
].copy()

sparse_file = os.path.join(
    OUTPUT_FOLDER,
    "extremely_sparse_features.csv"
)

sparse_features.to_csv(
    sparse_file,
    index=False
)


# ============================================================
# SAVE FEATURES WITH LOW MISSINGNESS
# ============================================================

low_missing_features = missingness_df[
    missingness_df["Missing_Percentage"] < 30
].copy()

low_missing_file = os.path.join(
    OUTPUT_FOLDER,
    "features_below_30_percent_missing.csv"
)

low_missing_features.to_csv(
    low_missing_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("MISSINGNESS ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nFull report:"
)

print(full_report_file)

print(
    "\nExtremely sparse features:"
)

print(sparse_file)

print(
    "\nFeatures below 30% missingness:"
)

print(low_missing_file)