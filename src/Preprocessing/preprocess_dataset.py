import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project"

INPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "ml_dataset"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "preprocessed"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# INPUT / OUTPUT FILES
# ============================================================

FILES = {
    "train": os.path.join(INPUT_FOLDER, "train.csv"),
    "validation": os.path.join(INPUT_FOLDER, "validation.csv"),
    "test": os.path.join(INPUT_FOLDER, "test.csv")
}


OUTPUT_FILES = {
    "train": os.path.join(OUTPUT_FOLDER, "train_preprocessed.csv"),
    "validation": os.path.join(OUTPUT_FOLDER, "validation_preprocessed.csv"),
    "test": os.path.join(OUTPUT_FOLDER, "test_preprocessed.csv")
}


# ============================================================
# FEATURES
# ============================================================

continuous_features = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "HospAdmTime",
    "ICULOS"
]


binary_features = [
    "Gender",
    "Unit1",
    "Unit2"
]


# ============================================================
# DATA-QUALITY REVIEW BOUNDS
# ============================================================
#
# These are broad screening bounds.
# Values outside these bounds are converted to NaN.
#
# They are NOT clinical diagnostic limits.
# ============================================================

review_bounds = {

    "HR": (20, 220),
    "O2Sat": (0, 100),
    "Temp": (25, 45),
    "SBP": (40, 300),
    "MAP": (30, 250),
    "DBP": (20, 200),
    "Resp": (1, 80),

    "EtCO2": (0, 100),
    "BaseExcess": (-40, 40),
    "HCO3": (5, 60),

    "FiO2": (0, 1),

    "pH": (6, 8.5),
    "PaCO2": (10, 150),
    "SaO2": (0, 100),

    "AST": (0, 10000),
    "BUN": (0, 300),
    "Alkalinephos": (0, 4000),
    "Calcium": (0, 30),
    "Chloride": (50, 160),
    "Creatinine": (0, 50),
    "Bilirubin_direct": (0, 50),
    "Glucose": (0, 1000),
    "Lactate": (0, 40),
    "Magnesium": (0, 15),
    "Phosphate": (0, 25),
    "Potassium": (0, 30),
    "Bilirubin_total": (0, 60),
    "TroponinI": (0, 500),
    "Hct": (0, 80),
    "Hgb": (0, 40),
    "PTT": (0, 300),
    "WBC": (0, 500),
    "Fibrinogen": (0, 3000),
    "Platelets": (0, 3000),

    "Age": (0, 120),

    "HospAdmTime": None,

    "ICULOS": (1, 1000)
}


# ============================================================
# FEATURES THAT WILL NOT BE FORWARD FILLED
# ============================================================
#
# Age, Gender and HospAdmTime describe relatively static
# patient information.
#
# ICULOS is the time index and must never be imputed.
#
# Unit1/Unit2 are handled separately.
# ============================================================

static_features = [
    "Age",
    "Gender",
    "HospAdmTime"
]


# ============================================================
# FORWARD FILL LIMIT
# ============================================================
#
# A value is carried forward for at most 6 ICU hours.
#
# This prevents a measurement from being carried forward
# indefinitely through a long missing period.
# ============================================================

FORWARD_FILL_LIMIT = 6


# ============================================================
# STEP 1: CLEAN DATA QUALITY ISSUES
# ============================================================

def apply_data_quality_rules(df):

    changed_values = 0

    for feature, bounds in review_bounds.items():

        if bounds is None:
            continue

        if feature not in df.columns:
            continue

        lower, upper = bounds

        invalid_mask = (
            (df[feature] < lower) |
            (df[feature] > upper)
        )

        count = invalid_mask.sum()

        if count > 0:

            df.loc[invalid_mask, feature] = np.nan

            changed_values += count

    return df, changed_values


# ============================================================
# STEP 2: CREATE MISSINGNESS INDICATORS
# ============================================================
#
# Missingness itself can contain useful information in ICU data.
#
# Example:
#
# Lactate = missing
#
# becomes:
#
# Lactate = imputed value
# Lactate_missing = 1
#
# ============================================================

def create_missingness_indicators(df):

    indicator_features = continuous_features.copy()

    indicator_features.remove("Age")
    indicator_features.remove("HospAdmTime")
    indicator_features.remove("ICULOS")

    for feature in indicator_features:

        if feature in df.columns:

            df[f"{feature}_missing"] = (
                df[feature].isna().astype(np.int8)
            )

    return df


# ============================================================
# STEP 3: PATIENT-WISE FORWARD FILL
# ============================================================
#
# IMPORTANT:
#
# We only use previous measurements.
#
# No backward filling.
# No interpolation.
#
# This prevents future information from entering the
# prediction at time t.
# ============================================================

def forward_fill_patient_data(df):

    df = df.sort_values(
        ["Patient_ID", "ICULOS"]
    ).copy()

    fill_features = [
        feature
        for feature in continuous_features
        if feature not in static_features
        and feature != "ICULOS"
    ]

    print(
        f"Forward filling {len(fill_features)} "
        f"time-dependent features..."
    )

    df[fill_features] = (
        df.groupby("Patient_ID", sort=False)[fill_features]
        .ffill(limit=FORWARD_FILL_LIMIT)
    )

    return df


# ============================================================
# STEP 4: TRAINING MEDIAN VALUES
# ============================================================
#
# IMPORTANT:
#
# These medians are calculated ONLY from the training set.
#
# Validation and test data will use the same training medians.
#
# This prevents information leakage.
# ============================================================

def calculate_training_medians(df):

    medians = {}

    for feature in continuous_features:

        if feature == "ICULOS":
            continue

        if feature in df.columns:

            medians[feature] = df[feature].median()

    return medians


# ============================================================
# STEP 5: APPLY MEDIAN FALLBACK
# ============================================================

def apply_median_fallback(df, medians):

    for feature, median_value in medians.items():

        if feature in df.columns:

            df[feature] = df[feature].fillna(
                median_value
            )

    return df


# ============================================================
# STEP 6: PROCESS TRAINING DATA
# ============================================================

print("=" * 70)
print("SEPSIS DATA PREPROCESSING")
print("=" * 70)

print("\nLoading training data...")

train = pd.read_csv(
    FILES["train"]
)

print(
    f"Training rows: {len(train)}"
)


# ============================================================
# DATA QUALITY
# ============================================================

print("\nApplying data-quality rules...")

train, train_invalid_count = apply_data_quality_rules(
    train
)

print(
    f"Suspicious values converted to NaN: "
    f"{train_invalid_count}"
)


# ============================================================
# MISSINGNESS INDICATORS
# ============================================================

print("\nCreating missingness indicators...")

train = create_missingness_indicators(
    train
)

print(
    "Missingness indicators created."
)


# ============================================================
# FORWARD FILL
# ============================================================

print("\nApplying patient-wise forward fill...")

train = forward_fill_patient_data(
    train
)


# ============================================================
# TRAINING MEDIANS
# ============================================================

print("\nCalculating training-set fallback medians...")

training_medians = calculate_training_medians(
    train
)

print(
    f"Training medians calculated for "
    f"{len(training_medians)} features."
)


# ============================================================
# MEDIAN FALLBACK
# ============================================================

print("\nApplying training median fallback...")

train = apply_median_fallback(
    train,
    training_medians
)


# ============================================================
# SAVE TRAINING DATA
# ============================================================

train.to_csv(
    OUTPUT_FILES["train"],
    index=False
)

print(
    f"\nTraining data saved to:\n"
    f"{OUTPUT_FILES['train']}"
)


# ============================================================
# PROCESS VALIDATION AND TEST
# ============================================================

for split in ["validation", "test"]:

    print("\n" + "=" * 70)
    print(f"PROCESSING {split.upper()} DATA")
    print("=" * 70)

    df = pd.read_csv(
        FILES[split]
    )

    print(
        f"Rows: {len(df)}"
    )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    df, invalid_count = apply_data_quality_rules(
        df
    )

    print(
        f"Suspicious values converted to NaN: "
        f"{invalid_count}"
    )

    # --------------------------------------------------------
    # MISSINGNESS INDICATORS
    # --------------------------------------------------------

    df = create_missingness_indicators(
        df
    )

    print(
        "Missingness indicators created."
    )

    # --------------------------------------------------------
    # FORWARD FILL
    # --------------------------------------------------------

    df = forward_fill_patient_data(
        df
    )

    # --------------------------------------------------------
    # TRAINING MEDIAN FALLBACK
    # --------------------------------------------------------

    df = apply_median_fallback(
        df,
        training_medians
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILES[split],
        index=False
    )

    print(
        f"Saved to:\n"
        f"{OUTPUT_FILES[split]}"
    )


# ============================================================
# SAVE TRAINING MEDIANS
# ============================================================

median_file = os.path.join(
    OUTPUT_FOLDER,
    "training_medians.csv"
)

median_df = pd.DataFrame(
    list(training_medians.items()),
    columns=[
        "Feature",
        "Training_Median"
    ]
)

median_df.to_csv(
    median_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_FOLDER
)

print("\nFiles created:")

print(
    "1. train_preprocessed.csv"
)

print(
    "2. validation_preprocessed.csv"
)

print(
    "3. test_preprocessed.csv"
)

print(
    "4. training_medians.csv"
)

print("\nImportant preprocessing rules:")
print(
    "- Suspicious values converted to missing"
)
print(
    "- Missingness indicators created"
)
print(
    "- Forward fill performed within each patient"
)
print(
    "- Forward fill limited to 6 hours"
)
print(
    "- No backward filling used"
)
print(
    "- Training medians used for final fallback"
)
print(
    "- Validation/test never used to calculate medians"
)

print("\nPreprocessing finished successfully.")