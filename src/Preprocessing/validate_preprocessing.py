import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project"

PREPROCESSED_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "preprocessed"
)

ML_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "ml_dataset"
)

SPLIT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "patient_split.csv"
)


# ============================================================
# FILES
# ============================================================

FILES = {
    "train": os.path.join(
        PREPROCESSED_FOLDER,
        "train_preprocessed.csv"
    ),
    "validation": os.path.join(
        PREPROCESSED_FOLDER,
        "validation_preprocessed.csv"
    ),
    "test": os.path.join(
        PREPROCESSED_FOLDER,
        "test_preprocessed.csv"
    )
}

ORIGINAL_FILES = {
    "train": os.path.join(
        ML_FOLDER,
        "train.csv"
    ),
    "validation": os.path.join(
        ML_FOLDER,
        "validation.csv"
    ),
    "test": os.path.join(
        ML_FOLDER,
        "test.csv"
    )
}


# ============================================================
# EXPECTED COLUMNS
# ============================================================

META_COLUMNS = [
    "Patient_ID",
    "Dataset",
    "Split",
    "Sepsis_12h"
]

BINARY_FEATURES = [
    "Gender",
    "Unit1",
    "Unit2"
]

CONTINUOUS_FEATURES = [
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


# ============================================================
# HELPER
# ============================================================

def print_result(name, passed, details=""):
    if passed:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")

    if details:
        print(f"       {details}")


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PREPROCESSING VALIDATION")
print("=" * 70)

all_checks_passed = True


# ============================================================
# 1. CHECK FILES EXIST
# ============================================================

print("\n1. CHECKING PREPROCESSED FILES")

for split, path in FILES.items():

    if os.path.exists(path):
        print_result(
            f"{split} file exists",
            True,
            path
        )
    else:
        print_result(
            f"{split} file exists",
            False,
            path
        )
        all_checks_passed = False


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading preprocessed datasets...")

data = {}
original_data = {}

for split in ["train", "validation", "test"]:

    if not os.path.exists(FILES[split]):
        continue

    print(f"Loading {split}...")

    data[split] = pd.read_csv(FILES[split])

    if os.path.exists(ORIGINAL_FILES[split]):
        original_data[split] = pd.read_csv(
            ORIGINAL_FILES[split]
        )


# ============================================================
# 2. CHECK ROW COUNTS
# ============================================================

print("\n2. CHECKING ROW COUNTS")

expected_rows = {
    "train": 1063284,
    "validation": 230305,
    "test": 227860
}

for split in data:

    actual = len(data[split])
    expected = expected_rows[split]

    passed = actual == expected

    print_result(
        f"{split} row count",
        passed,
        f"Expected: {expected:,} | Actual: {actual:,}"
    )

    if not passed:
        all_checks_passed = False


# ============================================================
# 3. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n3. CHECKING SEPSIS_12H TARGET")

expected_targets = {
    "train": {
        0: 1046854,
        1: 16430
    },
    "validation": {
        0: 226783,
        1: 3522
    },
    "test": {
        0: 224333,
        1: 3527
    }
}

for split in data:

    df = data[split]

    if "Sepsis_12h" not in df.columns:
        print_result(
            f"{split} target column exists",
            False
        )
        all_checks_passed = False
        continue

    values = set(df["Sepsis_12h"].dropna().unique())

    valid_values = values.issubset({0, 1})

    print_result(
        f"{split} target contains only 0/1",
        valid_values,
        f"Found values: {sorted(values)}"
    )

    if not valid_values:
        all_checks_passed = False

    counts = df["Sepsis_12h"].value_counts().to_dict()

    expected = expected_targets[split]

    counts_match = (
        counts.get(0, 0) == expected[0]
        and counts.get(1, 0) == expected[1]
    )

    print_result(
        f"{split} target distribution",
        counts_match,
        f"Expected: 0={expected[0]:,}, 1={expected[1]:,} | "
        f"Actual: 0={counts.get(0, 0):,}, "
        f"1={counts.get(1, 0):,}"
    )

    if not counts_match:
        all_checks_passed = False


# ============================================================
# 4. CHECK TARGET WAS NOT MODIFIED
# ============================================================

print("\n4. CHECKING TARGET AGAINST ORIGINAL ML DATASET")

for split in data:

    if split not in original_data:
        print_result(
            f"{split} target comparison",
            False,
            "Original ML dataset not found"
        )
        all_checks_passed = False
        continue

    df_processed = data[split]
    df_original = original_data[split]

    processed_counts = (
        df_processed["Sepsis_12h"]
        .value_counts()
        .sort_index()
    )

    original_counts = (
        df_original["Sepsis_12h"]
        .value_counts()
        .sort_index()
    )

    passed = processed_counts.equals(original_counts)

    print_result(
        f"{split} target unchanged",
        passed
    )

    if not passed:
        print(
            "       Original:",
            original_counts.to_dict()
        )
        print(
            "       Processed:",
            processed_counts.to_dict()
        )

        all_checks_passed = False


# ============================================================
# 5. CHECK REQUIRED META COLUMNS
# ============================================================

print("\n5. CHECKING REQUIRED META COLUMNS")

for split, df in data.items():

    missing_columns = [
        col for col in META_COLUMNS
        if col not in df.columns
    ]

    passed = len(missing_columns) == 0

    print_result(
        f"{split} metadata columns",
        passed,
        "Missing: " + str(missing_columns)
        if missing_columns
        else "All required columns present"
    )

    if not passed:
        all_checks_passed = False


# ============================================================
# 6. CHECK PATIENT IDs
# ============================================================

print("\n6. CHECKING PATIENT IDs")

for split, df in data.items():

    missing_ids = df["Patient_ID"].isna().sum()

    unique_patients = df["Patient_ID"].nunique()

    passed = missing_ids == 0

    print_result(
        f"{split} Patient_ID completeness",
        passed,
        f"Missing IDs: {missing_ids:,} | "
        f"Unique patients: {unique_patients:,}"
    )

    if not passed:
        all_checks_passed = False


# ============================================================
# 7. CHECK PATIENT OVERLAP
# ============================================================

print("\n7. CHECKING PATIENT OVERLAP")

patient_sets = {}

for split, df in data.items():
    patient_sets[split] = set(df["Patient_ID"].unique())

train_val = patient_sets["train"] & patient_sets["validation"]
train_test = patient_sets["train"] & patient_sets["test"]
val_test = patient_sets["validation"] & patient_sets["test"]

print_result(
    "Train vs Validation overlap",
    len(train_val) == 0,
    f"Overlapping patients: {len(train_val)}"
)

print_result(
    "Train vs Test overlap",
    len(train_test) == 0,
    f"Overlapping patients: {len(train_test)}"
)

print_result(
    "Validation vs Test overlap",
    len(val_test) == 0,
    f"Overlapping patients: {len(val_test)}"
)

if len(train_val) > 0 or len(train_test) > 0 or len(val_test) > 0:
    all_checks_passed = False


# ============================================================
# 8. CHECK ICULOS ORDER
# ============================================================

print("\n8. CHECKING ICULOS ORDER WITHIN PATIENTS")

for split, df in data.items():

    problems = 0
    duplicate_rows = 0

    for patient_id, group in df.groupby("Patient_ID"):

        icu = group["ICULOS"]

        if not icu.is_monotonic_increasing:
            problems += 1

        duplicate_rows += icu.duplicated().sum()

    passed = problems == 0 and duplicate_rows == 0

    print_result(
        f"{split} ICULOS ordering",
        passed,
        f"Patients with ordering problems: {problems} | "
        f"Duplicate Patient_ID + ICULOS rows: {duplicate_rows}"
    )

    if not passed:
        all_checks_passed = False


# ============================================================
# 9. CHECK NaN VALUES
# ============================================================

print("\n9. CHECKING REMAINING MISSING VALUES")

for split, df in data.items():

    total_missing = df.isna().sum().sum()

    binary_missing = {
        col: int(df[col].isna().sum())
        for col in BINARY_FEATURES
        if col in df.columns
    }

    continuous_missing = {
        col: int(df[col].isna().sum())
        for col in CONTINUOUS_FEATURES
        if col in df.columns
    }

    print(f"\n{split.upper()}")
    print(f"Total remaining NaN values: {total_missing:,}")

    print("Binary feature missing values:")

    for col, count in binary_missing.items():
        print(f"  {col}: {count:,}")

    print("Continuous feature missing values:")

    remaining_continuous = {
        col: count
        for col, count in continuous_missing.items()
        if count > 0
    }

    if remaining_continuous:
        for col, count in remaining_continuous.items():
            print(f"  {col}: {count:,}")
    else:
        print("  None")

    # Continuous features should have no missing values.
    continuous_pass = len(remaining_continuous) == 0

    print_result(
        f"{split} continuous features complete",
        continuous_pass
    )

    if not continuous_pass:
        all_checks_passed = False


# ============================================================
# 10. CHECK MISSINGNESS INDICATORS
# ============================================================

print("\n10. CHECKING MISSINGNESS INDICATORS")

for split, df in data.items():

    indicator_columns = [
        col
        for col in df.columns
        if col.endswith("_missing")
    ]

    invalid_indicators = []

    for col in indicator_columns:

        values = set(
            df[col].dropna().unique()
        )

        if not values.issubset({0, 1}):
            invalid_indicators.append(
                (col, sorted(values))
            )

    passed = len(invalid_indicators) == 0

    print_result(
        f"{split} missingness indicators",
        passed,
        f"Indicators found: {len(indicator_columns)}"
    )

    if not passed:

        for col, values in invalid_indicators:
            print(
                f"       {col}: {values}"
            )

        all_checks_passed = False


# ============================================================
# 11. CHECK SUSPICIOUS VALUES
# ============================================================

print("\n11. CHECKING SUSPICIOUS VALUES AFTER PREPROCESSING")

REVIEW_BOUNDS = {
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
    "ICULOS": (1, 1000)
}

for split, df in data.items():

    suspicious_count = 0

    for feature, (lower, upper) in REVIEW_BOUNDS.items():

        if feature not in df.columns:
            continue

        values = df[feature]

        suspicious = (
            (values < lower) |
            (values > upper)
        ).sum()

        suspicious_count += suspicious

    passed = suspicious_count == 0

    print_result(
        f"{split} suspicious-value check",
        passed,
        f"Suspicious observations: {suspicious_count:,}"
    )

    if not passed:
        all_checks_passed = False


# ============================================================
# 12. CHECK BINARY FEATURES
# ============================================================

print("\n12. CHECKING BINARY FEATURES")

for split, df in data.items():

    for feature in BINARY_FEATURES:

        if feature not in df.columns:
            continue

        values = set(
            df[feature].dropna().unique()
        )

        passed = values.issubset({0, 1})

        print_result(
            f"{split} {feature}",
            passed,
            f"Values found: {sorted(values)}"
        )

        if not passed:
            all_checks_passed = False


# ============================================================
# 13. CHECK SPLIT COLUMN
# ============================================================

print("\n13. CHECKING SPLIT LABELS")

for split, df in data.items():

    values = set(
        df["Split"].dropna().unique()
    )

    expected = {split}

    passed = values == expected

    print_result(
        f"{split} Split column",
        passed,
        f"Values found: {sorted(values)}"
    )

    if not passed:
        all_checks_passed = False


# ============================================================
# 14. CHECK DATASET COLUMN
# ============================================================

print("\n14. CHECKING DATASET LABELS")

for split, df in data.items():

    values = set(
        df["Dataset"].dropna().unique()
    )

    passed = values.issubset({
        "A",
        "B",
        "Dataset_A",
        "Dataset_B"
    })

    print_result(
        f"{split} Dataset column",
        passed,
        f"Values found: {sorted(values)}"
    )

    if not passed:
        all_checks_passed = False


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

if all_checks_passed:

    print("\nALL VALIDATION CHECKS PASSED.")
    print("\nThe preprocessing pipeline is structurally valid.")
    print("The next step can be baseline model training.")

else:

    print("\nSOME VALIDATION CHECKS FAILED.")
    print("\nDo NOT start model training yet.")
    print("Fix the failed checks first.")

print("=" * 70)