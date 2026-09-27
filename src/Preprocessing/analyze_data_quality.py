import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project"

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "ml_dataset",
    "train.csv"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "data_quality"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "data_quality_report.csv"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print(f"Training rows: {len(df)}")
print(f"Training columns: {len(df.columns)}")


# ============================================================
# FEATURES TO CHECK
# ============================================================

features = [
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
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS"
]


# ============================================================
# BROAD DATA-QUALITY REVIEW BOUNDS
# ============================================================
#
# These are NOT strict clinical diagnostic limits.
# They are broad screening thresholds used to identify
# values that deserve further investigation.
#
# A value outside these ranges is NOT automatically wrong.
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

    # FiO2 appears to use fractional representation
    # in this dataset (e.g. 0.21, 0.5, 1.0).
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

    "Gender": (0, 1),
    "Unit1": (0, 1),
    "Unit2": (0, 1),

    # HospAdmTime is a relative time variable.
    # We report its distribution but do not apply
    # a clinical validity range.
    "HospAdmTime": None,

    "ICULOS": (1, 1000)
}


# ============================================================
# ANALYZE EACH FEATURE
# ============================================================

results = []

print("\nAnalyzing data quality...\n")

for feature in features:

    if feature not in df.columns:
        print(f"WARNING: {feature} not found in dataset.")
        continue

    series = df[feature]

    total = len(series)
    missing = series.isna().sum()
    non_missing = series.notna().sum()

    if non_missing == 0:
        results.append({
            "Feature": feature,
            "Total_Values": total,
            "Missing_Count": missing,
            "Missing_Percent": 100,
            "Min": None,
            "Max": None,
            "Mean": None,
            "Median": None,
            "Outside_Bounds_Count": None,
            "Outside_Bounds_Percent": None
        })
        continue

    min_value = series.min()
    max_value = series.max()
    mean_value = series.mean()
    median_value = series.median()

    lower_bound = None
    upper_bound = None
    outside_count = 0

    if feature in review_bounds and review_bounds[feature] is not None:

        lower_bound, upper_bound = review_bounds[feature]

        outside_mask = (
            (series < lower_bound) |
            (series > upper_bound)
        )

        outside_count = outside_mask.sum()

    outside_percent = (
        outside_count / non_missing * 100
        if non_missing > 0
        else 0
    )

    results.append({
        "Feature": feature,
        "Total_Values": total,
        "Missing_Count": missing,
        "Missing_Percent": round(
            missing / total * 100, 4
        ),
        "Min": min_value,
        "Max": max_value,
        "Mean": mean_value,
        "Median": median_value,
        "Lower_Review_Bound": lower_bound,
        "Upper_Review_Bound": upper_bound,
        "Outside_Bounds_Count": outside_count,
        "Outside_Bounds_Percent": round(
            outside_percent, 4
        )
    })


# ============================================================
# CREATE REPORT
# ============================================================

report = pd.DataFrame(results)

report.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT IMPORTANT FINDINGS
# ============================================================

print("=" * 70)
print("DATA QUALITY SUMMARY")
print("=" * 70)

print(f"\nTraining rows: {len(df)}")
print(f"Features analyzed: {len(report)}")

print("\nFeatures with values outside review bounds:")

outside_features = report[
    report["Outside_Bounds_Count"].fillna(0) > 0
].sort_values(
    "Outside_Bounds_Percent",
    ascending=False
)

if len(outside_features) == 0:

    print("None found.")

else:

    for _, row in outside_features.iterrows():

        print(
            f"{row['Feature']}: "
            f"{int(row['Outside_Bounds_Count'])} values "
            f"({row['Outside_Bounds_Percent']:.4f}%)"
        )


# ============================================================
# SPECIAL FIO2 CHECK
# ============================================================

print("\n" + "=" * 70)
print("SPECIAL FiO2 CHECK")
print("=" * 70)

fio2 = df["FiO2"].dropna()

print(f"Non-missing FiO2 values: {len(fio2)}")

if len(fio2) > 0:

    print(f"Minimum FiO2: {fio2.min()}")
    print(f"Maximum FiO2: {fio2.max()}")
    print(f"Median FiO2: {fio2.median()}")

    negative_fio2 = (fio2 < 0).sum()
    above_one_fio2 = (fio2 > 1).sum()
    exactly_zero = (fio2 == 0).sum()
    exactly_one = (fio2 == 1).sum()

    print(f"FiO2 < 0: {negative_fio2}")
    print(f"FiO2 > 1: {above_one_fio2}")
    print(f"FiO2 = 0: {exactly_zero}")
    print(f"FiO2 = 1: {exactly_one}")

    print("\nLargest FiO2 values:")

    print(
        fio2
        .sort_values(ascending=False)
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# BINARY FEATURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("BINARY FEATURE CHECK")
print("=" * 70)

binary_features = [
    "Gender",
    "Unit1",
    "Unit2"
]

for feature in binary_features:

    values = df[feature].dropna().unique()

    print(
        f"{feature}: "
        f"{sorted(values.tolist())}"
    )


# ============================================================
# SAVE A LIST OF HIGH-PRIORITY ISSUES
# ============================================================

priority_features = report[
    (
        report["Outside_Bounds_Count"].fillna(0) > 0
    ) &
    (
        report["Outside_Bounds_Percent"].fillna(0) >= 1
    )
].copy()

priority_file = os.path.join(
    OUTPUT_FOLDER,
    "priority_data_quality_issues.csv"
)

priority_features.to_csv(
    priority_file,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"\nMain report:")
print(OUTPUT_FILE)

print(f"\nPriority issues:")
print(priority_file)

print("\nData quality analysis completed.")
print("No dataset values were modified.")