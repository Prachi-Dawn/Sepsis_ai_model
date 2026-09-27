import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))
)

BASE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "DataSets",
    "Kaggle Data set"
)

SPLIT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "patient_split.csv"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "ml_dataset"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PREDICTION_HORIZON = 12


# ============================================================
# DATASET FOLDERS
# ============================================================

DATASETS = {
    "Dataset A": os.path.join(
        BASE_FOLDER,
        "training_setA raw"
    ),
    "Dataset B": os.path.join(
        BASE_FOLDER,
        "training_setB raw"
    )
}


# ============================================================
# COLUMNS TO KEEP
# ============================================================

FEATURE_COLUMNS = [
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
# LOAD PATIENT SPLIT
# ============================================================

print("=" * 70)
print("LOADING PATIENT SPLIT")
print("=" * 70)

if not os.path.exists(SPLIT_FILE):

    raise FileNotFoundError(
        f"Patient split not found:\n{SPLIT_FILE}"
    )

split_df = pd.read_csv(SPLIT_FILE)

required_split_columns = [
    "Patient_ID",
    "Dataset",
    "File_Name",
    "File_Path",
    "Patient_Type",
    "Split"
]

missing_columns = [
    column
    for column in required_split_columns
    if column not in split_df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns in patient_split.csv: "
        f"{missing_columns}"
    )

print(
    f"Patients loaded: {len(split_df)}"
)


# ============================================================
# PREPARE OUTPUT FILES
# ============================================================

output_files = {
    "train": os.path.join(
        OUTPUT_FOLDER,
        "train.csv"
    ),

    "validation": os.path.join(
        OUTPUT_FOLDER,
        "validation.csv"
    ),

    "test": os.path.join(
        OUTPUT_FOLDER,
        "test.csv"
    )
}


# Remove old files if they exist

for file_path in output_files.values():

    if os.path.exists(file_path):

        os.remove(file_path)


# ============================================================
# PROCESS PATIENTS
# ============================================================

total_patients = 0
processed_patients = 0
skipped_patients = 0

total_rows = {
    "train": 0,
    "validation": 0,
    "test": 0
}

positive_rows = {
    "train": 0,
    "validation": 0,
    "test": 0
}

negative_rows = {
    "train": 0,
    "validation": 0,
    "test": 0
}


for index, patient in split_df.iterrows():

    total_patients += 1

    patient_id = patient["Patient_ID"]
    dataset_name = patient["Dataset"]
    file_path = patient["File_Path"]
    split = patient["Split"]

    # --------------------------------------------------------
    # CHECK SPLIT
    # --------------------------------------------------------

    if split not in output_files:

        print(
            f"Invalid split for {patient_id}: {split}"
        )

        skipped_patients += 1
        continue

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not os.path.exists(file_path):

        print(
            f"File not found: {file_path}"
        )

        skipped_patients += 1
        continue

    # --------------------------------------------------------
    # READ PATIENT FILE
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            file_path,
            sep="|"
        )

    except Exception as e:

        print(
            f"Could not read {file_path}"
        )

        print(e)

        skipped_patients += 1
        continue

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = FEATURE_COLUMNS + [
        "SepsisLabel"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print(
            f"Missing columns for {patient_id}: "
            f"{missing}"
        )

        skipped_patients += 1
        continue

    # --------------------------------------------------------
    # SORT BY ICU TIME
    # --------------------------------------------------------

    df = df.sort_values(
        "ICULOS"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # FIND FIRST SEPSIS HOUR
    # --------------------------------------------------------

    sepsis_rows = df[
        df["SepsisLabel"] == 1
    ]

    has_sepsis = not sepsis_rows.empty

    if has_sepsis:

        first_sepsis_hour = sepsis_rows[
            "ICULOS"
        ].min()

        # ----------------------------------------------------
        # REMOVE EARLY-SEPSIS PATIENTS
        # ----------------------------------------------------

        if first_sepsis_hour <= PREDICTION_HORIZON:

            skipped_patients += 1
            continue

        # ----------------------------------------------------
        # KEEP ONLY PRE-SEPSIS ROWS
        # ----------------------------------------------------

        df = df[
            df["ICULOS"] < first_sepsis_hour
        ].copy()

        # ----------------------------------------------------
        # CREATE 12-HOUR TARGET
        # ----------------------------------------------------

        df["Sepsis_12h"] = (
            (first_sepsis_hour > df["ICULOS"]) &
            (
                first_sepsis_hour
                <= df["ICULOS"] + PREDICTION_HORIZON
            )
        ).astype(int)

    else:

        # ----------------------------------------------------
        # NON-SEPSIS PATIENT
        # ----------------------------------------------------

        df["Sepsis_12h"] = 0

    # --------------------------------------------------------
    # ADD PATIENT INFORMATION
    # --------------------------------------------------------

    df.insert(
        0,
        "Patient_ID",
        patient_id
    )

    df.insert(
        1,
        "Dataset",
        dataset_name
    )

    df.insert(
        2,
        "Split",
        split
    )

    # --------------------------------------------------------
    # KEEP ONLY ML COLUMNS
    # --------------------------------------------------------

    final_columns = [
        "Patient_ID",
        "Dataset",
        "Split"
    ] + FEATURE_COLUMNS + [
        "Sepsis_12h"
    ]

    df = df[final_columns]

    # --------------------------------------------------------
    # COUNT TARGETS
    # --------------------------------------------------------

    rows = len(df)

    positives = int(
        df["Sepsis_12h"].sum()
    )

    negatives = rows - positives

    total_rows[split] += rows
    positive_rows[split] += positives
    negative_rows[split] += negatives

    # --------------------------------------------------------
    # SAVE TO CSV
    # --------------------------------------------------------

    file_exists = os.path.exists(
        output_files[split]
    )

    df.to_csv(
        output_files[split],
        mode="a",
        header=not file_exists,
        index=False
    )

    processed_patients += 1

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if processed_patients % 1000 == 0:

        print(
            f"Processed patients: "
            f"{processed_patients}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ML DATASET PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Total patients: "
    f"{total_patients}"
)

print(
    f"Processed patients: "
    f"{processed_patients}"
)

print(
    f"Skipped patients: "
    f"{skipped_patients}"
)


for split in [
    "train",
    "validation",
    "test"
]:

    print("\n" + "-" * 70)

    print(
        f"{split.upper()}"
    )

    print("-" * 70)

    print(
        f"Rows: "
        f"{total_rows[split]}"
    )

    print(
        f"Positive 12h rows: "
        f"{positive_rows[split]}"
    )

    print(
        f"Negative 12h rows: "
        f"{negative_rows[split]}"
    )

    if total_rows[split] > 0:

        positive_percentage = (
            positive_rows[split]
            / total_rows[split]
            * 100
        )

        print(
            f"Positive percentage: "
            f"{positive_percentage:.2f}%"
        )


print("\n")
print("Output files:")

for split, file_path in output_files.items():

    print(
        f"{split}: {file_path}"
    )