import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

BASE_FOLDER = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project\DataSets\Kaggle Data set"

DATASETS = {
    "Dataset A": os.path.join(BASE_FOLDER, "training_setA raw"),
    "Dataset B": os.path.join(BASE_FOLDER, "training_setB raw")
}

PREDICTION_HORIZON = 12

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "results"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# CREATE 12-HOUR TARGET FOR ONE PATIENT
# ============================================================

def create_12h_target(df):

    # Check required columns
    if "SepsisLabel" not in df.columns:
        return None, "missing_sepsis_label"

    if "ICULOS" not in df.columns:
        return None, "missing_iculos"

    # Sort by ICU time
    df = df.sort_values("ICULOS").reset_index(drop=True)

    # --------------------------------------------------------
    # CHECK WHETHER PATIENT DEVELOPS SEPSIS
    # --------------------------------------------------------

    sepsis_rows = df[df["SepsisLabel"] == 1]

    # --------------------------------------------------------
    # NON-SEPSIS PATIENT
    # --------------------------------------------------------

    if sepsis_rows.empty:

        df["Sepsis_12h"] = 0

        return df, "non_sepsis"

    # --------------------------------------------------------
    # FIRST SEPSIS HOUR
    # --------------------------------------------------------

    first_sepsis_hour = sepsis_rows["ICULOS"].min()

    # --------------------------------------------------------
    # EARLY SEPSIS PATIENT
    #
    # If sepsis starts within the first 12 hours,
    # there is not a complete 12-hour prediction window.
    # --------------------------------------------------------

    if first_sepsis_hour <= PREDICTION_HORIZON:

        return None, "early_sepsis"

    # --------------------------------------------------------
    # KEEP ONLY ROWS BEFORE FIRST SEPSIS
    #
    # Once sepsis has already started, those rows are no
    # longer future prediction examples.
    # --------------------------------------------------------

    df = df[
        df["ICULOS"] < first_sepsis_hour
    ].copy()

    # --------------------------------------------------------
    # CREATE 12-HOUR TARGET
    #
    # Target = 1 when sepsis will occur within the next
    # 12 hours.
    #
    # first_sepsis_hour > current hour
    # AND
    # first_sepsis_hour <= current hour + 12
    # --------------------------------------------------------

    df["Sepsis_12h"] = (
        (first_sepsis_hour > df["ICULOS"]) &
        (first_sepsis_hour <= df["ICULOS"] + PREDICTION_HORIZON)
    ).astype(int)

    return df, "sepsis"


# ============================================================
# PROCESS DATASETS
# ============================================================

summary = []

total_patients = 0
total_sepsis_patients = 0
excluded_early_patients = 0
usable_sepsis_patients = 0
non_sepsis_patients = 0

total_rows = 0
positive_target_rows = 0
negative_target_rows = 0

files_with_errors = 0


for dataset_name, dataset_folder in DATASETS.items():

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    if not os.path.exists(dataset_folder):

        print(f"Folder not found: {dataset_folder}")
        continue

    dataset_total = 0
    dataset_sepsis = 0
    dataset_excluded = 0
    dataset_usable_sepsis = 0
    dataset_non_sepsis = 0

    dataset_rows = 0
    dataset_positive = 0
    dataset_negative = 0

    # --------------------------------------------------------
    # FIND PSV FILES
    # --------------------------------------------------------

    psv_files = []

    for root, dirs, files in os.walk(dataset_folder):

        for file_name in files:

            if file_name.lower().endswith(".psv"):

                psv_files.append(
                    os.path.join(root, file_name)
                )

    # --------------------------------------------------------
    # PROCESS EACH PATIENT
    # --------------------------------------------------------

    for file_path in psv_files:

        dataset_total += 1
        total_patients += 1

        try:

            df = pd.read_csv(
                file_path,
                sep="|"
            )

        except Exception as e:

            files_with_errors += 1

            print(
                f"Could not read: {file_path}"
            )

            print(e)

            continue

        processed_df, status = create_12h_target(df)

        # ----------------------------------------------------
        # EARLY SEPSIS
        # ----------------------------------------------------

        if status == "early_sepsis":

            dataset_sepsis += 1
            dataset_excluded += 1

            total_sepsis_patients += 1
            excluded_early_patients += 1

            continue

        # ----------------------------------------------------
        # MISSING REQUIRED COLUMNS
        # ----------------------------------------------------

        if status in [
            "missing_sepsis_label",
            "missing_iculos"
        ]:

            files_with_errors += 1

            continue

        # ----------------------------------------------------
        # NON-SEPSIS PATIENT
        # ----------------------------------------------------

        if status == "non_sepsis":

            dataset_non_sepsis += 1
            non_sepsis_patients += 1

        # ----------------------------------------------------
        # USABLE SEPSIS PATIENT
        # ----------------------------------------------------

        elif status == "sepsis":

            dataset_sepsis += 1
            dataset_usable_sepsis += 1

            total_sepsis_patients += 1
            usable_sepsis_patients += 1

        # ----------------------------------------------------
        # COUNT ROWS
        # ----------------------------------------------------

        rows = len(processed_df)

        dataset_rows += rows
        total_rows += rows

        positives = int(
            processed_df["Sepsis_12h"].sum()
        )

        negatives = rows - positives

        dataset_positive += positives
        dataset_negative += negatives

        positive_target_rows += positives
        negative_target_rows += negatives

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    usable_patients = (
        dataset_usable_sepsis +
        dataset_non_sepsis
    )

    summary.append({

        "Dataset": dataset_name,

        "Total_Patients": dataset_total,

        "Sepsis_Patients": dataset_sepsis,

        "Early_Sepsis_Excluded":
            dataset_excluded,

        "Usable_Sepsis_Patients":
            dataset_usable_sepsis,

        "Non_Sepsis_Patients":
            dataset_non_sepsis,

        "Usable_Total_Patients":
            usable_patients,

        "Total_Usable_Rows":
            dataset_rows,

        "Positive_12h_Rows":
            dataset_positive,

        "Negative_12h_Rows":
            dataset_negative
    })

    # --------------------------------------------------------
    # PRINT DATASET RESULTS
    # --------------------------------------------------------

    print(
        f"Total patients: "
        f"{dataset_total}"
    )

    print(
        f"Sepsis patients: "
        f"{dataset_sepsis}"
    )

    print(
        f"Early sepsis excluded: "
        f"{dataset_excluded}"
    )

    print(
        f"Usable sepsis patients: "
        f"{dataset_usable_sepsis}"
    )

    print(
        f"Non-sepsis patients: "
        f"{dataset_non_sepsis}"
    )

    print(
        f"Usable total patients: "
        f"{usable_patients}"
    )

    print(
        f"Positive 12h rows: "
        f"{dataset_positive}"
    )

    print(
        f"Negative 12h rows: "
        f"{dataset_negative}"
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = os.path.join(
    OUTPUT_FOLDER,
    "12h_target_generation_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("12-HOUR TARGET GENERATION COMPLETE")
print("=" * 70)

print(
    f"Total patients processed: "
    f"{total_patients}"
)

print(
    f"Sepsis patients: "
    f"{total_sepsis_patients}"
)

print(
    f"Early sepsis patients excluded: "
    f"{excluded_early_patients}"
)

print(
    f"Usable sepsis patients: "
    f"{usable_sepsis_patients}"
)

print(
    f"Non-sepsis patients: "
    f"{non_sepsis_patients}"
)

print(
    f"Total usable rows: "
    f"{total_rows}"
)

print(
    f"Positive 12h target rows: "
    f"{positive_target_rows}"
)

print(
    f"Negative 12h target rows: "
    f"{negative_target_rows}"
)

print(
    f"Files with errors: "
    f"{files_with_errors}"
)

print("\nSummary saved to:")
print(summary_file)