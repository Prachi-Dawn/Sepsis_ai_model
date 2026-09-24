import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

base_folder = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project\DataSets\Kaggle Data set"

datasets = {
    "Dataset A": os.path.join(base_folder, "training_setA raw"),
    "Dataset B": os.path.join(base_folder, "training_setB raw")
}

# Number of sepsis patients to inspect
MAX_PATIENTS = 20

# Hours around the 12-hour prediction point to display
HOURS_AROUND_TARGET = 3


# ============================================================
# FIND PSV FILES FROM BOTH DATASETS
# ============================================================

print("Looking for PSV files...\n")

psv_files = []

for dataset_name, dataset_folder in datasets.items():

    print(f"Checking {dataset_name}:")
    print(dataset_folder)

    if not os.path.exists(dataset_folder):
        print("Folder not found!\n")
        continue

    dataset_count = 0

    for root, dirs, files in os.walk(dataset_folder):

        for file in files:

            if file.lower().endswith(".psv"):

                full_path = os.path.join(root, file)

                psv_files.append(full_path)

                dataset_count += 1

    print(f"PSV files found: {dataset_count}\n")


print("=" * 60)
print(f"TOTAL PSV FILES FOUND: {len(psv_files)}")
print("=" * 60)


if len(psv_files) == 0:

    print("\nNo PSV files found.")
    print("Please check your folder paths.")

    exit()


# ============================================================
# INSPECT SEPSIS PATIENTS
# ============================================================

sepsis_patients_found = 0

results = []

print("\nSearching for patients with sepsis...\n")


for file_path in psv_files:

    try:

        df = pd.read_csv(
            file_path,
            sep="|"
        )

    except Exception as e:

        print(f"Could not read: {file_path}")
        print(e)

        continue


    # --------------------------------------------------------
    # CHECK FOR SEPSIS LABEL
    # --------------------------------------------------------

    if "SepsisLabel" not in df.columns:

        print(f"SepsisLabel missing: {file_path}")

        continue


    # --------------------------------------------------------
    # FIND SEPSIS ROWS
    # --------------------------------------------------------

    sepsis_rows = df[df["SepsisLabel"] == 1]


    # Skip patients who never develop sepsis
    if sepsis_rows.empty:

        continue


    # --------------------------------------------------------
    # FIRST SEPSIS LABEL
    # --------------------------------------------------------

    first_sepsis_index = sepsis_rows.index[0]

    patient_id = os.path.basename(file_path)

    print("=" * 70)

    print(f"Patient: {patient_id}")

    print(
        f"First SepsisLabel = 1 at row: "
        f"{first_sepsis_index}"
    )


    # --------------------------------------------------------
    # 12 HOURS BEFORE FIRST SEPSIS LABEL
    # --------------------------------------------------------

    target_index = first_sepsis_index - 12

    print(
        f"12 hours before first sepsis label: "
        f"row {target_index}"
    )


    # --------------------------------------------------------
    # NOT ENOUGH HISTORY
    # --------------------------------------------------------

    if target_index < 0:

        print(
            "Not enough data before the sepsis event."
        )

        print()

        continue


    # --------------------------------------------------------
    # SELECT ROWS AROUND 12-HOUR POINT
    # --------------------------------------------------------

    start_index = max(
        0,
        target_index - HOURS_AROUND_TARGET
    )

    end_index = min(
        len(df) - 1,
        target_index + HOURS_AROUND_TARGET
    )


    # --------------------------------------------------------
    # IMPORTANT COLUMNS TO DISPLAY
    # --------------------------------------------------------

    possible_columns = [

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

        "Hgb",
        "Hct",
        "WBC",
        "Platelets",

        "Age",
        "Gender",
        "HospAdmTime",
        "ICULOS",

        "SepsisLabel"
    ]


    columns_to_show = []

    for column in possible_columns:

        if column in df.columns:

            columns_to_show.append(column)


    # --------------------------------------------------------
    # CREATE DISPLAY DATA
    # --------------------------------------------------------

    display_df = df.loc[
        start_index:end_index,
        columns_to_show
    ].copy()


    # Add row number
    display_df.insert(
        0,
        "Hour_Row",
        display_df.index
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\nData around the 12-hour prediction point:"
    )

    print(
        display_df.to_string()
    )


    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    results.append({

        "Patient": patient_id,

        "Dataset": (
            "Dataset A"
            if "training_setA raw" in file_path
            else "Dataset B"
        ),

        "First_Sepsis_Row": first_sepsis_index,

        "12_Hour_Target_Row": target_index,

        "Total_Rows": len(df)
    })


    sepsis_patients_found += 1

    print()


    # Stop after enough patients
    if sepsis_patients_found >= MAX_PATIENTS:

        break


# ============================================================
# SAVE RESULTS
# ============================================================

if len(results) > 0:

    results_df = pd.DataFrame(results)

    output_file = os.path.join(
        os.path.dirname(__file__),
        "12h_sepsis_inspection_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )


    print("=" * 70)

    print("INSPECTION COMPLETE")

    print("=" * 70)

    print(
        f"Sepsis patients inspected: "
        f"{len(results_df)}"
    )

    print(
        f"Results saved to:"
    )

    print(output_file)

    print("\nSummary:")

    print(
        results_df.to_string(index=False)
    )


else:

    print(
        "\nNo sepsis patients were found."
    )