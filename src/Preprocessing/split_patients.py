import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

BASE_FOLDER = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project\DataSets\Kaggle Data set"

DATASETS = {
    "Dataset A": os.path.join(BASE_FOLDER, "training_setA raw"),
    "Dataset B": os.path.join(BASE_FOLDER, "training_setB raw")
}

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "results"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# FIND ALL PATIENT FILES
# ============================================================

patients = []

for dataset_name, dataset_folder in DATASETS.items():

    print(f"\nScanning {dataset_name}...")

    if not os.path.exists(dataset_folder):
        print(f"Folder not found: {dataset_folder}")
        continue

    for root, dirs, files in os.walk(dataset_folder):

        for file_name in files:

            if not file_name.lower().endswith(".psv"):
                continue

            file_path = os.path.join(root, file_name)

            try:

                df = pd.read_csv(
                    file_path,
                    sep="|",
                    usecols=["SepsisLabel", "ICULOS"]
                )

            except Exception as e:

                print(f"Could not read: {file_path}")
                print(e)
                continue

            # Check whether patient has sepsis
            has_sepsis = (df["SepsisLabel"] == 1).any()

            # Find first sepsis hour
            if has_sepsis:

                first_sepsis_hour = df.loc[
                    df["SepsisLabel"] == 1,
                    "ICULOS"
                ].min()

                # Exclude early-sepsis patients
                if first_sepsis_hour <= 12:
                    continue

                patient_type = "sepsis"

            else:

                first_sepsis_hour = None
                patient_type = "non_sepsis"

            patient_id = (
                dataset_name.replace(" ", "_")
                + "_"
                + os.path.splitext(file_name)[0]
            )

            patients.append({
                "Patient_ID": patient_id,
                "Dataset": dataset_name,
                "File_Name": file_name,
                "File_Path": file_path,
                "Patient_Type": patient_type,
                "First_Sepsis_Hour": first_sepsis_hour
            })


# ============================================================
# CREATE PATIENT TABLE
# ============================================================

patients_df = pd.DataFrame(patients)

print("\n" + "=" * 70)
print("PATIENT COLLECTION COMPLETE")
print("=" * 70)

print(f"Eligible patients: {len(patients_df)}")

print(
    f"Sepsis patients: "
    f"{(patients_df['Patient_Type'] == 'sepsis').sum()}"
)

print(
    f"Non-sepsis patients: "
    f"{(patients_df['Patient_Type'] == 'non_sepsis').sum()}"
)


# ============================================================
# STRATIFIED TRAIN / TEMP SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    patients_df,
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    stratify=patients_df["Patient_Type"],
    random_state=RANDOM_STATE
)


# ============================================================
# VALIDATION / TEST SPLIT
# ============================================================

validation_ratio = (
    VALIDATION_SIZE /
    (VALIDATION_SIZE + TEST_SIZE)
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=(1 - validation_ratio),
    stratify=temp_df["Patient_Type"],
    random_state=RANDOM_STATE
)


# ============================================================
# ADD SPLIT LABEL
# ============================================================

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["Split"] = "train"
validation_df["Split"] = "validation"
test_df["Split"] = "test"


# ============================================================
# COMBINE
# ============================================================

split_df = pd.concat(
    [
        train_df,
        validation_df,
        test_df
    ],
    ignore_index=True
)


# ============================================================
# SAVE PATIENT SPLIT
# ============================================================

output_file = os.path.join(
    OUTPUT_FOLDER,
    "patient_split.csv"
)

split_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-LEVEL SPLIT COMPLETE")
print("=" * 70)

print(
    f"Total eligible patients: "
    f"{len(split_df)}"
)

print(
    f"Training patients: "
    f"{len(train_df)}"
)

print(
    f"Validation patients: "
    f"{len(validation_df)}"
)

print(
    f"Test patients: "
    f"{len(test_df)}"
)


print("\nPatient type distribution:")

print(
    "\nTRAIN"
)

print(
    train_df["Patient_Type"].value_counts()
)


print(
    "\nVALIDATION"
)

print(
    validation_df["Patient_Type"].value_counts()
)


print(
    "\nTEST"
)

print(
    test_df["Patient_Type"].value_counts()
)


# ============================================================
# VERIFY NO PATIENT OVERLAP
# ============================================================

train_ids = set(train_df["Patient_ID"])
validation_ids = set(validation_df["Patient_ID"])
test_ids = set(test_df["Patient_ID"])

train_validation_overlap = (
    train_ids & validation_ids
)

train_test_overlap = (
    train_ids & test_ids
)

validation_test_overlap = (
    validation_ids & test_ids
)


print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    f"Train ∩ Validation: "
    f"{len(train_validation_overlap)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(validation_test_overlap)}"
)


if (
    len(train_validation_overlap) == 0
    and len(train_test_overlap) == 0
    and len(validation_test_overlap) == 0
):

    print("\nNO PATIENT OVERLAP — SPLIT IS VALID")

else:

    print("\nWARNING: PATIENT OVERLAP DETECTED")


print("\nPatient split saved to:")
print(output_file)