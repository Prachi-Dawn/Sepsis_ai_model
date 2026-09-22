import os
import glob
import pandas as pd

base_folder = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project\DataSets\Kaggle Data set"

datasets = {
    "Dataset A": os.path.join(base_folder, "training_setA raw"),
    "Dataset B": os.path.join(base_folder, "training_setB raw")
}

HORIZONS = [6, 12]


def analyze_dataset(dataset_name, folder):

    files = glob.glob(os.path.join(folder, "*.psv"))

    total_patients = len(files)

    sepsis_patients = 0
    no_sepsis_patients = 0

    early_onset_6 = 0
    early_onset_12 = 0

    usable_sepsis_6 = 0
    usable_sepsis_12 = 0

    total_prediction_patients_6 = 0
    total_prediction_patients_12 = 0

    for i, file in enumerate(files):

        df = pd.read_csv(file, sep="|")

        sepsis_rows = df[df["SepsisLabel"] == 1]

        # ---------------------------------------------
        # NO SEPSIS PATIENT
        # ---------------------------------------------

        if len(sepsis_rows) == 0:

            no_sepsis_patients += 1

        # ---------------------------------------------
        # SEPSIS PATIENT
        # ---------------------------------------------

        else:

            sepsis_patients += 1

            first_sepsis_hour = sepsis_rows["ICULOS"].min()

            # 6-hour target
            if first_sepsis_hour <= 6:

                early_onset_6 += 1

            else:

                usable_sepsis_6 += 1

            # 12-hour target
            if first_sepsis_hour <= 12:

                early_onset_12 += 1

            else:

                usable_sepsis_12 += 1

        if (i + 1) % 2000 == 0:
            print(f"{dataset_name}: {i + 1}/{total_patients}")

    # -------------------------------------------------
    # PATIENT-LEVEL DATASET SIZE
    # -------------------------------------------------

    # For both targets, all patients without sepsis can
    # participate as negative patients.
    #
    # Sepsis patients are usable only when there is enough
    # history before their first sepsis label.

    total_prediction_patients_6 = (
        no_sepsis_patients +
        usable_sepsis_6
    )

    total_prediction_patients_12 = (
        no_sepsis_patients +
        usable_sepsis_12
    )

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    print(f"Total patients: {total_patients}")
    print(f"Sepsis patients: {sepsis_patients}")
    print(f"Non-sepsis patients: {no_sepsis_patients}")

    print("\n6-HOUR TARGET")
    print("-" * 40)

    print(f"Sepsis patients usable: {usable_sepsis_6}")
    print(f"Sepsis patients excluded (onset <= 6h): {early_onset_6}")
    print(f"Non-sepsis patients: {no_sepsis_patients}")
    print(f"Total usable patients: {total_prediction_patients_6}")

    positive_patient_percent_6 = (
        usable_sepsis_6 / total_prediction_patients_6 * 100
    )

    negative_patient_percent_6 = (
        no_sepsis_patients / total_prediction_patients_6 * 100
    )

    print(f"Positive patients: {positive_patient_percent_6:.2f}%")
    print(f"Negative patients: {negative_patient_percent_6:.2f}%")

    print("\n12-HOUR TARGET")
    print("-" * 40)

    print(f"Sepsis patients usable: {usable_sepsis_12}")
    print(f"Sepsis patients excluded (onset <= 12h): {early_onset_12}")
    print(f"Non-sepsis patients: {no_sepsis_patients}")
    print(f"Total usable patients: {total_prediction_patients_12}")

    positive_patient_percent_12 = (
        usable_sepsis_12 / total_prediction_patients_12 * 100
    )

    negative_patient_percent_12 = (
        no_sepsis_patients / total_prediction_patients_12 * 100
    )

    print(f"Positive patients: {positive_patient_percent_12:.2f}%")
    print(f"Negative patients: {negative_patient_percent_12:.2f}%")


# =====================================================
# RUN
# =====================================================

for dataset_name, folder in datasets.items():

    analyze_dataset(dataset_name, folder)

print("\n" + "=" * 70)
print("FINAL TARGET ANALYSIS COMPLETE")
print("=" * 70)