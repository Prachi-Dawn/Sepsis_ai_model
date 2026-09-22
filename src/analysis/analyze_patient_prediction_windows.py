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

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    total_patients = len(files)

    sepsis_patients = 0
    no_sepsis_patients = 0

    patients_with_usable_6h = 0
    patients_with_usable_12h = 0

    early_6h = 0
    early_12h = 0

    # Number of positive prediction hours per sepsis patient
    positive_hours_6h = []
    positive_hours_12h = []

    # Store onset information
    first_sepsis_hours = []

    for i, file in enumerate(files):

        df = pd.read_csv(file, sep="|")

        sepsis_rows = df[df["SepsisLabel"] == 1]

        # --------------------------------------------------
        # PATIENT WITHOUT SEPSIS
        # --------------------------------------------------

        if len(sepsis_rows) == 0:

            no_sepsis_patients += 1

        # --------------------------------------------------
        # PATIENT WITH SEPSIS
        # --------------------------------------------------

        else:

            sepsis_patients += 1

            first_sepsis_hour = sepsis_rows["ICULOS"].min()

            first_sepsis_hours.append(first_sepsis_hour)

            # ----------------------------------------------
            # 6-HOUR WINDOW
            # ----------------------------------------------

            if first_sepsis_hour <= 6:

                early_6h += 1

            else:

                patients_with_usable_6h += 1

                count_6h = 0

                for _, row in df.iterrows():

                    t = row["ICULOS"]

                    # Only use hours BEFORE sepsis
                    if t >= first_sepsis_hour:
                        continue

                    if first_sepsis_hour > t and first_sepsis_hour <= t + 6:
                        count_6h += 1

                positive_hours_6h.append(count_6h)

            # ----------------------------------------------
            # 12-HOUR WINDOW
            # ----------------------------------------------

            if first_sepsis_hour <= 12:

                early_12h += 1

            else:

                patients_with_usable_12h += 1

                count_12h = 0

                for _, row in df.iterrows():

                    t = row["ICULOS"]

                    # Only use hours BEFORE sepsis
                    if t >= first_sepsis_hour:
                        continue

                    if first_sepsis_hour > t and first_sepsis_hour <= t + 12:
                        count_12h += 1

                positive_hours_12h.append(count_12h)

        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1}/{total_patients}")

    # ======================================================
    # RESULTS
    # ======================================================

    print("\nPATIENT COUNTS")
    print("-" * 40)

    print(f"Total patients: {total_patients}")
    print(f"Patients with sepsis: {sepsis_patients}")
    print(f"Patients without sepsis: {no_sepsis_patients}")

    print("\n6-HOUR PREDICTION")
    print("-" * 40)

    print(f"Sepsis patients with onset <= 6h: {early_6h}")
    print(f"Sepsis patients usable for 6h prediction: {patients_with_usable_6h}")

    if positive_hours_6h:

        print(f"Average positive prediction hours per patient: {sum(positive_hours_6h) / len(positive_hours_6h):.2f}")
        print(f"Median positive prediction hours per patient: {pd.Series(positive_hours_6h).median():.2f}")
        print(f"Minimum positive prediction hours: {min(positive_hours_6h)}")
        print(f"Maximum positive prediction hours: {max(positive_hours_6h)}")

        print("\nDistribution of positive prediction hours:")

        distribution_6h = pd.Series(positive_hours_6h).value_counts().sort_index()

        for hours, patients in distribution_6h.items():
            print(f"{int(hours)} positive hours: {patients} patients")

    print("\n12-HOUR PREDICTION")
    print("-" * 40)

    print(f"Sepsis patients with onset <= 12h: {early_12h}")
    print(f"Sepsis patients usable for 12h prediction: {patients_with_usable_12h}")

    if positive_hours_12h:

        print(f"Average positive prediction hours per patient: {sum(positive_hours_12h) / len(positive_hours_12h):.2f}")
        print(f"Median positive prediction hours per patient: {pd.Series(positive_hours_12h).median():.2f}")
        print(f"Minimum positive prediction hours: {min(positive_hours_12h)}")
        print(f"Maximum positive prediction hours: {max(positive_hours_12h)}")

        print("\nDistribution of positive prediction hours:")

        distribution_12h = pd.Series(positive_hours_12h).value_counts().sort_index()

        for hours, patients in distribution_12h.items():
            print(f"{int(hours)} positive hours: {patients} patients")

    # ======================================================
    # ONSET DISTRIBUTION
    # ======================================================

    print("\nFIRST SEPSIS ONSET DISTRIBUTION")
    print("-" * 40)

    onset_series = pd.Series(first_sepsis_hours)

    buckets = {
        "0-6 hours": ((onset_series >= 1) & (onset_series <= 6)).sum(),
        "7-12 hours": ((onset_series >= 7) & (onset_series <= 12)).sum(),
        "13-24 hours": ((onset_series >= 13) & (onset_series <= 24)).sum(),
        "25-48 hours": ((onset_series >= 25) & (onset_series <= 48)).sum(),
        "49-72 hours": ((onset_series >= 49) & (onset_series <= 72)).sum(),
        "73+ hours": (onset_series >= 73).sum()
    }

    for bucket, count in buckets.items():
        print(f"{bucket}: {count}")


# ==========================================================
# RUN FOR BOTH DATASETS
# ==========================================================

for dataset_name, folder in datasets.items():
    analyze_dataset(dataset_name, folder)

print("\n" + "=" * 70)
print("PATIENT-LEVEL WINDOW ANALYSIS COMPLETE")
print("=" * 70)