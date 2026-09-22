import pandas as pd
import glob
import os
import statistics

base_folder = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project\DataSets\Kaggle Data set"

folders = {
    "Dataset A": os.path.join(base_folder, "training_setA raw"),
    "Dataset B": os.path.join(base_folder, "training_setB raw")
}

all_first_sepsis_hours = []
all_sepsis_durations = []

summary = []

for dataset_name, folder in folders.items():

    files = glob.glob(os.path.join(folder, "*.psv"))

    first_sepsis_hours = []
    sepsis_durations = []
    patients_with_sepsis = 0

    total_patients = len(files)

    print(f"\nProcessing {dataset_name}...")

    for i, file in enumerate(files):

        df = pd.read_csv(file, sep="|")

        positive_rows = df[df["SepsisLabel"] == 1]

        if len(positive_rows) > 0:

            patients_with_sepsis += 1

            first_hour = positive_rows["ICULOS"].iloc[0]
            duration = len(positive_rows)

            first_sepsis_hours.append(first_hour)
            sepsis_durations.append(duration)

            all_first_sepsis_hours.append(first_hour)
            all_sepsis_durations.append(duration)

        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1}/{total_patients}")

    summary.append(f"\n===== {dataset_name} =====")

    summary.append(f"Total patients: {total_patients}")
    summary.append(f"Patients with sepsis: {patients_with_sepsis}")

    if patients_with_sepsis > 0:

        summary.append(
            f"First sepsis hour - minimum: {min(first_sepsis_hours):.2f}"
        )

        summary.append(
            f"First sepsis hour - maximum: {max(first_sepsis_hours):.2f}"
        )

        summary.append(
            f"First sepsis hour - average: {statistics.mean(first_sepsis_hours):.2f}"
        )

        summary.append(
            f"First sepsis hour - median: {statistics.median(first_sepsis_hours):.2f}"
        )

        summary.append(
            f"Positive hours per patient - minimum: {min(sepsis_durations)}"
        )

        summary.append(
            f"Positive hours per patient - maximum: {max(sepsis_durations)}"
        )

        summary.append(
            f"Positive hours per patient - average: {statistics.mean(sepsis_durations):.2f}"
        )

        summary.append(
            f"Positive hours per patient - median: {statistics.median(sepsis_durations):.2f}"
        )


summary.append("\n===== COMBINED TIMING SUMMARY =====")

summary.append(
    f"Total patients with sepsis: {len(all_first_sepsis_hours)}"
)

summary.append(
    f"First sepsis hour - minimum: {min(all_first_sepsis_hours):.2f}"
)

summary.append(
    f"First sepsis hour - maximum: {max(all_first_sepsis_hours):.2f}"
)

summary.append(
    f"First sepsis hour - average: {statistics.mean(all_first_sepsis_hours):.2f}"
)

summary.append(
    f"First sepsis hour - median: {statistics.median(all_first_sepsis_hours):.2f}"
)

summary.append(
    f"Positive hours per patient - minimum: {min(all_sepsis_durations)}"
)

summary.append(
    f"Positive hours per patient - maximum: {max(all_sepsis_durations)}"
)

summary.append(
    f"Positive hours per patient - average: {statistics.mean(all_sepsis_durations):.2f}"
)

summary.append(
    f"Positive hours per patient - median: {statistics.median(all_sepsis_durations):.2f}"
)

summary_text = "\n".join(summary)

print(summary_text)

with open("sepsis_timing_summary.txt", "w") as file:
    file.write(summary_text)

print("\nTiming analysis saved to sepsis_timing_summary.txt")