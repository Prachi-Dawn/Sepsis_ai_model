import pandas as pd
import glob
import os
import statistics

base_folder = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project\DataSets\Kaggle Data set"

folders = {
    "Dataset A": os.path.join(base_folder, "training_setA raw"),
    "Dataset B": os.path.join(base_folder, "training_setB raw")
}

summary = []

all_pre_sepsis_hours = []
all_first_sepsis_hours = []

for dataset_name, folder in folders.items():

    files = glob.glob(os.path.join(folder, "*.psv"))

    first_sepsis_hours = []
    pre_sepsis_hours = []

    print(f"\nProcessing {dataset_name}...")

    for i, file in enumerate(files):

        df = pd.read_csv(file, sep="|")

        positive_rows = df[df["SepsisLabel"] == 1]

        if len(positive_rows) > 0:

            first_hour = positive_rows["ICULOS"].min()

            # Assuming ICULOS starts at 1
            pre_hours = first_hour - 1

            first_sepsis_hours.append(first_hour)
            pre_sepsis_hours.append(pre_hours)

            all_first_sepsis_hours.append(first_hour)
            all_pre_sepsis_hours.append(pre_hours)

        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1}/{len(files)}")

    summary.append(f"\n===== {dataset_name} =====")

    summary.append(
        f"Sepsis patients: {len(first_sepsis_hours)}"
    )

    summary.append(
        f"Pre-sepsis hours - minimum: {min(pre_sepsis_hours)}"
    )

    summary.append(
        f"Pre-sepsis hours - maximum: {max(pre_sepsis_hours)}"
    )

    summary.append(
        f"Pre-sepsis hours - average: {statistics.mean(pre_sepsis_hours):.2f}"
    )

    summary.append(
        f"Pre-sepsis hours - median: {statistics.median(pre_sepsis_hours):.2f}"
    )

    # Onset buckets
    buckets = {
        "0-6 hours": 0,
        "7-12 hours": 0,
        "13-24 hours": 0,
        "25-48 hours": 0,
        "49-72 hours": 0,
        "73+ hours": 0
    }

    for hour in first_sepsis_hours:

        if hour <= 6:
            buckets["0-6 hours"] += 1
        elif hour <= 12:
            buckets["7-12 hours"] += 1
        elif hour <= 24:
            buckets["13-24 hours"] += 1
        elif hour <= 48:
            buckets["25-48 hours"] += 1
        elif hour <= 72:
            buckets["49-72 hours"] += 1
        else:
            buckets["73+ hours"] += 1

    summary.append("\nFirst sepsis hour distribution:")

    for bucket, count in buckets.items():
        summary.append(
            f"{bucket}: {count}"
        )


summary.append("\n===== COMBINED =====")

summary.append(
    f"Total sepsis patients: {len(all_first_sepsis_hours)}"
)

summary.append(
    f"Pre-sepsis hours - minimum: {min(all_pre_sepsis_hours)}"
)

summary.append(
    f"Pre-sepsis hours - maximum: {max(all_pre_sepsis_hours)}"
)

summary.append(
    f"Pre-sepsis hours - average: {statistics.mean(all_pre_sepsis_hours):.2f}"
)

summary.append(
    f"Pre-sepsis hours - median: {statistics.median(all_pre_sepsis_hours):.2f}"
)

summary_text = "\n".join(summary)

print(summary_text)

with open("pre_sepsis_analysis.txt", "w") as file:
    file.write(summary_text)

print("\nAnalysis saved to pre_sepsis_analysis.txt")