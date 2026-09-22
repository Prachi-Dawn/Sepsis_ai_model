import pandas as pd
import glob
import os

base_folder = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project\DataSets\Kaggle Data set"

folders = {
    "Dataset A": os.path.join(base_folder, "training_setA raw"),
    "Dataset B": os.path.join(base_folder, "training_setB raw")
}

overall_patients = 0
overall_sepsis_patients = 0
overall_rows = 0
overall_positive_rows = 0

summary = []

for dataset_name, folder in folders.items():

    files = glob.glob(os.path.join(folder, "*.psv"))

    total_patients = len(files)
    sepsis_patients = 0
    total_rows = 0
    positive_rows = 0

    summary.append(f"\n===== {dataset_name} =====")
    summary.append(f"Files found: {total_patients}")

    for i, file in enumerate(files):

        df = pd.read_csv(file, sep="|")

        total_rows += len(df)

        positives = df["SepsisLabel"].sum()
        positive_rows += positives

        if positives > 0:
            sepsis_patients += 1

        if (i + 1) % 2000 == 0:
            print(f"{dataset_name}: Processed {i + 1}/{total_patients}")

    summary.append(f"Patients: {total_patients}")
    summary.append(f"Patients with sepsis: {sepsis_patients}")
    summary.append(f"Patients without sepsis: {total_patients - sepsis_patients}")
    summary.append(f"Total rows: {total_rows}")
    summary.append(f"Positive rows: {int(positive_rows)}")

    overall_patients += total_patients
    overall_sepsis_patients += sepsis_patients
    overall_rows += total_rows
    overall_positive_rows += positive_rows


summary.append("\n===== COMBINED SUMMARY =====")
summary.append(f"Total patients: {overall_patients}")
summary.append(f"Patients with sepsis: {overall_sepsis_patients}")
summary.append(f"Patients without sepsis: {overall_patients - overall_sepsis_patients}")
summary.append(f"Total rows: {overall_rows}")
summary.append(f"Total positive rows: {int(overall_positive_rows)}")

summary_text = "\n".join(summary)

print(summary_text)

with open("dataset_summary.txt", "w") as file:
    file.write(summary_text)

print("\nSummary saved to dataset_summary.txt")