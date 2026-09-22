import pandas as pd
import glob
import os
import numpy as np

base_folder = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project\DataSets\Kaggle Data set"

folders = {
    "Dataset A": os.path.join(base_folder, "training_setA raw"),
    "Dataset B": os.path.join(base_folder, "training_setB raw")
}

# Clinical variables we want to analyze
variables = [
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
    "Platelets"
]

# Store results from every sepsis patient
all_records = []

for dataset_name, folder in folders.items():

    files = glob.glob(os.path.join(folder, "*.psv"))

    print(f"\nProcessing {dataset_name}...")
    print(f"Total files: {len(files)}")

    for i, file in enumerate(files):

        df = pd.read_csv(file, sep="|")

        positive_rows = df[df["SepsisLabel"] == 1]

        if len(positive_rows) == 0:
            continue

        # First hour with SepsisLabel = 1
        first_sepsis_hour = positive_rows["ICULOS"].min()

        # Examine the 24 hours before first positive label
        start_hour = max(1, first_sepsis_hour - 24)
        end_hour = first_sepsis_hour - 1

        if end_hour < start_hour:
            continue

        pre_window = df[
            (df["ICULOS"] >= start_hour) &
            (df["ICULOS"] <= end_hour)
        ].copy()

        if len(pre_window) == 0:
            continue

        # Convert each hour into relative time before sepsis
        # -1 = one hour before
        # -24 = 24 hours before
        pre_window["HoursBeforeSepsis"] = (
            pre_window["ICULOS"] - first_sepsis_hour
        )

        pre_window["Dataset"] = dataset_name

        # Patient ID from filename
        pre_window["PatientID"] = os.path.splitext(
            os.path.basename(file)
        )[0]

        all_records.append(
            pre_window[
                ["PatientID", "Dataset", "ICULOS", "HoursBeforeSepsis"]
                + variables
            ]
        )

    print(f"Processed {i + 1}/{len(files)}")


# Combine all patient windows
combined = pd.concat(all_records, ignore_index=True)

print("\n====================================")
print("24-HOUR PRE-SEPSIS ANALYSIS")
print("====================================")

print(
    f"Patients included: {combined['PatientID'].nunique()}"
)

print(
    f"Total pre-sepsis observations: {len(combined)}"
)


# ---------------------------------------------------------
# 1. Overall analysis
# ---------------------------------------------------------

overall_results = []

for hour in range(-24, 0):

    subset = combined[
        combined["HoursBeforeSepsis"] == hour
    ]

    row = {
        "HoursBeforeSepsis": hour,
        "PatientObservations": len(subset),
        "UniquePatients": subset["PatientID"].nunique()
    }

    for variable in variables:

        values = pd.to_numeric(
            subset[variable],
            errors="coerce"
        )

        row[f"{variable}_Mean"] = values.mean()
        row[f"{variable}_Median"] = values.median()
        row[f"{variable}_MissingPct"] = (
            values.isna().mean() * 100
        )

    overall_results.append(row)


overall_df = pd.DataFrame(overall_results)


# ---------------------------------------------------------
# 2. Dataset-specific analysis
# ---------------------------------------------------------

dataset_results = []

for dataset_name in folders.keys():

    dataset_data = combined[
        combined["Dataset"] == dataset_name
    ]

    for hour in range(-24, 0):

        subset = dataset_data[
            dataset_data["HoursBeforeSepsis"] == hour
        ]

        row = {
            "Dataset": dataset_name,
            "HoursBeforeSepsis": hour,
            "PatientObservations": len(subset),
            "UniquePatients": subset["PatientID"].nunique()
        }

        for variable in variables:

            values = pd.to_numeric(
                subset[variable],
                errors="coerce"
            )

            row[f"{variable}_Mean"] = values.mean()
            row[f"{variable}_Median"] = values.median()
            row[f"{variable}_MissingPct"] = (
                values.isna().mean() * 100
            )

        dataset_results.append(row)


dataset_df = pd.DataFrame(dataset_results)


# ---------------------------------------------------------
# 3. Compare early vs late pre-sepsis period
# ---------------------------------------------------------

comparison_results = []

for variable in variables:

    early = combined[
        combined["HoursBeforeSepsis"].between(-24, -13)
    ][variable]

    late = combined[
        combined["HoursBeforeSepsis"].between(-12, -1)
    ][variable]

    early = pd.to_numeric(early, errors="coerce")
    late = pd.to_numeric(late, errors="coerce")

    comparison_results.append({
        "Variable": variable,

        "Early_24_to_13h_Mean": early.mean(),
        "Late_12_to_1h_Mean": late.mean(),

        "Mean_Change": late.mean() - early.mean(),

        "Early_MissingPct": early.isna().mean() * 100,
        "Late_MissingPct": late.isna().mean() * 100
    })


comparison_df = pd.DataFrame(comparison_results)


# ---------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------

overall_df.to_csv(
    "24h_pre_sepsis_overall.csv",
    index=False
)

dataset_df.to_csv(
    "24h_pre_sepsis_by_dataset.csv",
    index=False
)

comparison_df.to_csv(
    "24h_pre_sepsis_early_vs_late.csv",
    index=False
)


# ---------------------------------------------------------
# 5. Create readable text summary
# ---------------------------------------------------------

summary = []

summary.append("========================================")
summary.append("24-HOUR PRE-SEPSIS ANALYSIS")
summary.append("========================================")

summary.append(
    f"Patients with usable pre-sepsis data: "
    f"{combined['PatientID'].nunique()}"
)

summary.append(
    f"Total observations analyzed: {len(combined)}"
)

summary.append(
    "\nRelative hours:"
)

summary.append(
    "-24 = 24 hours before first positive label"
)

summary.append(
    "-1 = 1 hour before first positive label"
)

summary.append(
    "\nPatients contributing at each hour:"
)

for hour in range(-24, 0):

    count = combined[
        combined["HoursBeforeSepsis"] == hour
    ]["PatientID"].nunique()

    summary.append(
        f"{hour} hours: {count} patients"
    )


summary.append(
    "\nEarly vs late pre-sepsis comparison:"
)

for _, row in comparison_df.iterrows():

    summary.append(
        f"\n{row['Variable']}"
    )

    summary.append(
        f"  24–13h mean: "
        f"{row['Early_24_to_13h_Mean']:.3f}"
    )

    summary.append(
        f"  12–1h mean: "
        f"{row['Late_12_to_1h_Mean']:.3f}"
    )

    summary.append(
        f"  Mean change: "
        f"{row['Mean_Change']:.3f}"
    )

    summary.append(
        f"  Early missing: "
        f"{row['Early_MissingPct']:.2f}%"
    )

    summary.append(
        f"  Late missing: "
        f"{row['Late_MissingPct']:.2f}%"
    )


summary_text = "\n".join(summary)

print("\n" + summary_text)

with open(
    "24h_pre_sepsis_summary.txt",
    "w"
) as file:

    file.write(summary_text)


print(
    "\nResults saved:"
)

print("24h_pre_sepsis_overall.csv")
print("24h_pre_sepsis_by_dataset.csv")
print("24h_pre_sepsis_early_vs_late.csv")
print("24h_pre_sepsis_summary.txt")