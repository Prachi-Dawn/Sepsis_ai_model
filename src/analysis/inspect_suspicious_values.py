import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\KIIT0001\Desktop\Cse\7th Sem\Sepsis\sepsis Project"

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "ml_dataset",
    "train.csv"
)

SUSPICIOUS_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "data_quality",
    "suspicious_values.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "data_quality",
    "suspicious_neighbors.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

suspicious = pd.read_csv(SUSPICIOUS_FILE)

print(f"Training rows: {len(df)}")
print(f"Suspicious observations: {len(suspicious)}")


# ============================================================
# FEATURES TO INSPECT
# ============================================================

features = [
    "MAP",
    "SBP",
    "DBP",
    "Resp",
    "FiO2",
    "Temp",
    "BaseExcess",
    "Chloride"
]


# ============================================================
# FIND NEIGHBORING VALUES
# ============================================================

results = []

for _, row in suspicious.iterrows():

    feature = row["Feature"]
    patient_id = row["Patient_ID"]
    iculos = row["ICULOS"]

    if feature not in features:
        continue

    patient_data = df[
        df["Patient_ID"] == patient_id
    ].sort_values("ICULOS")

    neighbors = patient_data[
        patient_data["ICULOS"].between(
            iculos - 2,
            iculos + 2
        )
    ]

    for _, neighbor in neighbors.iterrows():

        results.append({
            "Feature": feature,
            "Patient_ID": patient_id,
            "Suspicious_ICULOS": iculos,
            "Suspicious_Value": row["Value"],
            "Neighbor_ICULOS": neighbor["ICULOS"],
            "Neighbor_Value": neighbor[feature],
            "Sepsis_12h": neighbor["Sepsis_12h"]
        })


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(results)

result = result.sort_values(
    [
        "Feature",
        "Patient_ID",
        "Suspicious_ICULOS",
        "Neighbor_ICULOS"
    ]
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 70)
print("SUSPICIOUS VALUES WITH NEIGHBORING HOURS")
print("=" * 70)

print(
    result.to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nInspection completed.")