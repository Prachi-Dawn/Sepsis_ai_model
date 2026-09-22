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

    current_positive = 0
    current_negative = 0

    results = {
        6: {"positive": 0, "negative": 0, "usable_patients": 0, "early_excluded": 0},
        12: {"positive": 0, "negative": 0, "usable_patients": 0, "early_excluded": 0}
    }

    for i, file in enumerate(files):
        df = pd.read_csv(file, sep="|")

        sepsis_rows = df[df["SepsisLabel"] == 1]

        if len(sepsis_rows) == 0:
            no_sepsis_patients += 1
            first_sepsis_hour = None
        else:
            sepsis_patients += 1
            first_sepsis_hour = sepsis_rows["ICULOS"].min()

        # Current-state detection
        current_positive += (df["SepsisLabel"] == 1).sum()
        current_negative += (df["SepsisLabel"] == 0).sum()

        # Future prediction targets
        for horizon in HORIZONS:

            if first_sepsis_hour is not None and first_sepsis_hour <= horizon:
                results[horizon]["early_excluded"] += 1
                continue

            results[horizon]["usable_patients"] += 1

            for _, row in df.iterrows():

                t = row["ICULOS"]

                # Do not use the actual sepsis hour as a future prediction example
                if first_sepsis_hour is not None and t >= first_sepsis_hour:
                    continue

                if first_sepsis_hour is not None:
                    future_sepsis = (
                        first_sepsis_hour > t
                        and first_sepsis_hour <= t + horizon
                    )
                else:
                    future_sepsis = False

                if future_sepsis:
                    results[horizon]["positive"] += 1
                else:
                    results[horizon]["negative"] += 1

        if (i + 1) % 2000 == 0:
            print(f"{dataset_name}: processed {i + 1}/{total_patients}")

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(f"Total patients: {total_patients}")
    print(f"Patients with sepsis: {sepsis_patients}")
    print(f"Patients without sepsis: {no_sepsis_patients}")

    print("\nCurrent-state detection:")
    print(f"Positive rows: {current_positive}")
    print(f"Negative rows: {current_negative}")

    for horizon in HORIZONS:
        r = results[horizon]

        total_examples = r["positive"] + r["negative"]

        print(f"\n{horizon}-hour early prediction:")
        print(f"Usable patients: {r['usable_patients']}")
        print(f"Patients excluded because sepsis began within first {horizon} hours: {r['early_excluded']}")
        print(f"Positive examples: {r['positive']}")
        print(f"Negative examples: {r['negative']}")
        print(f"Total examples: {total_examples}")

        if total_examples > 0:
            ratio = r["positive"] / total_examples
            print(f"Positive percentage: {ratio * 100:.2f}%")

        if r["negative"] > 0:
            imbalance = r["negative"] / r["positive"] if r["positive"] > 0 else 0
            print(f"Negative : Positive ratio: {imbalance:.2f} : 1")


all_results = []

for dataset_name, folder in datasets.items():
    analyze_dataset(dataset_name, folder)

print("\n" + "=" * 60)
print("TARGET ANALYSIS COMPLETE")
print("=" * 60)