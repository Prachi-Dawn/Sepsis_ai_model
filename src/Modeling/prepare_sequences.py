
import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Results",
    "preprocessed"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Results",
    "sequences"
)

SEQUENCE_LENGTH = 12


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
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
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime"
]

MISSINGNESS_COLUMNS = [
    "HR_missing",
    "O2Sat_missing",
    "Temp_missing",
    "SBP_missing",
    "MAP_missing",
    "DBP_missing",
    "Resp_missing",
    "EtCO2_missing",
    "BaseExcess_missing",
    "HCO3_missing",
    "FiO2_missing",
    "pH_missing",
    "PaCO2_missing",
    "SaO2_missing",
    "AST_missing",
    "BUN_missing",
    "Alkalinephos_missing",
    "Calcium_missing",
    "Chloride_missing",
    "Creatinine_missing",
    "Bilirubin_direct_missing",
    "Glucose_missing",
    "Lactate_missing",
    "Magnesium_missing",
    "Phosphate_missing",
    "Potassium_missing",
    "Bilirubin_total_missing",
    "TroponinI_missing",
    "Hct_missing",
    "Hgb_missing",
    "PTT_missing",
    "WBC_missing",
    "Fibrinogen_missing",
    "Platelets_missing"
]

FEATURE_COLUMNS = FEATURE_COLUMNS + MISSINGNESS_COLUMNS

TARGET_COLUMN = "Sepsis_12h"


# ============================================================
# CREATE ONE SPLIT
# ============================================================

def prepare_split(split_name):

    input_file = os.path.join(
        INPUT_FOLDER,
        f"{split_name}_preprocessed.csv"
    )

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"{split_name}_sequences.npz"
    )

    print("\n" + "=" * 70)
    print(f"PREPARING {split_name.upper()} SEQUENCES")
    print("=" * 70)

    print(f"Loading:")
    print(input_file)

    df = pd.read_csv(input_file)

    print(f"Rows loaded: {len(df):,}")
    print(f"Patients: {df['Patient_ID'].nunique():,}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "Patient_ID",
        "ICULOS",
        TARGET_COLUMN
    ] + FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Sort by patient and ICU hour
    # --------------------------------------------------------

    df = df.sort_values(
        ["Patient_ID", "ICULOS"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    sequences = []
    labels = []

    total_possible_windows = 0
    valid_windows = 0
    skipped_windows = 0

    positive_sequences = 0
    negative_sequences = 0

    # --------------------------------------------------------
    # Process each patient separately
    # --------------------------------------------------------

    grouped = df.groupby(
        "Patient_ID",
        sort=False
    )

    for patient_id, patient_df in grouped:

        patient_df = patient_df.sort_values("ICULOS")

        patient_hours = patient_df["ICULOS"].to_numpy()

        if len(patient_df) < SEQUENCE_LENGTH:
            continue

        # ----------------------------------------------------
        # Sliding 12-hour windows
        # ----------------------------------------------------

        for start in range(
            0,
            len(patient_df) - SEQUENCE_LENGTH + 1
        ):

            end = start + SEQUENCE_LENGTH

            total_possible_windows += 1

            window = patient_df.iloc[start:end]

            hours = window["ICULOS"].to_numpy()

            # ------------------------------------------------
            # Require consecutive ICU hours
            # ------------------------------------------------

            expected_hours = np.arange(
                hours[0],
                hours[0] + SEQUENCE_LENGTH
            )

            if not np.array_equal(
                hours,
                expected_hours
            ):
                skipped_windows += 1
                continue

            # ------------------------------------------------
            # Extract features
            # ------------------------------------------------

            X_window = window[FEATURE_COLUMNS].copy()
            X_window["Unit1"] = X_window["Unit1"].fillna(0)
            X_window["Unit2"] = X_window["Unit2"].fillna(0)

            if X_window[["Unit1", "Unit2"]].isna().any().any():
                print("ERROR: Unit1/Unit2 still contain NaN after fillna")
                print(X_window[["Unit1", "Unit2"]].isna().sum())
                raise ValueError("Unit1/Unit2 NaN remains")

            X_window = X_window.to_numpy(
                dtype=np.float32
            )

            # ------------------------------------------------
            # Extract target at final hour
            # ------------------------------------------------

            y_value = window[
                TARGET_COLUMN
            ].iloc[-1]

            y_value = int(y_value)

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if X_window.shape != (
                SEQUENCE_LENGTH,
                len(FEATURE_COLUMNS)
            ):
                raise ValueError(
                    f"Unexpected sequence shape for "
                    f"{patient_id}: {X_window.shape}"
                )

            sequences.append(X_window)
            labels.append(y_value)

            valid_windows += 1

            if y_value == 1:
                positive_sequences += 1
            else:
                negative_sequences += 1

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    if len(sequences) == 0:
        raise ValueError(
            f"No valid sequences were created for {split_name}."
        )

    X = np.asarray(
        sequences,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    positive_percentage = (
        positive_sequences / valid_windows * 100
    )

    negative_percentage = (
        negative_sequences / valid_windows * 100
    )

    print("\nSequence preparation complete.")

    print("\nStatistics:")
    print(f"Patients:              {df['Patient_ID'].nunique():,}")
    print(f"Rows:                  {len(df):,}")
    print(f"Features per hour:     {len(FEATURE_COLUMNS)}")
    print(f"Sequence length:       {SEQUENCE_LENGTH}")
    print(f"Possible windows:      {total_possible_windows:,}")
    print(f"Valid windows:         {valid_windows:,}")
    print(f"Skipped windows:       {skipped_windows:,}")

    print("\nTarget distribution:")
    print(
        f"Negative sequences:    "
        f"{negative_sequences:,} "
        f"({negative_percentage:.2f}%)"
    )

    print(
        f"Positive sequences:    "
        f"{positive_sequences:,} "
        f"({positive_percentage:.2f}%)"
    )

    print("\nArray shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nSaved to:")
    print(output_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("12-HOUR SEPSIS SEQUENCE PREPARATION")
    print("=" * 70)

    prepare_split("train")
    prepare_split("validation")
    prepare_split("test")

    print("\n" + "=" * 70)
    print("ALL SEQUENCE FILES CREATED")
    print("=" * 70)

