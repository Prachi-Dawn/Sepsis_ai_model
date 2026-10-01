import os
import numpy as np


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Results",
    "sequences"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Results",
    "training_arrays"
)


def convert_split(split_name):

    input_file = os.path.join(
        INPUT_FOLDER,
        f"{split_name}_sequences.npz"
    )

    x_output = os.path.join(
        OUTPUT_FOLDER,
        f"{split_name}_X.npy"
    )

    y_output = os.path.join(
        OUTPUT_FOLDER,
        f"{split_name}_y.npy"
    )

    print("\n" + "=" * 70)
    print(f"CONVERTING {split_name.upper()}")
    print("=" * 70)

    data = np.load(input_file)

    X = data["X"]
    y = data["y"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save(x_output, X)
    np.save(y_output, y)

    print("\nSaved:")
    print(x_output)
    print(y_output)


if __name__ == "__main__":

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    convert_split("train")
    convert_split("validation")
    convert_split("test")

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)