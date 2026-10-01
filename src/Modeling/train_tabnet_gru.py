import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Results",
    "training_arrays"
)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Results",
    "models"
)

DEVICE = torch.device("cpu")

INPUT_FEATURES = 73
SEQUENCE_LENGTH = 12

TABNET_DIM = 128
GRU_HIDDEN = 128

DROPOUT = 0.3

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

EPOCHS = 3
PATIENCE = 2

NUM_WORKERS = 0

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# DATASET
# ============================================================

class SepsisDataset(Dataset):

    def __init__(self, x_path, y_path):

        self.X = np.load(
            x_path,
            mmap_mode="r"
        )

        self.y = np.load(
            y_path,
            mmap_mode="r"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = np.asarray(
            self.X[index],
            dtype=np.float32
        )

        y = np.float32(
            self.y[index]
        )

        return (
            torch.from_numpy(x),
            torch.tensor(y)
        )


# ============================================================
# TABNET-STYLE ENCODER
# ============================================================

class TabNetStyleEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim
    ):

        super().__init__()

        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        attention_mask = self.attention(x)

        x_selected = x * attention_mask

        embedding = self.feature_transform(
            x_selected
        )

        return embedding


# ============================================================
# TABNET-GRU MODEL
# ============================================================

class TabNetGRU(nn.Module):

    def __init__(
        self,
        input_dim=73,
        tabnet_dim=128,
        gru_hidden=128,
        dropout=0.3
    ):

        super().__init__()

        self.tabnet = TabNetStyleEncoder(
            input_dim=input_dim,
            embedding_dim=tabnet_dim
        )

        self.gru = nn.GRU(
            input_size=tabnet_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        batch_size, sequence_length, feature_count = x.shape

        x = x.reshape(
            batch_size * sequence_length,
            feature_count
        )

        embeddings = self.tabnet(x)

        embeddings = embeddings.reshape(
            batch_size,
            sequence_length,
            -1
        )

        gru_output, _ = self.gru(
            embeddings
        )

        final_output = gru_output[:, -1, :]

        logits = self.classifier(
            final_output
        )

        return logits.squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    auprc = average_precision_score(
        y_true,
        probabilities
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "AUROC": roc_auc,
        "AUPRC": auprc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0
    probabilities = []
    targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item() * len(y)
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

            targets.extend(
                y.cpu().numpy()
            )

    probabilities = np.asarray(
        probabilities
    )

    targets = np.asarray(
        targets
    )

    metrics = calculate_metrics(
        targets,
        probabilities
    )

    metrics["Loss"] = (
        total_loss / len(targets)
    )

    return metrics


# ============================================================
# TRAINING
# ============================================================

def train():

    print("=" * 70)
    print("TABNET-GRU SEPSIS MODEL")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------
    print("\nPROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_FOLDER:", DATA_FOLDER)
    print("train_X path:", os.path.join(DATA_FOLDER, "train_X.npy"))
    print("train_X exists:", os.path.exists(os.path.join(DATA_FOLDER, "train_X.npy")))
    print("train_y path:", os.path.join(DATA_FOLDER, "train_y.npy"))
    print("train_y exists:", os.path.exists(os.path.join(DATA_FOLDER, "train_y.npy")))
    train_dataset = SepsisDataset(
        os.path.join(
            DATA_FOLDER,
            "train_X.npy"
        ),
        os.path.join(
            DATA_FOLDER,
            "train_y.npy"
        )
    )

    validation_dataset = SepsisDataset(
        os.path.join(
            DATA_FOLDER,
            "validation_X.npy"
        ),
        os.path.join(
            DATA_FOLDER,
            "validation_y.npy"
        )
    )

    print(
        "\nTraining samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Calculate class weight
    # --------------------------------------------------------

    train_y = np.load(
        os.path.join(
            DATA_FOLDER,
            "train_y.npy"
        ),
        mmap_mode="r"
    )

    positive_count = np.sum(
        train_y == 1
    )

    negative_count = np.sum(
        train_y == 0
    )

    positive_weight = (
        negative_count / positive_count
    )

    print("\nClass distribution:")
    print(
        "Negative:",
        f"{negative_count:,}"
    )

    print(
        "Positive:",
        f"{positive_count:,}"
    )

    print(
        "Positive class weight:",
        f"{positive_weight:.2f}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TabNetGRU(
        input_dim=INPUT_FEATURES,
        tabnet_dim=TABNET_DIM,
        gru_hidden=GRU_HIDDEN,
        dropout=DROPOUT
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    pos_weight = torch.tensor(
        [positive_weight],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_auprc = -1
    patience_counter = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0

        for batch_index, (X, y) in enumerate(
            train_loader,
            start=1
        ):

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * len(y)
            )

            if batch_index % 500 == 0:

                print(
                    f"Epoch {epoch} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = (
            running_loss /
            len(train_dataset)
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion
        )

        print("\n" + "-" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_metrics['Loss']:.4f}"
        )

        print(
            f"AUROC: "
            f"{validation_metrics['AUROC']:.4f}"
        )

        print(
            f"AUPRC: "
            f"{validation_metrics['AUPRC']:.4f}"
        )

        print(
            f"Precision: "
            f"{validation_metrics['Precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{validation_metrics['Recall']:.4f}"
        )

        print(
            f"F1: "
            f"{validation_metrics['F1']:.4f}"
        )

        print(
            f"Specificity: "
            f"{validation_metrics['Specificity']:.4f}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_metrics["AUPRC"] > best_auprc:

            best_auprc = validation_metrics[
                "AUPRC"
            ]

            patience_counter = 0

            model_path = os.path.join(
                MODEL_FOLDER,
                "best_tabnet_gru.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_features": INPUT_FEATURES,
                    "sequence_length": SEQUENCE_LENGTH,
                    "tabnet_dim": TABNET_DIM,
                    "gru_hidden": GRU_HIDDEN,
                    "dropout": DROPOUT
                },
                model_path
            )

            print(
                "\nBest model saved."
            )

        else:

            patience_counter += 1

            print(
                f"\nNo AUPRC improvement. "
                f"Patience: {patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping."
            )

            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    train()

