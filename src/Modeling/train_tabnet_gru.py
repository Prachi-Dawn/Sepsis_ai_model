"""
TabNet-GRU Hybrid Model for Sepsis Prediction
- Multi-step sparse attention TabNet encoder (3 decision steps)
- Unidirectional GRU temporal module
- Focal loss + moderate class weighting
- Training with proper early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import time
from datetime import datetime


# ============================================================
# SPARSEMAX ACTIVATION (replaces Softmax for sparse attention)
# ============================================================
class Sparsemax(nn.Module):
    """Sparsemax activation function (Martins & Astudillo, 2016)."""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        sorted_x, _ = torch.sort(x, descending=True, dim=self.dim)
        cumsum = torch.cumsum(sorted_x, dim=self.dim)
        k = torch.arange(1, x.size(self.dim) + 1, device=x.device, dtype=x.dtype)

        # Reshape k for broadcasting
        shape = [1] * x.dim()
        shape[self.dim] = -1
        k = k.reshape(shape)

        support = (sorted_x - (cumsum - 1) / k) > 0
        k_max = support.sum(dim=self.dim, keepdim=True).float()
        tau = (cumsum.gather(self.dim, (k_max - 1).long().clamp(min=0)) - 1) / k_max

        output = torch.clamp(x - tau, min=0)
        return output


# ============================================================
# GHOST BATCH NORMALIZATION
# ============================================================
class GhostBatchNorm(nn.Module):
    """Ghost Batch Normalization for TabNet."""
    def __init__(self, n_features, virtual_batch_size=64, momentum=0.02):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features, momentum=momentum)
        self.virtual_batch_size = virtual_batch_size

    def forward(self, x):
        if not self.training or x.size(0) <= self.virtual_batch_size:
            return self.bn(x)

        chunks = x.chunk(
            max(1, x.size(0) // self.virtual_batch_size), dim=0
        )
        result = [self.bn(chunk) for chunk in chunks]
        return torch.cat(result, dim=0)


# ============================================================
# TABNET FEATURE TRANSFORMER
# ============================================================
class FeatureTransformer(nn.Module):
    """Shared + step-specific feature transformer block."""
    def __init__(self, input_dim, output_dim, shared_layers=None, virtual_batch_size=64):
        super().__init__()

        # Shared layers (across all steps)
        if shared_layers is not None:
            self.shared_fc = shared_layers
        else:
            self.shared_fc = nn.Linear(input_dim, output_dim, bias=False)

        # Step-specific layers
        self.step_fc = nn.Linear(output_dim, output_dim, bias=False)
        self.bn1 = GhostBatchNorm(output_dim, virtual_batch_size)
        self.bn2 = GhostBatchNorm(output_dim, virtual_batch_size)

    def forward(self, x):
        x = self.shared_fc(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.step_fc(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


# ============================================================
# MULTI-STEP TABNET ENCODER
# ============================================================
class TabNetEncoder(nn.Module):
    """
    Multi-step TabNet encoder with sparse attention.
    Implements N_steps decision steps with complementary attention masks.
    """
    def __init__(self, input_dim, embed_dim=128, n_steps=3,
                 relaxation_factor=1.5, virtual_batch_size=64,
                 sparsity_coefficient=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_steps = n_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient

        # Initial batch normalization
        self.initial_bn = nn.BatchNorm1d(input_dim)

        # Shared feature transformer layer (shared across steps)
        self.shared_fc = nn.Linear(input_dim, embed_dim, bias=False)

        # Step-specific components
        self.feature_transformers = nn.ModuleList()
        self.attention_transformers = nn.ModuleList()

        for step in range(n_steps):
            # Feature transformer for each step
            self.feature_transformers.append(
                FeatureTransformer(
                    input_dim, embed_dim,
                    shared_layers=self.shared_fc,
                    virtual_batch_size=virtual_batch_size
                )
            )

            # Attention transformer for each step
            self.attention_transformers.append(nn.Sequential(
                nn.Linear(embed_dim, input_dim, bias=False),
                GhostBatchNorm(input_dim, virtual_batch_size)
            ))

        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.initial_bn(x)

        # Initialize
        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)
        aggregated_output = torch.zeros(batch_size, self.embed_dim, device=x.device)
        entropy_loss = 0.0

        # Compute initial features for attention
        h = self.feature_transformers[0](x)

        for step in range(self.n_steps):
            # Attention mask
            attention_input = self.attention_transformers[step](h)
            attention_input = attention_input * prior_scales
            attention_mask = self.sparsemax(attention_input)

            # Update prior scales (complementary attention)
            prior_scales = prior_scales * (self.relaxation_factor - attention_mask)

            # Entropy loss for sparsity regularization
            entropy_loss += torch.mean(
                torch.sum(-attention_mask * torch.log(attention_mask + 1e-15), dim=-1)
            )

            # Masked features
            masked_x = attention_mask * x

            # Feature transformation
            h = self.feature_transformers[step](masked_x)

            # Aggregate
            aggregated_output = aggregated_output + h

        # Average over steps
        aggregated_output = aggregated_output / self.n_steps

        # Scale entropy loss
        entropy_loss = self.sparsity_coefficient * entropy_loss / self.n_steps

        return aggregated_output, entropy_loss


# ============================================================
# COMPLETE TabNet-GRU MODEL
# ============================================================
class TabNetGRU(nn.Module):
    """
    Hybrid TabNet-GRU for sequential sepsis prediction.
    TabNet encoder processes each timestep -> GRU captures temporal patterns.
    """
    def __init__(self, input_dim=73, tabnet_dim=128, n_steps=3,
                 gru_hidden=128, gru_layers=1, dropout=0.3,
                 relaxation_factor=1.5, virtual_batch_size=64,
                 sparsity_coefficient=1e-3):
        super().__init__()

        self.tabnet_encoder = TabNetEncoder(
            input_dim=input_dim,
            embed_dim=tabnet_dim,
            n_steps=n_steps,
            relaxation_factor=relaxation_factor,
            virtual_batch_size=virtual_batch_size,
            sparsity_coefficient=sparsity_coefficient
        )

        self.gru = nn.GRU(
            input_size=tabnet_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, features)
        """
        batch_size, seq_len, n_features = x.shape

        # Process each timestep through TabNet encoder
        tabnet_outputs = []
        total_entropy_loss = 0.0

        for t in range(seq_len):
            timestep_data = x[:, t, :]  # (batch, features)
            encoded, entropy = self.tabnet_encoder(timestep_data)
            tabnet_outputs.append(encoded)
            total_entropy_loss += entropy

        total_entropy_loss = total_entropy_loss / seq_len

        # Stack: (batch, seq_len, tabnet_dim)
        tabnet_sequence = torch.stack(tabnet_outputs, dim=1)

        # GRU temporal processing
        gru_out, _ = self.gru(tabnet_sequence)

        # Use last timestep output
        last_hidden = gru_out[:, -1, :]  # (batch, gru_hidden)

        # Classification
        logit = self.classifier(last_hidden)  # (batch, 1)

        return logit, total_entropy_loss


# ============================================================
# FOCAL LOSS (better for class imbalance than weighted BCE)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: focuses learning on hard-to-classify examples.
    Reduces the contribution of easy negatives.
    """
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.pos_weight is not None:
            alpha_weight = targets * self.pos_weight + (1 - targets) * 1.0
            focal_weight = focal_weight * alpha_weight

        loss = focal_weight * bce
        return loss.mean()


# ============================================================
# METRICS COMPUTATION
# ============================================================
def compute_metrics(logits, labels, threshold=0.5):
    """Compute all metrics from raw logits."""
    probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    labels_np = labels.detach().cpu().numpy().flatten()
    preds = (probs >= threshold).astype(int)

    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score,
        confusion_matrix
    )

    metrics = {}

    try:
        metrics['auroc'] = roc_auc_score(labels_np, probs)
    except ValueError:
        metrics['auroc'] = 0.0

    try:
        metrics['auprc'] = average_precision_score(labels_np, probs)
    except ValueError:
        metrics['auprc'] = 0.0

    metrics['precision'] = precision_score(labels_np, preds, zero_division=0)
    metrics['recall'] = recall_score(labels_np, preds, zero_division=0)
    metrics['f1'] = f1_score(labels_np, preds, zero_division=0)

    if len(np.unique(labels_np)) == 2 and len(np.unique(preds)) >= 1:
        tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0

    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['tn'] = int(tn)
    metrics['fn'] = int(fn)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['accuracy'] = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return metrics


# ============================================================
# OPTIMAL THRESHOLD FINDER
# ============================================================
def find_optimal_threshold(model, dataloader, device):
    """Find threshold that maximizes F1 on validation data."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            logits, _ = model(batch_X)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(batch_y.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    from sklearn.metrics import f1_score
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train():
    # === CONFIGURATION ===
    CONFIG = {
        "experiment": "exp002_standardized_multistep_focal",
        "data_dir": "/content/drive/MyDrive/ai sepsis training/standardized",
        "model_dir": "/content/drive/MyDrive/ai sepsis training/models",
        "results_dir": "/content/drive/MyDrive/ai sepsis training/results",
        # Architecture
        "input_dim": 73,
        "tabnet_dim": 128,
        "n_steps": 3,
        "gru_hidden": 128,
        "gru_layers": 1,
        "dropout": 0.3,
        "relaxation_factor": 1.5,
        "sparsity_coefficient": 1e-3,
        "virtual_batch_size": 64,
        # Training
        "batch_size": 256,
        "max_epochs": 15,
        "patience": 5,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "focal_gamma": 2.0,
        "pos_weight_cap": 10.0,  # Cap the positive weight to prevent collapse
        # Device
        "num_workers": 2,
    }

    # Create directories
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # === LOAD STANDARDIZED DATA ===
    print("\n[1/6] Loading standardized data...")
    data_dir = CONFIG["data_dir"]

    train_X = np.load(os.path.join(data_dir, "train_X.npy"))
    train_y = np.load(os.path.join(data_dir, "train_y.npy"))
    val_X = np.load(os.path.join(data_dir, "val_X.npy"))
    val_y = np.load(os.path.join(data_dir, "val_y.npy"))
    test_X = np.load(os.path.join(data_dir, "test_X.npy"))
    test_y = np.load(os.path.join(data_dir, "test_y.npy"))

    print(f"  Train: X={train_X.shape}, y={train_y.shape}")
    print(f"  Val:   X={val_X.shape},   y={val_y.shape}")
    print(f"  Test:  X={test_X.shape},  y={test_y.shape}")

    # === CLASS DISTRIBUTION ===
    n_pos = int(train_y.sum())
    n_neg = len(train_y) - n_pos
    raw_weight = n_neg / n_pos if n_pos > 0 else 1.0
    capped_weight = min(raw_weight, CONFIG["pos_weight_cap"])

    print(f"\n[2/6] Class distribution:")
    print(f"  Negative: {n_neg:,}")
    print(f"  Positive: {n_pos:,}")
    print(f"  Raw weight: {raw_weight:.2f}")
    print(f"  Capped weight: {capped_weight:.2f} (cap={CONFIG['pos_weight_cap']})")

    # === DATA LOADERS ===
    print("\n[3/6] Creating data loaders...")
    train_dataset = TensorDataset(
        torch.FloatTensor(train_X),
        torch.FloatTensor(train_y).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_X),
        torch.FloatTensor(val_y).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_X),
        torch.FloatTensor(test_y).unsqueeze(1)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=CONFIG["num_workers"],
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"],
        pin_memory=True
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # === MODEL ===
    print("\n[4/6] Building model...")
    model = TabNetGRU(
        input_dim=CONFIG["input_dim"],
        tabnet_dim=CONFIG["tabnet_dim"],
        n_steps=CONFIG["n_steps"],
        gru_hidden=CONFIG["gru_hidden"],
        gru_layers=CONFIG["gru_layers"],
        dropout=CONFIG["dropout"],
        relaxation_factor=CONFIG["relaxation_factor"],
        virtual_batch_size=CONFIG["virtual_batch_size"],
        sparsity_coefficient=CONFIG["sparsity_coefficient"]
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # === LOSS & OPTIMIZER ===
    criterion = FocalLoss(
        alpha=1.0,
        gamma=CONFIG["focal_gamma"],
        pos_weight=torch.tensor(capped_weight).to(DEVICE)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    print(f"  Loss: Focal Loss (gamma={CONFIG['focal_gamma']}, pos_weight={capped_weight:.2f})")
    print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']}, wd={CONFIG['weight_decay']})")
    print(f"  Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)")

    # === TRAINING LOOP ===
    print("\n[5/6] Training...")
    print("=" * 100)

    best_auprc = 0.0
    best_epoch = 0
    patience_counter = 0
    training_history = []

    checkpoint_path = os.path.join(
        CONFIG["model_dir"],
        f"best_tabnet_gru_{CONFIG['experiment']}.pt"
    )

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits, entropy_loss = model(batch_X)
            classification_loss = criterion(logits, batch_y)
            loss = classification_loss + entropy_loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_val_logits = []
        all_val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                logits, entropy_loss = model(batch_X)
                classification_loss = criterion(logits, batch_y)
                loss = classification_loss + entropy_loss

                val_loss_sum += loss.item()
                val_batches += 1
                all_val_logits.append(logits.cpu())
                all_val_labels.append(batch_y.cpu())

        avg_val_loss = val_loss_sum / val_batches
        all_val_logits = torch.cat(all_val_logits)
        all_val_labels = torch.cat(all_val_labels)

        val_metrics = compute_metrics(all_val_logits, all_val_labels)
        epoch_time = time.time() - epoch_start

        # Scheduler step
        scheduler.step(val_metrics['auprc'])

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{CONFIG['max_epochs']} | "
              f"Time: {epoch_time:.0f}s | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"AUROC: {val_metrics['auroc']:.4f} | "
              f"AUPRC: {val_metrics['auprc']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"Spec: {val_metrics['specificity']:.4f}")

        # Save history
        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr,
            **val_metrics,
            "time_seconds": epoch_time
        })

        # Checkpoint
        if val_metrics['auprc'] > best_auprc:
            best_auprc = val_metrics['auprc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': CONFIG
            }, checkpoint_path)
            print(f"  >>> New best AUPRC: {best_auprc:.6f} — checkpoint saved")
        else:
            patience_counter += 1
            print(f"  --- No improvement ({patience_counter}/{CONFIG['patience']})")

        if patience_counter >= CONFIG["patience"]:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    print("=" * 100)
    print(f"Training complete. Best epoch: {best_epoch}, Best AUPRC: {best_auprc:.6f}")

    # === TEST EVALUATION ===
    print("\n[6/6] Test evaluation...")

    # Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Find optimal threshold on validation
    print("  Finding optimal threshold on validation set...")
    best_threshold, best_val_f1 = find_optimal_threshold(model, val_loader, DEVICE)
    print(f"  Optimal threshold: {best_threshold:.2f} (val F1: {best_val_f1:.4f})")

    # Test with default threshold (0.5)
    all_test_logits = []
    all_test_labels = []
    test_loss_sum = 0.0
    test_batches = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits, entropy_loss = model(batch_X)
            classification_loss = criterion(logits, batch_y)
            loss = classification_loss + entropy_loss

            test_loss_sum += loss.item()
            test_batches += 1
            all_test_logits.append(logits.cpu())
            all_test_labels.append(batch_y.cpu())

    all_test_logits = torch.cat(all_test_logits)
    all_test_labels = torch.cat(all_test_labels)

    # Metrics at default threshold
    test_metrics_default = compute_metrics(all_test_logits, all_test_labels, threshold=0.5)
    # Metrics at optimal threshold
    test_metrics_optimal = compute_metrics(all_test_logits, all_test_labels, threshold=best_threshold)

    avg_test_loss = test_loss_sum / test_batches

    print("\n" + "=" * 60)
    print("TEST RESULTS (threshold=0.50)")
    print("=" * 60)
    print(f"  Loss:        {avg_test_loss:.4f}")
    print(f"  AUROC:       {test_metrics_default['auroc']:.4f}")
    print(f"  AUPRC:       {test_metrics_default['auprc']:.4f}")
    print(f"  Accuracy:    {test_metrics_default['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics_default['precision']:.4f}")
    print(f"  Recall:      {test_metrics_default['recall']:.4f}")
    print(f"  F1:          {test_metrics_default['f1']:.4f}")
    print(f"  Specificity: {test_metrics_default['specificity']:.4f}")
    print(f"  TP: {test_metrics_default['tp']:,} | FP: {test_metrics_default['fp']:,} | "
          f"TN: {test_metrics_default['tn']:,} | FN: {test_metrics_default['fn']:,}")

    print(f"\nTEST RESULTS (threshold={best_threshold:.2f} — optimized on val)")
    print("=" * 60)
    print(f"  Accuracy:    {test_metrics_optimal['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics_optimal['precision']:.4f}")
    print(f"  Recall:      {test_metrics_optimal['recall']:.4f}")
    print(f"  F1:          {test_metrics_optimal['f1']:.4f}")
    print(f"  Specificity: {test_metrics_optimal['specificity']:.4f}")
    print(f"  TP: {test_metrics_optimal['tp']:,} | FP: {test_metrics_optimal['fp']:,} | "
          f"TN: {test_metrics_optimal['tn']:,} | FN: {test_metrics_optimal['fn']:,}")

    # === SAVE RESULTS ===
    results = {
        "experiment": CONFIG["experiment"],
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "training_history": training_history,
        "best_epoch": best_epoch,
        "best_val_auprc": best_auprc,
        "optimal_threshold": best_threshold,
        "test_loss": avg_test_loss,
        "test_metrics_default_threshold": test_metrics_default,
        "test_metrics_optimal_threshold": test_metrics_optimal,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    results_path = os.path.join(
        CONFIG["results_dir"],
        f"results_{CONFIG['experiment']}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return results


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    train()
