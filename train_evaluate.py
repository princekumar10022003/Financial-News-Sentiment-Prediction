"""
train_evaluate.py
-----------------
Reusable training loop, evaluation, confusion matrix, and model comparison
for all RNN-based classifiers.

Run directly to train all three models and print the comparison table.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────
# Training loop with early stopping
# ─────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    patience: int = 3,
    save_path: str = None,
) -> dict:
    """
    Train a model with early stopping.

    Returns a dict with:
        train_losses, val_losses, val_accs, best_epoch
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_weights  = None
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>10} {'Val Acc':>9} {'Macro F1':>9}")
    print("-" * 52)

    for epoch in range(1, epochs + 1):
        # ── Training ────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation ──────────────────────────────────────────────
        val_loss, val_acc, val_f1 = evaluate_loss_acc(model, valid_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"{epoch:>5} {avg_train_loss:>11.4f} {val_loss:>10.4f} "
              f"{val_acc:>8.2%} {val_f1:>9.4f}")

        # ── Early stopping ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch       = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    history["best_epoch"] = best_epoch

    # Optionally save checkpoint
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return history


# ─────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────
def evaluate_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return (avg_loss, accuracy, macro_f1) on a DataLoader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits  = model(X_batch)
            loss    = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str = "Model",
) -> dict:
    """
    Full evaluation: classification report + confusion matrix plot.
    Returns dict with accuracy, macro_f1, per_class_f1, all predictions.
    """
    label_names = ["Bearish", "Bullish", "Neutral"]
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            preds   = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_cls  = f1_score(all_labels, all_preds, average=None, zero_division=0)

    print(f"\n{'='*55}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc:.2%})")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print(f"{'─'*55}")
    print(classification_report(all_labels, all_preds,
                                target_names=label_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"models/{model_name.replace(' ', '_')}_confusion.png", dpi=150)
    plt.show()
    print(f"Confusion matrix saved.")

    return {
        "model":        model_name,
        "accuracy":     round(acc, 4),
        "macro_f1":     round(macro_f1, 4),
        "bearish_f1":   round(per_cls[0], 4),
        "bullish_f1":   round(per_cls[1], 4),
        "neutral_f1":   round(per_cls[2], 4),
        "preds":        all_preds,
        "labels":       all_labels,
    }


# ─────────────────────────────────────────────────────────────────
# Loss / accuracy curves
# ─────────────────────────────────────────────────────────────────
def plot_history(history: dict, model_name: str = "Model"):
    """Plot train/val loss and val accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s", markersize=4)
    axes[0].axvline(history.get("best_epoch", 0), color="gray", linestyle="--", alpha=0.6,
                    label=f"Best epoch ({history.get('best_epoch', '?')})")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Macro F1
    axes[1].plot(epochs, history["val_f1"],  label="Val Macro F1",  color="green", marker="^", markersize=4)
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy",   color="blue",  marker="o", markersize=4)
    axes[1].set_title(f"{model_name} — Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(model_name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"models/{model_name.replace(' ', '_')}_history.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────
def print_comparison_table(results: list[dict]):
    """Print a nicely formatted comparison table of all models."""
    df = pd.DataFrame(results)[
        ["model", "accuracy", "macro_f1", "bearish_f1", "bullish_f1", "neutral_f1"]
    ]
    df.columns = ["Model", "Accuracy", "Macro F1", "Bearish F1", "Bullish F1", "Neutral F1"]
    df = df.sort_values("Macro F1", ascending=False).reset_index(drop=True)

    print("\n" + "="*75)
    print("  MODEL COMPARISON TABLE")
    print("="*75)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*75)
    return df


def plot_comparison_bar(results: list[dict]):
    """Grouped bar chart comparing all models on key metrics."""
    df = pd.DataFrame(results)
    metrics = ["accuracy", "macro_f1", "bearish_f1", "bullish_f1", "neutral_f1"]
    labels  = ["Accuracy", "Macro F1", "Bearish F1", "Bullish F1", "Neutral F1"]

    x    = np.arange(len(metrics))
    w    = 0.8 / len(df)
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * w - 0.4 + w/2, vals, width=w * 0.9,
               label=row["model"], color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/model_comparison.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────
# Main: train all 3 RNN models
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import (
        load_data, Vocabulary, build_loaders,
        get_class_weights, VOCAB_SIZE, PAD_IDX
    )
    from rnn_models import SimpleRNNClassifier, LSTMClassifier, GRUClassifier

    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load & preprocess data ─────────────────────────────────────
    train_df, valid_df = load_data("data/sent_train.csv", "data/sent_valid.csv")

    vocab = Vocabulary(max_size=VOCAB_SIZE)
    vocab.build(train_df["clean"].tolist())

    train_loader, valid_loader = build_loaders(train_df, valid_df, vocab)

    class_weights = get_class_weights(train_df["label"].values, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Define models ─────────────────────────────────────────────
    model_configs = [
        ("Simple RNN", SimpleRNNClassifier, "models/simplernn_best.pt"),
        ("LSTM",       LSTMClassifier,      "models/lstm_best.pt"),
        ("GRU",        GRUClassifier,       "models/gru_best.pt"),
    ]

    all_results = []

    for model_name, ModelClass, save_path in model_configs:
        print(f"\n{'#'*60}")
        print(f"  Training: {model_name}")
        print(f"{'#'*60}")

        model = ModelClass(vocab_size=len(vocab))
        print(model)

        start = time.time()
        history = train_model(
            model, train_loader, valid_loader,
            criterion=criterion,
            device=device,
            epochs=15,
            lr=1e-3,
            patience=3,
            save_path=save_path,
        )
        elapsed = time.time() - start
        print(f"Training time: {elapsed/60:.1f} min")

        plot_history(history, model_name)

        results = full_evaluation(model, valid_loader, device, model_name)
        all_results.append(results)

    # ── Comparison ────────────────────────────────────────────────
    comp_df = print_comparison_table(all_results)
    plot_comparison_bar(all_results)

    # Save comparison to CSV for the report
    comp_df.to_csv("models/rnn_comparison.csv", index=False)
    print("\nComparison saved to models/rnn_comparison.csv")

    # Save vocab for Streamlit app
    import pickle
    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved to models/vocab.pkl")
