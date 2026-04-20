"""
utils.py — Shared helper functions for the CIFAR-10 Adversarial Robustness project.

Covers:
  - Data loading & preprocessing
  - Visualization (training history, adversarial examples, confusion matrix)
  - Model evaluation utilities
  - Attack result summarization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMG_SIZE = (32, 32, 3)
NUM_CLASSES = 10


# ─────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────

def load_and_preprocess_data():
    """
    Load CIFAR-10 from Keras, normalize pixel values to [0, 1],
    and one-hot encode labels.

    Returns:
        tuple: (x_train, y_train, x_test, y_test) as NumPy arrays.
    """
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical

    print("📦 Loading CIFAR-10 dataset …")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # One-hot encode
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test  = to_categorical(y_test,  NUM_CLASSES)

    print(f"✅ Train: {x_train.shape}  |  Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test


def get_sample_batch(x, y, n=100):
    """
    Return the first *n* samples — useful for quick attack demos
    so you don't run attacks on the full 10 000 test images.

    Args:
        x (np.ndarray): Images array.
        y (np.ndarray): One-hot labels array.
        n (int): Number of samples to return.

    Returns:
        tuple: (x_sample, y_sample)
    """
    return x[:n], y[:n]


# ─────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────

def evaluate_model(model, x, y, label=""):
    """
    Evaluate the model and print accuracy, loss, and a full
    classification report.

    Args:
        model: Keras model.
        x (np.ndarray): Input images.
        y (np.ndarray): One-hot labels.
        label (str): Human-readable name shown in the output.

    Returns:
        float: Accuracy on the supplied data.
    """
    loss, acc = model.evaluate(x, y, verbose=0)
    tag = f"[{label}]" if label else ""
    print(f"\n{tag} Accuracy: {acc:.4f}  |  Loss: {loss:.4f}")

    y_pred_classes = np.argmax(model.predict(x, verbose=0), axis=1)
    y_true         = np.argmax(y, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

    return acc


def print_results_table(results: list):
    """
    Pretty-print a summary table of attack results.

    Args:
        results (list): List of (attack_name, epsilon, accuracy) tuples.
    """
    print("\n" + "=" * 45)
    print(f"{'Attack':<12} {'Epsilon':>10} {'Accuracy':>10}")
    print("=" * 45)
    for name, eps, acc in results:
        eps_str = f"{eps:.4f}" if isinstance(eps, float) else str(eps)
        print(f"{name:<12} {eps_str:>10} {acc*100:>9.2f}%")
    print("=" * 45)


# ─────────────────────────────────────────────
# Visualization utilities
# ─────────────────────────────────────────────

def plot_training_history(history, save_path: str = None):
    """
    Plot training/validation accuracy and loss curves.

    Args:
        history: Keras History object returned by model.fit().
        save_path (str, optional): If provided, saves the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History — CIFAR-10 CNN", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train",      linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train",      linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Training history saved → {save_path}")
    plt.show()


def plot_confusion_matrix(model, x, y, label="", save_path: str = None):
    """
    Compute predictions and render a colour-coded confusion matrix.

    Args:
        model: Keras model.
        x (np.ndarray): Input images.
        y (np.ndarray): One-hot labels.
        label (str): Title suffix.
        save_path (str, optional): If provided, saves the figure.
    """
    y_pred   = np.argmax(model.predict(x, verbose=0), axis=1)
    y_true   = np.argmax(y, axis=1)
    cm       = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.title(f"Confusion Matrix — {label}" if label else "Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Confusion matrix saved → {save_path}")
    plt.show()


def visualize_adversarial_examples(
    x_clean: np.ndarray,
    x_adv:   np.ndarray,
    model,
    n: int = 10,
    title: str = "Adversarial Examples",
    save_path: str = None,
):
    """
    Side-by-side comparison: clean | adversarial | perturbation (amplified).

    Rows:
        Row 1 — Original images
        Row 2 — Adversarial images
        Row 3 — Perturbation (×10 for visibility)

    Args:
        x_clean:   Original test images  [N, 32, 32, 3].
        x_adv:     Adversarial images    [N, 32, 32, 3].
        model:     Keras model for obtaining predictions.
        n (int):   Number of examples to show (≤ 10 recommended).
        title:     Figure title.
        save_path: Optional save location.
    """
    n = min(n, len(x_clean), 10)
    clean_preds = np.argmax(model.predict(x_clean[:n], verbose=0), axis=1)
    adv_preds   = np.argmax(model.predict(x_adv[:n],   verbose=0), axis=1)

    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i in range(n):
        # Original
        axes[0, i].imshow(np.clip(x_clean[i], 0, 1))
        axes[0, i].set_title(CLASS_NAMES[clean_preds[i]], fontsize=7)
        axes[0, i].axis("off")

        # Adversarial
        axes[1, i].imshow(np.clip(x_adv[i], 0, 1))
        axes[1, i].set_title(CLASS_NAMES[adv_preds[i]], fontsize=7,
                              color="red" if adv_preds[i] != clean_preds[i] else "green")
        axes[1, i].axis("off")

        # Amplified perturbation
        diff = np.clip((x_adv[i] - x_clean[i]) * 10 + 0.5, 0, 1)
        axes[2, i].imshow(diff)
        axes[2, i].axis("off")

    # Row labels
    for row_idx, row_label in enumerate(["Original", "Adversarial", "Perturbation ×10"]):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Adversarial examples saved → {save_path}")
    plt.show()


def plot_attack_accuracy_bar(results: list, save_path: str = None):
    """
    Bar chart comparing clean vs. attacked accuracy across all attacks.

    Args:
        results: List of (attack_name, epsilon, accuracy) tuples.
        save_path: Optional save path.
    """
    labels    = [f"{r[0]}\nε={r[1]}" for r in results]
    accuracies = [r[2] * 100 for r in results]
    colors    = ["#4CAF50"] + ["#F44336"] * (len(results) - 1)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, accuracies, color=colors, edgecolor="white", linewidth=0.8)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Under Adversarial Attacks", fontweight="bold")

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Attack bar chart saved → {save_path}")
    plt.show()
