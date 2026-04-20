"""
train.py — Build, train, and save the CIFAR-10 CNN model.

Usage:
    python train.py                         # default 30 epochs
    python train.py --epochs 50             # custom epoch count
    python train.py --epochs 10 --batch 128 # fast debug run

The trained model is saved to  model/cifar10_cnn.keras
Training plots are saved to    assets/training_history.png
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

# Local imports
from utils import (
    load_and_preprocess_data,
    plot_training_history,
    plot_confusion_matrix,
    evaluate_model,
    IMG_SIZE,
    NUM_CLASSES,
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn.keras")


# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────

def build_model(input_shape: tuple = IMG_SIZE, num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build a three-block CNN suitable for CIFAR-10.

    Architecture summary:
        Conv(32,3×3) → MaxPool(2×2)
        Conv(64,5×5) → MaxPool(2×2)
        Conv(128,7×7)
        Flatten → Dense(128) → Dense(10, softmax)

    Dropout is added after each dense layer to improve generalisation.

    Args:
        input_shape (tuple): Image dimensions — default (32, 32, 3).
        num_classes (int):   Number of output classes — default 10.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = tf.keras.Sequential(
        [
            # ── Block 1 ──────────────────────────────
            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", activation="relu",
                input_shape=input_shape,
                name="conv1",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),

            # ── Block 2 ──────────────────────────────
            tf.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu", name="conv2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool2"),

            # ── Block 3 ──────────────────────────────
            tf.keras.layers.Conv2D(128, (7, 7), padding="same", activation="relu", name="conv3"),
            tf.keras.layers.BatchNormalization(),

            # ── Classifier ───────────────────────────
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu", name="fc1"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="cifar10_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(epochs: int = 30, batch_size: int = 64):
    """
    Full training pipeline:
        1. Load & preprocess data
        2. Build model
        3. Train with early stopping + LR reduction
        4. Evaluate on test set
        5. Save model & plots

    Args:
        epochs (int):     Maximum training epochs.
        batch_size (int): Mini-batch size.
    """
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # ── Model ─────────────────────────────────
    model = build_model()
    model.summary()

    # ── Callbacks ─────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1
        ),
    ]

    # ── Training ──────────────────────────────
    print(f"\n🚀 Training for up to {epochs} epoch(s) …\n")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation ────────────────────────────
    print("\n📊 Evaluating on clean test set …")
    evaluate_model(model, x_test, y_test, label="Clean Test Set")

    # ── Plots ─────────────────────────────────
    plot_training_history(
        history,
        save_path=os.path.join(ASSETS_DIR, "training_history.png"),
    )
    plot_confusion_matrix(
        model, x_test, y_test,
        label="Clean",
        save_path=os.path.join(ASSETS_DIR, "confusion_matrix_clean.png"),
    )

    # ── Persist model ─────────────────────────
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    return model, history


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 CNN for adversarial robustness experiments."
    )
    parser.add_argument("--epochs",    type=int, default=30,  help="Max training epochs (default 30)")
    parser.add_argument("--batch",     type=int, default=64,  help="Batch size (default 64)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    train(epochs=args.epochs, batch_size=args.batch)
