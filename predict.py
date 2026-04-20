"""
predict.py — Adversarial attack implementations and inference pipeline.

Attacks implemented:
    • FGSM  — Fast Gradient Sign Method  (single step)
    • BIM   — Basic Iterative Method     (iterative FGSM)
    • PGD   — Projected Gradient Descent (iterative + random start)

Usage (CLI):
    python predict.py --attack fgsm --epsilon 0.03
    python predict.py --attack pgd  --epsilon 0.03 --iterations 20
    python predict.py --attack all  --epsilon 0.03

Usage (as module):
    from predict import load_model, fgsm_attack, pgd_attack
    model = load_model()
    adv   = pgd_attack(model, x_batch, y_batch, epsilon=0.03)
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from utils import (
    load_and_preprocess_data,
    get_sample_batch,
    evaluate_model,
    print_results_table,
    visualize_adversarial_examples,
    plot_attack_accuracy_bar,
    plot_confusion_matrix,
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cifar10_cnn.keras")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Shared loss object (re-used across attacks for efficiency)
_LOSS_FN = tf.keras.losses.CategoricalCrossentropy()


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model(path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Load a saved Keras model from disk.

    Args:
        path (str): Path to the saved model file (.keras or .h5).

    Returns:
        tf.keras.Model: Loaded model ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Please run  python train.py  first to train and save the model."
        )
    print(f"📂 Loading model from {path} …")
    return tf.keras.models.load_model(path)


# ─────────────────────────────────────────────
# Adversarial attack helpers
# ─────────────────────────────────────────────

def _compute_gradient(model: tf.keras.Model, images: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Compute the gradient of the loss w.r.t. the input images.

    Args:
        model:  Keras model.
        images: Input tensor [B, H, W, C].
        labels: One-hot label tensor [B, num_classes].

    Returns:
        tf.Tensor: Gradient tensor with the same shape as *images*.
    """
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images, training=False)
        loss  = _LOSS_FN(labels, preds)
    return tape.gradient(loss, images)


# ── FGSM ──────────────────────────────────────

def fgsm_attack(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 0.03,
) -> np.ndarray:
    """
    Fast Gradient Sign Method (FGSM) — single-step attack.

    Perturbs each pixel by ε in the direction of the loss gradient.

    Reference: Goodfellow et al., "Explaining and Harnessing Adversarial
    Examples", ICLR 2015.

    Args:
        model:   Keras model.
        images:  Clean images  [N, 32, 32, 3], values in [0, 1].
        labels:  One-hot labels [N, 10].
        epsilon: Perturbation magnitude (typical: 0.01 – 0.1).

    Returns:
        np.ndarray: Adversarial images clipped to [0, 1].
    """
    images_t = tf.convert_to_tensor(images, dtype=tf.float32)
    labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)

    grad        = _compute_gradient(model, images_t, labels_t)
    signed_grad = tf.sign(grad)
    adv         = images_t + epsilon * signed_grad
    adv         = tf.clip_by_value(adv, 0.0, 1.0)

    return adv.numpy()


# ── BIM ───────────────────────────────────────

def bim_attack(
    model:      tf.keras.Model,
    images:     np.ndarray,
    labels:     np.ndarray,
    epsilon:    float = 0.03,
    alpha:      float = 0.005,
    iterations: int   = 10,
) -> np.ndarray:
    """
    Basic Iterative Method (BIM) — iterative FGSM with ε-ball projection.

    Applies FGSM *iterations* times with step size *alpha*, projecting
    back onto the ε-ball around the original images after each step.

    Reference: Kurakin et al., "Adversarial Examples in the Physical
    World", ICLR 2017.

    Args:
        model:      Keras model.
        images:     Clean images  [N, 32, 32, 3].
        labels:     One-hot labels [N, 10].
        epsilon:    Maximum total perturbation.
        alpha:      Per-step perturbation size.
        iterations: Number of attack steps.

    Returns:
        np.ndarray: Adversarial images clipped to [0, 1].
    """
    images_t = tf.convert_to_tensor(images, dtype=tf.float32)
    labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)
    adv      = images_t

    for _ in range(iterations):
        grad        = _compute_gradient(model, adv, labels_t)
        signed_grad = tf.sign(grad)
        adv         = adv + alpha * signed_grad

        # Project onto ε-ball around original
        adv = tf.clip_by_value(adv, images_t - epsilon, images_t + epsilon)
        adv = tf.clip_by_value(adv, 0.0, 1.0)

    return adv.numpy()


# ── PGD ───────────────────────────────────────

def pgd_attack(
    model:      tf.keras.Model,
    images:     np.ndarray,
    labels:     np.ndarray,
    epsilon:    float = 0.03,
    alpha:      float = 0.005,
    iterations: int   = 20,
    random_start: bool = True,
) -> np.ndarray:
    """
    Projected Gradient Descent (PGD) — the strongest of the three attacks.

    Adds a uniform random noise initialisation before iterating, making
    it less likely to get stuck in local optima.

    Reference: Madry et al., "Towards Deep Learning Models Resistant to
    Adversarial Attacks", ICLR 2018.

    Args:
        model:        Keras model.
        images:       Clean images  [N, 32, 32, 3].
        labels:       One-hot labels [N, 10].
        epsilon:      Maximum total perturbation (ℓ∞ ball radius).
        alpha:        Per-step perturbation size.
        iterations:   Number of gradient steps.
        random_start: Whether to add uniform noise before iterating.

    Returns:
        np.ndarray: Adversarial images clipped to [0, 1].
    """
    images_t = tf.convert_to_tensor(images, dtype=tf.float32)
    labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)

    if random_start:
        noise = tf.random.uniform(tf.shape(images_t), -epsilon, epsilon)
        adv   = tf.clip_by_value(images_t + noise, 0.0, 1.0)
    else:
        adv = images_t

    for _ in range(iterations):
        grad        = _compute_gradient(model, adv, labels_t)
        signed_grad = tf.sign(grad)
        adv         = adv + alpha * signed_grad

        # Project
        adv = tf.clip_by_value(adv, images_t - epsilon, images_t + epsilon)
        adv = tf.clip_by_value(adv, 0.0, 1.0)

    return adv.numpy()


# ─────────────────────────────────────────────
# Single-image inference
# ─────────────────────────────────────────────

def predict_single(model: tf.keras.Model, image: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    Run inference on a single image and return the top-k predictions.

    Args:
        model: Keras model.
        image: Image array — shape (32, 32, 3) or (1, 32, 32, 3), values in [0, 1].
        top_k: Number of top predictions to return (1–10).

    Returns:
        list[dict]: Each dict has 'class', 'label', and 'confidence' keys.
    """
    from utils import CLASS_NAMES

    if image.ndim == 3:
        image = image[np.newaxis, ...]          # add batch dim

    probs   = model.predict(image, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]

    return [
        {
            "class":      int(idx),
            "label":      CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
        }
        for idx in top_idx
    ]


# ─────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────

def run_all_attacks(
    model:           tf.keras.Model,
    x_test:          np.ndarray,
    y_test:          np.ndarray,
    sample_size:     int   = 500,
    epsilon:         float = 0.03,
    save_visuals:    bool  = True,
) -> list[tuple]:
    """
    Run clean evaluation + all three attacks and collect accuracy results.

    Args:
        model:        Keras model.
        x_test:       Full test images.
        y_test:       Full test labels.
        sample_size:  Number of test samples to attack (attacks are slow).
        epsilon:      Perturbation bound for all attacks.
        save_visuals: Whether to save visualisation plots.

    Returns:
        list[tuple]: (attack_name, epsilon_str, accuracy) for each run.
    """
    os.makedirs(ASSETS_DIR, exist_ok=True)
    x_sample, y_sample = get_sample_batch(x_test, y_test, n=sample_size)
    results             = []

    # ── Clean ─────────────────────────────────
    print("\n🔍 Evaluating on clean data …")
    acc = evaluate_model(model, x_sample, y_sample, label="Clean")
    results.append(("Clean", "-", acc))

    # ── FGSM across multiple ε ────────────────
    for eps in [0.01, 0.03, 0.05, 0.1]:
        print(f"\n⚡ FGSM attack  ε={eps} …")
        adv = fgsm_attack(model, x_sample, y_sample, epsilon=eps)
        acc = evaluate_model(model, adv, y_sample, label=f"FGSM ε={eps}")
        results.append(("FGSM", eps, acc))

    # ── BIM ───────────────────────────────────
    print(f"\n🔁 BIM attack  ε={epsilon} …")
    adv_bim = bim_attack(model, x_sample, y_sample, epsilon=epsilon)
    acc      = evaluate_model(model, adv_bim, y_sample, label="BIM")
    results.append(("BIM", epsilon, acc))

    if save_visuals:
        visualize_adversarial_examples(
            x_sample, adv_bim, model, title="BIM Adversarial Examples",
            save_path=os.path.join(ASSETS_DIR, "adv_bim.png"),
        )

    # ── PGD ───────────────────────────────────
    print(f"\n🎯 PGD attack  ε={epsilon} …")
    adv_pgd = pgd_attack(model, x_sample, y_sample, epsilon=epsilon)
    acc      = evaluate_model(model, adv_pgd, y_sample, label="PGD")
    results.append(("PGD", epsilon, acc))

    if save_visuals:
        visualize_adversarial_examples(
            x_sample, adv_pgd, model, title="PGD Adversarial Examples",
            save_path=os.path.join(ASSETS_DIR, "adv_pgd.png"),
        )
        plot_attack_accuracy_bar(
            results,
            save_path=os.path.join(ASSETS_DIR, "attack_comparison.png"),
        )
        plot_confusion_matrix(
            model, adv_pgd, y_sample, label="PGD",
            save_path=os.path.join(ASSETS_DIR, "confusion_matrix_pgd.png"),
        )

    print_results_table(results)
    return results


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run adversarial attacks on the trained CIFAR-10 CNN."
    )
    parser.add_argument(
        "--attack",
        choices=["fgsm", "bim", "pgd", "all"],
        default="all",
        help="Which attack to run (default: all)",
    )
    parser.add_argument("--epsilon",    type=float, default=0.03, help="Perturbation bound ε (default 0.03)")
    parser.add_argument("--alpha",      type=float, default=0.005, help="Step size for BIM/PGD (default 0.005)")
    parser.add_argument("--iterations", type=int,   default=20,    help="Iterations for BIM/PGD (default 20)")
    parser.add_argument("--samples",    type=int,   default=500,   help="Test samples to attack (default 500)")
    return parser.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    model = load_model()

    _, _, x_test, y_test = load_and_preprocess_data()
    x_s, y_s = get_sample_batch(x_test, y_test, n=args.samples)

    results = []

    if args.attack in ("fgsm", "all"):
        print(f"\n⚡ Running FGSM  ε={args.epsilon} …")
        adv = fgsm_attack(model, x_s, y_s, epsilon=args.epsilon)
        acc = evaluate_model(model, adv, y_s, label=f"FGSM ε={args.epsilon}")
        results.append(("FGSM", args.epsilon, acc))

    if args.attack in ("bim", "all"):
        print(f"\n🔁 Running BIM  ε={args.epsilon} …")
        adv = bim_attack(model, x_s, y_s, epsilon=args.epsilon, alpha=args.alpha, iterations=args.iterations)
        acc = evaluate_model(model, adv, y_s, label="BIM")
        results.append(("BIM", args.epsilon, acc))

    if args.attack in ("pgd", "all"):
        print(f"\n🎯 Running PGD  ε={args.epsilon} …")
        adv = pgd_attack(model, x_s, y_s, epsilon=args.epsilon, alpha=args.alpha, iterations=args.iterations)
        acc = evaluate_model(model, adv, y_s, label="PGD")
        results.append(("PGD", args.epsilon, acc))

    if results:
        print_results_table(results)
