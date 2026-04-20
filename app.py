"""
app.py — Streamlit web application for the CIFAR-10 Adversarial Robustness Explorer.

Run:
    streamlit run app.py

Features:
    • Upload any 32×32 (or larger, auto-resized) RGB image
    • Choose attack type and ε strength via sliders
    • See original vs adversarial image side-by-side
    • Top-5 prediction confidence bars for both
    • Live perturbation magnitude heatmap
"""

import io
import os

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="CIFAR-10 Adversarial Robustness Explorer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lazy TF import so Streamlit loads fast
import tensorflow as tf
from predict import fgsm_attack, bim_attack, pgd_attack, predict_single, load_model
from utils import CLASS_NAMES


# ─────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cifar10_cnn.keras")


@st.cache_resource(show_spinner="Loading model …")
def get_model():
    """Load model once and cache for the session."""
    return load_model(MODEL_PATH)


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def pil_to_cifar_array(pil_image: Image.Image) -> np.ndarray:
    """Resize a PIL image to 32×32 and convert to float32 [0, 1]."""
    img = pil_image.convert("RGB").resize((32, 32), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def confidence_bar(label: str, confidence: float, colour: str = "#4CAF50"):
    """Render a single labelled confidence bar using raw HTML."""
    bar_pct  = confidence * 100
    bar_html = (
        f"<div style='margin-bottom:6px'>"
        f"  <div style='font-size:13px;margin-bottom:2px'>{label}</div>"
        f"  <div style='background:#e0e0e0;border-radius:4px;height:20px;width:100%'>"
        f"    <div style='background:{colour};width:{bar_pct:.1f}%;height:100%;"
        f"         border-radius:4px;min-width:2px'></div>"
        f"  </div>"
        f"  <div style='font-size:12px;color:#555'>{bar_pct:.1f}%</div>"
        f"</div>"
    )
    st.markdown(bar_html, unsafe_allow_html=True)


def render_top_k(predictions: list, title: str, colour: str = "#4CAF50"):
    """Render a top-k prediction card."""
    st.markdown(f"**{title}**")
    for pred in predictions:
        confidence_bar(pred["label"].capitalize(), pred["confidence"], colour=colour)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/"
        "Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png",
        width=60,
    )
    st.title("⚙️ Controls")
    st.markdown("---")

    attack_type = st.selectbox(
        "Attack type",
        ["FGSM", "BIM", "PGD"],
        help=(
            "**FGSM** — single-step, fastest.\n\n"
            "**BIM** — iterative FGSM, stronger.\n\n"
            "**PGD** — iterative + random start, strongest."
        ),
    )

    epsilon = st.slider(
        "Epsilon (ε) — perturbation strength",
        min_value=0.001, max_value=0.15,
        value=0.03, step=0.005,
        help="Higher ε = stronger (more visible) attack.",
    )

    if attack_type in ("BIM", "PGD"):
        iterations = st.slider("Iterations", min_value=5, max_value=50, value=20, step=5)
        alpha       = st.slider("Step size (α)", min_value=0.001, max_value=0.02, value=0.005, step=0.001)
    else:
        iterations = 10
        alpha       = 0.005

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app demonstrates how small, imperceptible pixel perturbations "
        "can fool a deep CNN trained on CIFAR-10.  \n\n"
        "Upload any image to see how each attack affects the model's prediction."
    )
    st.markdown("---")
    st.markdown("Built with 🤍 using TensorFlow + Streamlit")


# ─────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────

st.title("🛡️ CIFAR-10 Adversarial Robustness Explorer")
st.markdown(
    "Upload an image (or use the built-in test sample) to see how adversarial attacks "
    "fool the CNN — and compare original vs. adversarial predictions side-by-side."
)
st.markdown("---")

# ── Model loading ─────────────────────────────
try:
    model = get_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("💡 Train the model first by running:  `python train.py`")
    st.stop()

# ── Image input ───────────────────────────────
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG — any size, will be resized to 32×32)",
        type=["jpg", "jpeg", "png"],
    )

with col_sample:
    use_sample = st.button("🎲 Use random\ntest image")

# Load image
image_array = None

if uploaded is not None:
    pil_img     = Image.open(uploaded)
    image_array = pil_to_cifar_array(pil_img)
    st.success("✅ Image uploaded and resized to 32×32.")

elif use_sample:
    from tensorflow.keras.datasets import cifar10
    (_, _), (x_test, y_test) = cifar10.load_data()
    idx         = np.random.randint(0, len(x_test))
    image_array = x_test[idx].astype("float32") / 255.0
    st.info(f"🎲 Loaded CIFAR-10 test image #{idx}  (true label: **{CLASS_NAMES[y_test[idx][0]]}**)")

# ── Attack + display ──────────────────────────
if image_array is not None:
    st.markdown("---")

    # Run attack
    with st.spinner(f"Running {attack_type} attack …"):
        img_batch = image_array[np.newaxis, ...]  # [1, 32, 32, 3]

        # Dummy one-hot label — attacks are label-agnostic in untargeted mode
        top_pred   = np.argmax(model.predict(img_batch, verbose=0), axis=1)[0]
        dummy_label = tf.keras.utils.to_categorical([top_pred], num_classes=10)

        if attack_type == "FGSM":
            adv_array = fgsm_attack(model, img_batch, dummy_label, epsilon=epsilon)[0]
        elif attack_type == "BIM":
            adv_array = bim_attack(model, img_batch, dummy_label,
                                   epsilon=epsilon, alpha=alpha, iterations=iterations)[0]
        else:  # PGD
            adv_array = pgd_attack(model, img_batch, dummy_label,
                                   epsilon=epsilon, alpha=alpha, iterations=iterations)[0]

    # Predictions
    orig_preds = predict_single(model, image_array, top_k=5)
    adv_preds  = predict_single(model, adv_array,   top_k=5)

    # ── Layout: image | predictions ───────────
    c1, c2, c3, c4 = st.columns([1, 1.4, 1, 1.4])

    with c1:
        st.markdown("#### 🖼️ Original")
        disp_orig = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8))
        disp_orig = disp_orig.resize((160, 160), Image.NEAREST)
        st.image(disp_orig, caption="Original (32×32 × 5)")

    with c2:
        render_top_k(orig_preds, "Original Predictions", colour="#4CAF50")

    with c3:
        st.markdown(f"#### 💥 Adversarial ({attack_type})")
        disp_adv = Image.fromarray((np.clip(adv_array, 0, 1) * 255).astype(np.uint8))
        disp_adv = disp_adv.resize((160, 160), Image.NEAREST)
        st.image(disp_adv, caption=f"{attack_type}  ε={epsilon:.3f}")

    with c4:
        render_top_k(adv_preds, "Adversarial Predictions", colour="#F44336")

    st.markdown("---")

    # ── Perturbation heatmap ───────────────────
    with st.expander("🔬 View perturbation heatmap"):
        import matplotlib.pyplot as plt

        diff     = np.abs(adv_array - image_array)
        diff_amp = np.clip(diff * 10 + 0.5, 0, 1)          # amplified
        diff_mag = diff.mean(axis=-1)                        # channel mean → 2-D

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(np.clip(image_array, 0, 1));  axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(np.clip(adv_array,   0, 1));  axes[1].set_title("Adversarial"); axes[1].axis("off")
        im = axes[2].imshow(diff_mag, cmap="hot");   axes[2].set_title("Perturbation (×10)"); axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        fig.suptitle(f"{attack_type} — ε={epsilon:.3f}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        l2  = float(np.linalg.norm(adv_array - image_array))
        linf = float(np.max(np.abs(adv_array - image_array)))
        mc1, mc2 = st.columns(2)
        mc1.metric("ℓ∞ distance", f"{linf:.5f}")
        mc2.metric("ℓ₂ distance", f"{l2:.4f}")

    # ── Key insight ───────────────────────────
    orig_top  = orig_preds[0]["label"].capitalize()
    adv_top   = adv_preds[0]["label"].capitalize()
    fooled    = orig_top != adv_top

    if fooled:
        st.warning(
            f"⚠️  **Attack succeeded!**  The model changed its prediction from "
            f"**{orig_top}** → **{adv_top}** with only ε={epsilon:.3f} perturbation."
        )
    else:
        st.success(
            f"✅  **Model held firm!**  Still predicts **{orig_top}** despite the {attack_type} attack."
        )

else:
    # No image loaded yet
    st.info("👆 Upload an image above or click **Use random test image** to get started.")

    # ── Overview cards ────────────────────────
    st.markdown("### 🗂️ Attack Overview")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown("#### ⚡ FGSM")
        st.markdown(
            "Single-step attack.  Fast but weaker.  "
            "Perturbs every pixel by exactly ε in the gradient direction."
        )
    with ac2:
        st.markdown("#### 🔁 BIM")
        st.markdown(
            "Iterative FGSM with ε-ball projection.  "
            "Significantly stronger than FGSM for the same ε."
        )
    with ac3:
        st.markdown("#### 🎯 PGD")
        st.markdown(
            "BIM + random initialisation.  The gold-standard white-box attack.  "
            "Highest fooling rate at any given ε."
        )
