import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Helmet Detection",
    page_icon="🪖",
    layout="centered"
)

# ── Parameter HOG (harus sama persis dengan saat training) ───
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

IMG_SIZE = (128, 128)
MODEL_PATH = "model/svm_helmet_model_final.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_helmet(image_bgr, model):
    image_resized = cv2.resize(image_bgr, IMG_SIZE)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray, **HOG_PARAMS).reshape(1, -1)

    pred = model.predict(features)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]

    return pred, proba


# ── UI ─────────────────────────────────────────────────────────
st.title("🪖 Helmet Detection")
st.write(
    "Upload gambar untuk mendeteksi apakah orang dalam gambar "
    "menggunakan helm atau tidak. Model menggunakan fitur **HOG** "
    "dan classifier **SVM**."
)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_image, caption="Gambar diupload", use_container_width=True)

    pred, proba = predict_helmet(image_bgr, model)
    label = "With Helmet" if pred == 1 else "No Helmet"

    with col2:
        st.subheader("Hasil Prediksi")
        if pred == 1:
            st.success(f"✅ {label}")
        else:
            st.error(f"⚠️ {label}")

        if proba is not None:
            st.write("**Confidence:**")
            st.write(f"- Helmet: {proba[1] * 100:.1f}%")
            st.write(f"- No Helmet: {proba[0] * 100:.1f}%")
        else:
            st.caption(
                "Model dilatih tanpa `probability=True`, "
                "jadi confidence score tidak tersedia."
            )
else:
    st.info("⬆️ Upload gambar untuk mulai deteksi.")

st.markdown("---")
st.caption("Model: SVM + HOG Feature Extraction + SMOTE Balancing")