from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "plant_disease_mobilenetv2.keras"
CLASSES_PATH = ROOT / "reports" / "class_names.json"
IMAGE_SIZE = (160, 160)

st.set_page_config(
    page_title="Tomato Disease AI",
    page_icon="🍅",
    layout="centered",
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def readable_name(label):
    name = label.replace("Tomato___", "")
    return name.replace("_", " ").replace("  ", " ")

st.title("🍅 Tomato Disease Classification")
st.write(
    "Upload a tomato leaf image to get an AI-generated "
    "disease classification prediction."
)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("Model or class names file is missing. Check project files.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png", "jfif", "webp"],
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf image", width="stretch")

        with st.spinner("Analyzing image..."):
            model = load_model()
            class_names = load_classes()

            resized = image.resize(IMAGE_SIZE)
            array = np.asarray(resized, dtype=np.float32)
            array = np.expand_dims(array, axis=0)

            probabilities = model.predict(array, verbose=0)[0]
            top_indices = np.argsort(probabilities)[::-1][:3]
            best_index = int(top_indices[0])
            confidence = float(probabilities[best_index])

        st.subheader("Prediction")
        st.write(f"**Class:** {readable_name(class_names[best_index])}")
        st.metric("Model confidence", f"{confidence:.1%}")

        st.subheader("Top 3 predictions")
        for index in top_indices:
            label = readable_name(class_names[int(index)])
            score = float(probabilities[index])
            st.write(f"{label} — {score:.1%}")
            st.progress(score)

        st.warning(
            "This is an experimental image classifier trained on PlantVillage "
            "images. Predictions may be wrong on real-world photos and should "
            "not be treated as a definitive plant-health diagnosis."
        )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

