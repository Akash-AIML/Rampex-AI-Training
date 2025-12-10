import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import joblib
import os

# -------------------- BASIC UI FIRST --------------------
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🐾 Animal Classification (CLIP + ML)")
st.write("Upload an animal image, choose a model, and predict.")

# -------------------- CONFIG --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "saved_models"

LOGREG_PATH = os.path.join(MODELS_DIR, "logreg_clip_animals.pkl")
KNN_PATH    = os.path.join(MODELS_DIR, "knn_clip_animals.pkl")
NB_PATH     = os.path.join(MODELS_DIR, "nb_clip_animals.pkl")
CLASSES_PATH = os.path.join(MODELS_DIR, "class_names.pkl")

# -------------------- SAFE LOADERS --------------------
@st.cache_resource(show_spinner=False)
def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    return model, preprocess

@st.cache_resource(show_spinner=False)
def load_models():
    return (
        joblib.load(LOGREG_PATH),
        joblib.load(KNN_PATH),
        joblib.load(NB_PATH),
        joblib.load(CLASSES_PATH),
    )

# -------------------- TRY LOADING --------------------
try:
    model, preprocess = load_clip()
    log_reg, knn, nb, class_names = load_models()
except Exception as e:
    st.error("❌ Failed to load model or files")
    st.exception(e)
    st.stop()

# -------------------- HELPERS --------------------
def get_embedding(image: Image.Image) -> np.ndarray:
    with torch.no_grad():
        img = preprocess(image).unsqueeze(0).to(DEVICE)
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

def predict(embedding, model_name):
    if model_name == "Logistic Regression":
        clf = log_reg
    elif model_name == "KNN":
        clf = knn
    else:
        clf = nb

    idx = int(clf.predict(embedding)[0])
    return class_names[idx]

# -------------------- UI --------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

model_choice = st.selectbox(
    "Select a model",
    ["Logistic Regression", "KNN", "Naive Bayes"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Running prediction..."):
            emb = get_embedding(image)
            pred = predict(emb, model_choice)

        st.success(f"✅ Prediction: **{pred}**")
