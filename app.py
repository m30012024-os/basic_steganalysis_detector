import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
from PIL import Image
from stegano import lsb

# ---------- Settings (must match the notebook) ----------
MODEL_PATH = Path(__file__).parent / "stego_detector_rf.joblib"
SIZE = (256, 256)
TOP_ROWS = 32
STAT_NAMES = ["mean", "std", "ones_zeros_ratio", "adj_diff"]
COLUMNS = [f"{ch}_{region}_{stat}" for region in ["full", "top"]
           for ch in "RGB" for stat in STAT_NAMES]


# ---------- Load the trained model once ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ---------- Same feature code as the notebook ----------
def lsb_stats(plane):
    """Four simple stats for one 2D array of 0s and 1s."""
    ones = plane.sum()
    zeros = plane.size - ones
    plane = plane.astype(np.int8)
    adj_diff = np.abs(plane[:, 1:] - plane[:, :-1]).mean()
    return [plane.mean(), plane.std(), ones / max(zeros, 1), adj_diff]


def extract_features(img):
    """img is a PIL image. Returns a one-row DataFrame with the 24 features."""
    arr = np.array(img.convert("RGB").resize(SIZE, Image.LANCZOS))  # same size as training
    lsb_plane = arr & 1
    feats = []
    for region in [lsb_plane, lsb_plane[:TOP_ROWS]]:
        for c in range(3):
            feats += lsb_stats(region[:, :, c])
    return pd.DataFrame([feats], columns=COLUMNS)


# ---------- Page layout ----------
st.set_page_config(page_title="Steganalysis Detector", page_icon="🕵️")
st.title("🕵️ Steganalysis Detector")
st.write("Upload a **PNG** image to check whether it hides a secret text message.")
st.caption("Use PNG only: JPEG compression destroys hidden LSB data.")

uploaded = st.file_uploader("Choose a PNG image", type=["png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    # # 1) Machine-learning prediction
    # model = load_model()
    # features = extract_features(image)
    # prediction = model.predict(features)[0]
    # prob_stego = model.predict_proba(features)[0][1]

    # st.subheader("Model verdict")
    # if prediction == 1:
    #     st.error(f"⚠️ Likely contains hidden data (confidence: {prob_stego:.0%})")
    # else:
    #     st.success(f"✅ Likely clean (stego probability: {prob_stego:.0%})")

    # # 2) Try to reveal a hidden message from the ORIGINAL upload (never the resized copy)
    # st.subheader("Hidden message")
    # try:
    #     uploaded.seek(0)                      # rewind the file before reading it again
    #     message = lsb.reveal(uploaded)
    # except Exception:
    #     message = None                        # stegano raises errors on images with no message

    # if message:
    #     st.warning("A hidden message was found:")
    #     st.code(message, language=None)
    # else:
    #     st.info("No readable hidden message was found with stegano.lsb.")
    # 1) Try to reveal a hidden message from the ORIGINAL upload first
    try:
        uploaded.seek(0)                      # rewind the file before reading it
        message = lsb.reveal(uploaded)
    except Exception:
        message = None                        # no stegano message found

    # 2) Machine-learning prediction (statistical guess)
    model = load_model()
    features = extract_features(image)
    prob_stego = model.predict_proba(features)[0][1]

    # 3) Final verdict: a decoded message always wins over the model
    st.subheader("Verdict")
    if message:
        st.error("⚠️ STEGO IMAGE: a hidden message was extracted (confirmed).")
    elif prob_stego >= 0.5:
        st.error(f"⚠️ Likely contains hidden data (model confidence: {prob_stego:.0%}). "
                 "No readable stegano.lsb message was found.")
    else:
        st.success(f"✅ Likely clean (model stego probability: {prob_stego:.0%})")

    st.caption(f"Model's stego probability: {prob_stego:.0%}")

    st.subheader("Hidden message")
    if message:
        st.code(message, language=None)
    else:
        st.info("No readable hidden message was found with stegano.lsb.")