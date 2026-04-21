import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from utils import predict_and_explain

# ✅ MUST BE FIRST
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# ---------------- PATH ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw"

# ---------------- LOAD ----------------
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_DIR / "model.pkl")
    rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    symptoms_list = joblib.load(MODEL_DIR / "symptoms.pkl")
    scores = joblib.load(MODEL_DIR / "model_scores.pkl")
    best_model = joblib.load(MODEL_DIR / "best_model.pkl")

    desc_df = pd.read_csv(DATA_DIR / "symptom_Description.csv")
    prec_df = pd.read_csv(DATA_DIR / "symptom_precaution.csv")

    desc_df.columns = desc_df.columns.str.strip()
    prec_df.columns = prec_df.columns.str.strip()

    return (
        model,
        rf_model,
        label_encoder,
        symptoms_list,
        scores,
        best_model,
        desc_df,
        prec_df,
    )

(
    model,
    rf_model,
    label_encoder,
    symptoms_list,
    scores,
    best_model,
    desc_df,
    prec_df,
) = load_resources()

# ---------------- STATE ----------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.results = None
    st.session_state.explanation = None

# ---------------- UI ----------------
st.title("🩺 Explainable Disease Prediction System")

# ---------------- MODEL INFO (HIDDEN) ----------------
with st.expander("📊 View Model Performance"):
    for name, score in scores.items():
        st.write(f"{name}: {score:.4f}")

    st.success(f"🏆 Best Model: {best_model}")

# ---------------- INPUT ----------------
selected = st.multiselect("Select Symptoms", sorted(symptoms_list))

severity_input = {}
for s in selected:
    severity_input[s] = st.slider(f"Severity of {s}", 1, 3, 1)

# ---------------- PREDICT ----------------
if st.button("Predict"):

    if not selected:
        st.warning("⚠️ Please select symptoms")
    else:
        with st.spinner("🔍 Predicting..."):

            results, explanation = predict_and_explain(
                severity_input,
                symptoms_list,
                model,
                rf_model,
                label_encoder,
            )

            # save to session
            st.session_state.predicted = True
            st.session_state.results = results
            st.session_state.explanation = explanation

# ---------------- SHOW RESULTS ONLY AFTER CLICK ----------------
if st.session_state.predicted:

    results = st.session_state.results
    explanation = st.session_state.explanation

    st.subheader("🔍 Predictions")

    for disease, prob in results:
        st.write(f"### {disease}")
        st.progress(float(prob))
        st.write(f"{prob * 100:.2f}% confidence")

    # ---------------- EXPLAIN ----------------
    st.subheader("🧠 Why this prediction?")

    if explanation:
        for s, score in explanation:
            st.write(f"- {s} → contribution: {score:.3f}")
    else:
        st.write("No strong contributing symptoms.")

    # ---------------- DETAILS ----------------
    top = results[0][0]

    try:
        desc = desc_df.loc[
            desc_df["Disease"] == top, "Description"
        ].values[0]

        st.subheader("📖 Description")
        st.write(desc)
    except:
        st.warning("No description available")

    try:
        precautions = prec_df.loc[
            prec_df["Disease"] == top
        ].values[0][1:]

        st.subheader("💊 Precautions")

        for p in precautions:
            if pd.notna(p):
                st.write(f"- {p}")
    except:
        st.warning("No precautions available")

# ---------------- FOOTER ----------------
st.info("⚠️ This is not a medical diagnosis. Consult a professional.")