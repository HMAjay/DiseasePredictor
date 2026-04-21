import streamlit as st
import pandas as pd
import joblib
import os
from utils import predict

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models/model.pkl")
encoder_path = os.path.join(BASE_DIR, "models/label_encoder.pkl")
symptoms_path = os.path.join(BASE_DIR, "models/symptoms.pkl")

desc_path = os.path.join(BASE_DIR, "data/raw/symptom_Description.csv")
prec_path = os.path.join(BASE_DIR, "data/raw/symptom_precaution.csv")

# ---------------- LOAD FILES ----------------
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)
symptoms_list = joblib.load(symptoms_path)

desc_df = pd.read_csv(desc_path)
prec_df = pd.read_csv(prec_path)

# Clean metadata columns
desc_df.columns = desc_df.columns.str.strip()
prec_df.columns = prec_df.columns.str.strip()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Disease Predictor", layout="wide")

st.title("🩺 Disease Prediction System")
st.write("Select symptoms to predict possible diseases")

# ---------------- INPUT ----------------
selected_symptoms = st.multiselect(
    "Select Symptoms",
    sorted(symptoms_list)
)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    if not selected_symptoms:
        st.warning("Please select at least one symptom")
    else:
        results = predict(
            selected_symptoms,
            symptoms_list,
            model,
            label_encoder
        )

        st.subheader("Top Predictions")

        for disease, prob in results:

            st.markdown(f"### {disease}")
            st.progress(float(prob))
            st.write(f"Confidence: {prob * 100:.2f}%")

            # ---------------- DESCRIPTION ----------------
            try:
                description = desc_df[
                    desc_df["Disease"] == disease
                ]["Description"].values[0]

                with st.expander("Description"):
                    st.write(description)
            except:
                st.write("No description available")

            # ---------------- PRECAUTIONS ----------------
            try:
                precautions = prec_df[
                    prec_df["Disease"] == disease
                ].values[0][1:]

                with st.expander("Precautions"):
                    for p in precautions:
                        if pd.notna(p):
                            st.write(f"- {p}")
            except:
                st.write("No precautions available")

            st.markdown("---")

# ---------------- FOOTER ----------------
st.info("This is not a medical diagnosis. Consult a doctor.")