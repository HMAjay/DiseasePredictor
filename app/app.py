import streamlit as st
import joblib
import pandas as pd
import requests
from pathlib import Path
from streamlit_lottie import st_lottie
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from style import set_custom_style, card
from utils import predict_and_explain

# ✅ MUST BE FIRST
st.set_page_config(
    page_title="HealthAI | Smart Disease Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Styling
set_custom_style()

# ---------------- PATH ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw"

# ---------------- ASSETS ----------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


lottie_health = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_5njp3v83.json") # Medical animation
lottie_scanning = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_698wi08a.json") # Scanning animation

# ---------------- LOAD RESOURCES ----------------
@st.cache_resource
def load_resources():
    try:
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
            model, rf_model, label_encoder, symptoms_list,
            scores, best_model, desc_df, prec_df
        )
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None

resources = load_resources()
if resources:
    (model, rf_model, label_encoder, symptoms_list, 
     scores, best_model, desc_df, prec_df) = resources

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image(str(BASE_DIR / "app" / "hero.png"), use_column_width=True)
    st.markdown("<h1 style='text-align: center; color: #64FFDA;'>HealthAI</h1>", unsafe_allow_html=True)


    # st_lottie(lottie_health, height=150, key="sidebar_lottie")

    
    add_vertical_space(2)
    st.markdown("### 📊 Engine Performance")
    for name, score in scores.items():
        st.caption(f"{name}")
        st.progress(score)
    
    add_vertical_space(2)
    st.info("💡 **Tip:** Select at least 3-4 symptoms for better accuracy.")

# ---------------- MAIN UI ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1 style='margin-bottom: 0;'>🩺 Smart Disease Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892B0; font-size: 1.2rem;'>Advanced explainable AI for preliminary health screening.</p>", unsafe_allow_html=True)

with col2:
    if "predicted" not in st.session_state or not st.session_state.predicted:
        if lottie_health:
            st_lottie(lottie_health, height=200, key="main_lottie")


add_vertical_space(2)

# ---------------- INPUT SECTION ----------------
st.markdown("### 🔍 Start Your Assessment")

# ---------------- INPUT SECTION ----------------
st.markdown("### 🔍 Start Your Assessment")

tab1, tab2 = st.tabs(["🎯 Smart Select", "💬 Describe Symptoms"])

with tab1:
    selected_select = st.multiselect(
        "Which symptoms are you experiencing?",
        options=sorted(symptoms_list),
        help="Search and select multiple symptoms",
        key="select_symptoms"
    )

with tab2:
    nl_input = st.text_area(
        "Describe how you're feeling...",
        placeholder="e.g., I have a bad headache, high fever, and I feel nauseous.",
        help="Our AI will try to extract symptoms from your description."
    )
    
    selected_nl = []
    if nl_input:
        from utils import get_best_matches
        words = nl_input.lower().replace(",", " ").replace(".", " ").split()
        for word in words:
            if len(word) > 3:
                matches = get_best_matches(word, symptoms_list, limit=1)
                if matches:
                    selected_nl.append(matches[0])
        selected_nl = list(set(selected_nl))
        if selected_nl:
            st.success(f"Detected symptoms: {', '.join([s.replace('_', ' ').title() for s in selected_nl])}")

# Combine selections
selected = list(set(selected_select + selected_nl))

if selected:
    st.markdown("#### 📏 Severity Analysis")
    st.caption("How intense are these symptoms? (1: Low, 3: High)")
    
    # Grid for severity
    cols = st.columns(3)
    severity_input = {}
    for i, s in enumerate(selected):
        with cols[i % 3]:
            # Try to get existing value from session state if any
            default_val = st.session_state.get(f"sev_{s}", 1)
            severity_input[s] = st.select_slider(
                f"{s.replace('_', ' ').title()}",
                options=[1, 2, 3],
                value=default_val,
                key=f"sev_{s}_slider"
            )

add_vertical_space(1)
predict_btn = st.button("Analyze Symptoms →")




# ---------------- LOGIC ----------------
if predict_btn:
    if not selected:
        st.warning("⚠️ Please select at least one symptom to begin.")
    else:
        with st.status("🧬 Processing Medical Data...", expanded=True) as status:
            st.write("Encoding symptoms...")
            # Predict
            results, explanation = predict_and_explain(
                severity_input, symptoms_list, model, rf_model, label_encoder
            )
            st.write("Analyzing patterns with Ensemble Models...")
            st.write("Generating explainable insights...")
            
            st.session_state.predicted = True
            st.session_state.results = results
            st.session_state.explanation = explanation
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

# ---------------- RESULTS SECTION ----------------
if st.session_state.get("predicted"):
    results = st.session_state.results
    explanation = st.session_state.explanation
    top_disease = results[0][0]
    
    add_vertical_space(2)
    colored_header("🔍 Diagnostic Insights", color_name="blue-70", description="Top 3 likely conditions based on your symptoms")
    
    # Top 3 Results Cards
    res_cols = st.columns(3)
    for i, (disease, prob) in enumerate(results):
        with res_cols[i]:
            card(
                disease, 
                f"Confidence: **{prob*100:.1f}%**",
                icon="🩺" if i == 0 else "⚠️"
            )
            st.progress(float(prob))

    add_vertical_space(2)
    
    # Detailed Info
    detail_col1, detail_col2 = st.columns([1, 1])
    
    with detail_col1:
        st.markdown(f"### 📖 About {top_disease}")
        try:
            desc = desc_df.loc[desc_df["Disease"] == top_disease, "Description"].values[0]
            st.markdown(f"<div class='stCard'>{desc}</div>", unsafe_allow_html=True)
        except:
            st.warning("No detailed description available.")

        add_vertical_space(1)
        st.markdown("### 💊 Recommended Precautions")
        try:
            precautions = prec_df.loc[prec_df["Disease"] == top_disease].values[0][1:]
            prec_html = "<ul>"
            for p in precautions:
                if pd.notna(p):
                    prec_html += f"<li>{p.capitalize()}</li>"
            prec_html += "</ul>"
            st.markdown(f"<div class='stCard'>{prec_html}</div>", unsafe_allow_html=True)
        except:
            st.warning("No precautions available.")

    with detail_col2:
        st.markdown("### 🧠 AI Explanation")
        st.caption("Which symptoms contributed most to this prediction?")
        if explanation:
            # Create a small dataframe for the chart
            exp_df = pd.DataFrame(explanation, columns=["Symptom", "Impact"])
            st.bar_chart(exp_df.set_index("Symptom"))
            
            for s, score in explanation:
                st.markdown(f"**{s.title()}**")
                st.progress(min(score * 10, 1.0)) # Scale for visibility
        else:
            st.info("No single symptom was dominant in this prediction.")

    # ---------------- REPORT DOWNLOAD ----------------
    add_vertical_space(2)
    report_text = f"HealthAI Assessment Report\n{'='*30}\n"
    report_text += f"Primary Condition: {top_disease}\n"
    report_text += f"Confidence: {results[0][1]*100:.2f}%\n\n"
    report_text += "Symptoms Analyzed:\n"
    for s, v in severity_input.items():
        report_text += f"- {s}: Severity {v}\n"
    
    st.download_button(
        label="📥 Download Assessment Report",
        data=report_text,
        file_name=f"health_report_{top_disease.lower().replace(' ', '_')}.txt",
        mime="text/plain",
    )

# ---------------- FOOTER ----------------
st.markdown("""
    <div class="footer">
        <p>⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
        <p>&copy; 2026 HealthAI Systems | Built with Premium Precision</p>
    </div>
""", unsafe_allow_html=True)