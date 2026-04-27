import streamlit as st

def set_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0A192F 0%, #112240 100%);
        color: #E6F1FF;
    }

    /* Glassmorphism Card */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }

    .stCard:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(100, 255, 218, 0.3);
    }

    /* Modern Headers */
    h1, h2, h3 {
        color: #64FFDA !important;
        font-weight: 600 !important;
    }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #64FFDA 0%, #48D1CC 100%);
        color: #0A192F !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(100, 255, 218, 0.4);
        transform: scale(1.02);
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #112240;
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: #64FFDA;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #64FFDA , #48D1CC);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
    }
    
    /* Aggressively disable default Streamlit fade-out during re-runs */
    [data-testid="stAppViewContainer"], 
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stApp"],
    .main {
        opacity: 1 !important;
        filter: none !important;
    }

    /* Target the loading overlay specifically */
    div[data-testid="stAppViewContainer"] > section {
        opacity: 1 !important;
    }

    /* Footer */
    .footer {
        width: 100%;
        background-color: rgba(10, 25, 47, 0.5);
        color: #4B5563;
        text-align: center;
        padding: 20px;
        font-size: 11px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 50px;
    }

    </style>
    """, unsafe_allow_html=True)

def card(title, content, icon="🩺"):
    st.markdown(f"""
    <div class="stCard">
        <h3 style="margin-top:0;">{icon} {title}</h3>
        <div style="color: #8892B0;">{content}</div>
    </div>
    """, unsafe_allow_html=True)
