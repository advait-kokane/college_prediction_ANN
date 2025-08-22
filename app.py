import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ------------------ Page setup ------------------
st.set_page_config(page_title="College Admission Predictor", page_icon="🎓", layout="centered")

CUSTOM_CSS = """
<style>
:root {
    --deep-navy: #0a192f;
    --cool-teal: #64ffda;
    --light-slate: #ccd6f6;
    --slate: #8892b0;
    --dark-slate: #495670;
    --accent: #1e90ff;
    --gradient: linear-gradient(135deg, #64ffda 0%, #1e90ff 100%);
    --highlight: rgba(100, 255, 218, 0.2);
}

/* Main app styling */
.stApp {
    background: linear-gradient(135deg, var(--deep-navy), #0d2142) !important;
    color: var(--light-slate) !important;
    padding: 2rem !important;
}

/* Title styling */
h1 {
    text-align: center !important;
    margin: 0 auto 1.5rem !important;
    font-size: 2.5rem !important;
    background: var(--gradient) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    color: transparent !important;
    position: relative !important;
    padding-bottom: 1rem !important;
    width: fit-content !important;
}

h1::after {
    content: '' !important;
    position: absolute !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 120px !important;
    height: 3px !important;
    background: var(--gradient) !important;
    border-radius: 3px !important;
}

/* Card container */
div[data-testid="stBlock"] {
    background: rgba(10, 25, 47, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
    padding: 2.5rem !important;
    border: 1px solid rgba(100, 255, 218, 0.15) !important;
    box-shadow: 0 8px 32px rgba(2, 12, 27, 0.5) !important;
    margin: 0 auto 2rem !important;
    max-width: 800px !important;
}

/* Form elements */
.stSelectbox, .stSlider, .stTextInput, .stNumberInput {
    margin-bottom: 1.5rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 0 rgba(100,255,218,0) !important;
    border-radius: 8px !important;
}

/* Input focus effect */
.stSelectbox:focus-within, .stSlider:focus-within, .stTextInput:focus-within, .stNumberInput:focus-within {
    box-shadow: 0 0 8px var(--cool-teal) !important;
}

/* Label styling */
label {
    font-size: 1rem !important;
    color: var(--cool-teal) !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
    font-weight: 500 !important;
}

/* Custom slider styling */
div[data-testid="stSlider"] > div {
    padding: 0.8rem 1rem !important;
    background: rgba(10, 25, 47, 0.6) !important;
    border-radius: 10px !important;
    border: 1px solid var(--dark-slate) !important;
}

div[data-testid="stThumbValue"] {
    color: var(--cool-teal) !important;
    font-weight: 600 !important;
}

div[data-testid="stThumbValue"] + div > div {
    background: var(--dark-slate) !important;
    height: 6px !important;
}

div[data-testid="stThumbValue"] + div > div > div {
    background: var(--gradient) !important;
    border: none !important;
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
}

/* Prediction button */
.stButton > button {
    width: 100% !important;
    padding: 1.2rem !important;
    font-size: 1.2rem !important;
    background: var(--gradient) !important;
    color: var(--deep-navy) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(100, 255, 218, 0.4) !important;
    margin: 1.5rem 0 0 !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(100, 255, 218, 0.6) !important;
}

/* Results Table */
.result-table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1.5rem 0 !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35) !important;
    background: rgba(10, 25, 47, 0.65) !important;
    backdrop-filter: blur(10px) !important;
}

.result-table th {
    background: var(--gradient) !important;
    color: var(--deep-navy) !important;
    text-align: center !important;
    padding: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}

.result-table td {
    padding: 12px !important;
    text-align: center !important;
    border-bottom: 1px solid rgba(100, 255, 218, 0.15) !important;
    color: var(--light-slate) !important;
    font-weight: 500 !important;
}

.result-table tr:hover td {
    background: rgba(100, 255, 218, 0.08) !important;
    color: var(--cool-teal) !important;
    transition: all 0.3s ease !important;
}

.result-table td:first-child {
    font-weight: 700 !important;
    color: var(--cool-teal) !important;
}

/* Fixed footer note */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
    background: rgba(10, 25, 47, 0.95);
    color: #e6f1ff;
    text-align: center;
    padding: 8px 15px;
    font-size: 0.9rem;
    border-top: 1px solid rgba(100, 255, 218, 0.3);
    z-index: 9999;
}
.footer strong {
    color: #64ffda;
}
"""


</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🎓 College Admission Predictor")

# ------------------ Load artifacts ------------------
def fail(msg, e):
    st.error(f"{msg}: {e}")
    st.stop()

try:
    model = load_model("ann_model.h5")
except Exception as e:
    fail("❌ Failed to load ANN model", e)

try:
    with open("encoders.pkl","rb") as f:
        encoders = pickle.load(f)
except Exception as e:
    fail("❌ Failed to load encoders.pkl", e)

try:
    df_original = pd.read_csv("ALL COLLEGE.csv")
except Exception as e:
    fail("❌ Failed to load ALL COLLEGE.csv", e)

# ------------------ Encode copy for model input ------------------
df_encoded = df_original.copy()
categorical_cols = ['College','Branch','Category','Gender','Region','Technical/Non Technical','Round','Location']
for col in categorical_cols:
    df_encoded[col] = encoders[col].transform(df_original[col])

X_COLUMNS = [
    'College','Branch','Category','Gender','Region',
    'Technical/Non Technical','Round','Location','Student_Marks'
]

# ------------------ UI ------------------
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        branch  = st.selectbox("Branch",   sorted(df_original['Branch'].unique()))
        category= st.selectbox("Category", sorted(df_original['Category'].unique()))
        gender  = st.selectbox("Gender",   sorted(df_original['Gender'].unique()))
        region  = st.selectbox("Region",   sorted(df_original['Region'].unique()))
    with c2:
        location= st.selectbox("Location", sorted(df_original['Location'].unique()))
        tech    = st.selectbox("Technical / Non-Technical", sorted(df_original['Technical/Non Technical'].unique()))
        round_  = st.selectbox("Round",    sorted(df_original['Round'].unique()))
        marks   = st.number_input("Your Marks (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)

# ------------------ Predict ------------------
if st.button("🔍 Predict Possible Colleges"):
    filtered_df = df_original[
        (df_original['Branch'] == branch) &
        (df_original['Category'] == category) &
        (df_original['Gender'] == gender) &
        (df_original['Region'] == region) &
        (df_original['Location'] == location) &
        (df_original['Technical/Non Technical'] == tech) &
        (df_original['Round'] == round_)
    ]

    st.write(f"✅ Found {len(filtered_df)} colleges matching your criteria")

    if filtered_df.empty:
        st.warning("Try changing filters.")
        st.stop()

    college_best = {}

    for idx, row in filtered_df.iterrows():
        encoded_row = df_encoded.iloc[idx].copy()
        encoded_row['Student_Marks'] = marks  

        try:
            features = encoded_row.reindex(X_COLUMNS).astype(float).values.reshape(1, -1)
        except KeyError as e:
            fail("Feature mismatch (training vs app). Missing column", e)

        prob = float(model.predict(features, verbose=0)[0][0])
        cutoff = float(row['Cutoff %'])  

        if marks >= cutoff:
            college_name = row['College']
            prob_pct = prob * 100.0
            if college_name in college_best:
                college_best[college_name] = max(college_best[college_name], prob_pct)
            else:
                college_best[college_name] = prob_pct

    if not college_best:
        st.error("❌ No colleges qualify with your marks. ")
        st.error("Cutoff is higher than your marks for this requirement. ")
        st.stop()

    items = sorted(college_best.items(), key=lambda x: -x[1])
    results_df = pd.DataFrame(
        [(i+1, name, f"{score:.2f}%") for i, (name, score) in enumerate(items)],
        columns=["Rank","College","Admission Chance"]
    )

    # ✅ Show full list directly (no top5 / no expander / no badge)
    st.markdown("### 🎯 Recommended Colleges")
    st.markdown(results_df.to_html(index=False, classes="result-table"), unsafe_allow_html=True)

# Permanent fixed footer
st.markdown(
    """
    <div class="footer">
        🔔 <strong>Note:</strong> This prediction is only a guide. Final admission depends on the college. 
        Sometimes seats remain after CAP rounds, and colleges may allot them differently. 
        Always confirm with the college before making a decision. 🎓
    </div>
    """,
    unsafe_allow_html=True
)





