# ============================================================
#   DIABETES PREDICTION SYSTEM — STREAMLIT WEB APP
#   IILM University, Greater Noida
#   Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_model():
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 10px 0 5px 0;'>
    <h1 style='color:#1565C0; font-size:2rem;'>🩺 Diabetes Prediction System</h1>
    <p style='color:#555; font-size:1rem;'>Using Support Vector Machine (SVM) — Machine Learning</p>
    <p style='color:#888; font-size:0.85rem;'>IILM University, Greater Noida &nbsp;|&nbsp; Guide: Ms. Pushpa Singh</p>
</div>
<hr style='border:1px solid #ddd; margin-bottom:20px;'>
""", unsafe_allow_html=True)

st.markdown("### 📋 Enter Patient Health Details")
st.markdown("Fill in the values below and click **Predict** to check diabetes risk.")

# ── Input Form ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1,
        help="Number of times pregnant"
    )
    glucose = st.number_input(
        "Glucose (mg/dL)", min_value=0, max_value=300, value=110,
        help="Plasma glucose concentration (2-hour oral glucose tolerance test)"
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)", min_value=0, max_value=150, value=72,
        help="Diastolic blood pressure"
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)", min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    insulin = st.number_input(
        "Insulin (µU/mL)", min_value=0, max_value=900, value=80,
        help="2-Hour serum insulin"
    )
    bmi = st.number_input(
        "BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        help="Body Mass Index (weight in kg / height in m²)"
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
        value=0.5, step=0.01,
        help="A function that scores likelihood of diabetes based on family history"
    )
    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=30,
        help="Age of the patient"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── BMI Indicator ─────────────────────────────────────────────
bmi_category = ""
bmi_color = ""
if bmi < 18.5:
    bmi_category, bmi_color = "Underweight", "#FF9800"
elif bmi < 25:
    bmi_category, bmi_color = "Normal", "#4CAF50"
elif bmi < 30:
    bmi_category, bmi_color = "Overweight", "#FF9800"
else:
    bmi_category, bmi_color = "Obese", "#F44336"

st.markdown(f"**BMI Category:** <span style='color:{bmi_color}; font-weight:bold;'>{bmi_category}</span>",
            unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary")

if predict_btn:
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.markdown("### 🔬 Prediction Result")

    if prediction == 1:
        risk_pct = probability[1] * 100
        st.markdown(f"""
        <div style='background:#FFEBEE; border-left:5px solid #F44336;
                    padding:20px; border-radius:8px; margin:10px 0;'>
            <h2 style='color:#C62828; margin:0;'>⚠️ HIGH RISK — Diabetic</h2>
            <p style='color:#B71C1C; margin:8px 0 0 0; font-size:1.1rem;'>
                Confidence: <b>{risk_pct:.1f}%</b>
            </p>
            <p style='color:#555; margin:8px 0 0 0;'>
                This patient shows a high likelihood of diabetes based on the provided parameters.
                Please consult a medical professional immediately for proper diagnosis and treatment.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        safe_pct = probability[0] * 100
        st.markdown(f"""
        <div style='background:#E8F5E9; border-left:5px solid #4CAF50;
                    padding:20px; border-radius:8px; margin:10px 0;'>
            <h2 style='color:#1B5E20; margin:0;'>✅ LOW RISK — Non-Diabetic</h2>
            <p style='color:#2E7D32; margin:8px 0 0 0; font-size:1.1rem;'>
                Confidence: <b>{safe_pct:.1f}%</b>
            </p>
            <p style='color:#555; margin:8px 0 0 0;'>
                This patient shows a low likelihood of diabetes based on the provided parameters.
                Maintain a healthy lifestyle with regular check-ups.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Input summary table
    st.markdown("#### 📊 Input Summary")
    import pandas as pd
    summary = pd.DataFrame({
        'Parameter': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                      'Insulin', 'BMI', 'Diabetes Pedigree Fn', 'Age'],
        'Value': [pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age],
        'Unit': ['count', 'mg/dL', 'mm Hg', 'mm', 'µU/mL', 'kg/m²', '-', 'years']
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#999; font-size:0.8rem; padding: 10px 0;'>
    ⚠️ <i>This tool is for educational purposes only. It does not replace professional medical advice.</i><br><br>
    <b>Team:</b> Adyasha Das | Khushi Rana | Laxmi Bhanti | Anand Kumar Jha | Vanshika Singh<br>
    <b>Guide:</b> Ms. Pushpa Singh &nbsp;|&nbsp; <b>Project ID:</b> BTP2CSE163<br>
    IILM University, Greater Noida — 2026
</div>
""", unsafe_allow_html=True)