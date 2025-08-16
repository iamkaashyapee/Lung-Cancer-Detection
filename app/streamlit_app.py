import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Lung Cancer Survival Predictor", page_icon="🫁", layout="centered")

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "lung_cancer_survival_model.joblib"
    return joblib.load(model_path)

st.title("🫁 Lung Cancer Survival Predictor")
st.caption("Demo app: enter patient details to estimate survival probability.")

pipe = load_model()

with st.form("input"):
    age = st.number_input("Age", min_value=0, max_value=120, value=60)
    gender = st.selectbox("Gender", ["Male","Female"])
    country = st.text_input("Country", "India")
    cancer_stage = st.selectbox("Cancer Stage", ["Stage I","Stage II","Stage III","Stage IV"])
    family_history = st.selectbox("Family History", ["Yes","No"])
    smoking_status = st.selectbox("Smoking Status", ["Never Smoked","Former Smoker","Current Smoker","Passive Smoker"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    cholesterol_level = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    hypertension = st.selectbox("Hypertension", [0,1])
    asthma = st.selectbox("Asthma", [0,1])
    cirrhosis = st.selectbox("Cirrhosis", [0,1])
    other_cancer = st.selectbox("Other Cancer", [0,1])
    diagnosis_year = st.number_input("Diagnosis Year", min_value=1990, max_value=2030, value=2020)
    diagnosis_month = st.number_input("Diagnosis Month", min_value=1, max_value=12, value=1)
    treatment_year = st.number_input("End of Treatment Year", min_value=1990, max_value=2030, value=2021)
    treatment_month = st.number_input("End of Treatment Month", min_value=1, max_value=12, value=6)
    treatment_duration_days = st.number_input("Treatment Duration (days)", min_value=0, max_value=5000, value=365)
    treatment_type = st.selectbox("Treatment Type", ["Chemotherapy", "Surgery", "Combined", "Radiation"])

    submitted = st.form_submit_button("Predict")

if submitted:
    features = {
        "age": age,
        "gender": gender,
        "country": country,
        "cancer_stage": cancer_stage,
        "family_history": family_history,
        "smoking_status": smoking_status,
        "bmi": bmi,
        "cholesterol_level": cholesterol_level,
        "hypertension": hypertension,
        "asthma": asthma,
        "cirrhosis": cirrhosis,
        "other_cancer": other_cancer,
        "diagnosis_year": diagnosis_year,
        "diagnosis_month": diagnosis_month,
        "treatment_year": treatment_year,
        "treatment_month": treatment_month,
        "treatment_duration_days": treatment_duration_days,
        "treatment_type": treatment_type,
    }
    X = pd.DataFrame([features])
    proba = pipe.predict_proba(X)[0,1] if hasattr(pipe, "predict_proba") else None
    pred = pipe.predict(X)[0]
    st.subheader("Result")
    if proba is not None:
        st.metric("Survival probability", f"{proba:.2%}")
    st.write("Predicted outcome:", "Survived" if pred==1 else "Not Survived")
