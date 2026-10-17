"""Streamlit front-end that calls the Flask /predict endpoint."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:7201")

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️")
st.title("Heart Disease Risk Predictor")
st.caption(f"Calls `POST {API_URL}/predict`")

CP = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
RESTECG = {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}
SLOPE = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}

with st.form("patient"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 1, 120, 55)
    sex = c2.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v else "Female")
    trestbps = c1.number_input("Resting blood pressure (mm Hg)", 60, 250, 130)
    chol = c2.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    thalach = c1.number_input("Max heart rate achieved", 60, 220, 150)
    oldpeak = c2.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    cp = c1.selectbox("Chest pain type", list(CP), format_func=CP.get)
    restecg = c2.selectbox("Resting ECG", list(RESTECG), format_func=RESTECG.get)
    slope = c1.selectbox("ST slope", list(SLOPE), format_func=SLOPE.get)
    thal = c2.selectbox("Thalassemia", list(THAL), format_func=THAL.get)
    ca = c1.selectbox("Major vessels coloured (0-3)", [0, 1, 2, 3])
    fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                       format_func=lambda v: "Yes" if v else "No")
    exang = c1.selectbox("Exercise-induced angina", [0, 1],
                         format_func=lambda v: "Yes" if v else "No")
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                   restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
                   slope=slope, ca=ca, thal=thal)
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        out = resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
    else:
        render = st.error if out["prediction"] == 1 else st.success
        render(f"**{out['label']}** (confidence {out['confidence']:.1%})")
        st.progress(out["probability_disease"],
                    text=f"Probability of disease: {out['probability_disease']:.1%}")
        with st.expander("Request / response"):
            st.json({"request": payload, "response": out})
