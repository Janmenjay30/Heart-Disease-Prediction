import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("models") / "best_pipeline.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.title("Heart Disease Risk Predictor")
    st.write("Enter patient information to predict the probability of heart disease.")

    with st.form("input_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex", options={"Male":1, "Female":0}, index=0)
        cp = st.selectbox("Chest pain type (cp)", options=[0,1,2,3], index=1)
        trestbps = st.number_input("Resting blood pressure (trestbps)", value=130)
        chol = st.number_input("Cholesterol (mg/dl)", value=250)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options={"Yes":1, "No":0}, index=1)
        restecg = st.selectbox("Resting ECG (restecg)", options=[0,1,2], index=0)
        thalach = st.number_input("Max heart rate achieved (thalach)", value=150)
        exang = st.selectbox("Exercise induced angina (exang)", options={"Yes":1, "No":0}, index=1)
        oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", value=1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment (slope)", options=[0,1,2], index=1)
        ca = st.selectbox("Number of major vessels (0-3) colored by flourosopy (ca)", options=[0,1,2,3], index=0)
        thal = st.selectbox("Thalassemia (thal)", options=[3,6,7], index=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        model = load_model()
        X = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }])

        try:
            proba = model.predict_proba(X)[0,1]
        except Exception:
            proba = None
        pred = model.predict(X)[0]

        if proba is not None:
            st.subheader(f"Predicted probability of heart disease: {proba:.2%}")
        st.write(f"Predicted class (0=no disease, 1=disease): {int(pred)}")


if __name__ == "__main__":
    main()
