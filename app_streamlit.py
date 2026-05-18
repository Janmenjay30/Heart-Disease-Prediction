import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

# Ensure src is in python path to import data loader
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data_utils import load_cleveland
import lime
import lime.lime_tabular
import streamlit.components.v1 as components

MODEL_PATH = Path("results") / "cleveland" / "models" / "best_pipeline.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def get_explainer():
    df = load_cleveland(PROJECT_ROOT / "data" / "processed.cleveland.data")
    X = df.drop(columns=["target"])

    # CRITICAL: Drop NaN rows — LIME cannot handle NaN in training data
    X = X.dropna()

    # LIME does: name = int(data_row[i]) then categorical_names[feat_idx][name]
    # So keys MUST be plain Python ints matching actual values in the dataset.
    # Cleveland dataset: cp=1-4 (1-indexed!), slope=1-3 (1-indexed!), thal=3/6/7
    categorical_names = {
        X.columns.get_loc("sex"):     {0: "Female",          1: "Male"},
        X.columns.get_loc("cp"):      {1: "Typical angina",  2: "Atypical angina",
                                       3: "Non-anginal pain", 4: "Asymptomatic"},
        X.columns.get_loc("fbs"):     {0: "No",  1: "Yes"},
        X.columns.get_loc("restecg"): {0: "Normal", 1: "ST-T abnormality",
                                       2: "LV hypertrophy"},
        X.columns.get_loc("exang"):   {0: "No",  1: "Yes"},
        X.columns.get_loc("slope"):   {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
        X.columns.get_loc("ca"):      {0: "0", 1: "1", 2: "2", 3: "3"},
        X.columns.get_loc("thal"):    {3: "Normal", 6: "Fixed defect",
                                       7: "Reversable defect"},
    }
    categorical_feature_indices = list(categorical_names.keys())

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values.astype(float),
        feature_names=X.columns.tolist(),
        class_names=["No Disease", "Disease"],
        categorical_features=categorical_feature_indices,
        categorical_names=categorical_names,
        mode="classification",
        discretize_continuous=True,
    )
    return explainer



def main():
    st.title("Heart Disease Risk Predictor")
    st.write("Enter patient information to predict the probability of heart disease.")

    with st.form("input_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", index=0)
        # cp: Cleveland dataset uses 1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic
        cp = st.selectbox(
            "Chest pain type (cp)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1:"Typical angina",2:"Atypical angina",3:"Non-anginal pain",4:"Asymptomatic"}[x],
            index=1
        )
        trestbps = st.number_input("Resting blood pressure (trestbps)", value=130)
        chol = st.number_input("Cholesterol (mg/dl)", value=250)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        restecg = st.selectbox(
            "Resting ECG (restecg)",
            options=[0, 1, 2],
            format_func=lambda x: {0:"Normal",1:"ST-T wave abnormality",2:"Left ventricular hypertrophy"}[x],
            index=0
        )
        thalach = st.number_input("Max heart rate achieved (thalach)", value=150)
        exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", value=1.0, step=0.1)
        # slope: Cleveland dataset uses 1=Upsloping, 2=Flat, 3=Downsloping
        slope = st.selectbox(
            "Slope of peak exercise ST segment (slope)",
            options=[1, 2, 3],
            format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x],
            index=1
        )
        ca = st.selectbox("Number of major vessels (0-3) colored by flourosopy (ca)", options=[0,1,2,3], index=0)
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[3, 6, 7],
            format_func=lambda x: {3:"Normal",6:"Fixed defect",7:"Reversable defect"}[x],
            index=0
        )

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
        
        st.subheader("Model Explanation (LIME)")
        with st.spinner("Generating explanation..."):
            explainer = get_explainer()
            
            # Create a prediction function wrapper for LIME
            # LIME expects a function that takes a 2D numpy array and returns probabilities
            def predict_fn(x_numpy):
                # Convert back to dataframe since our pipeline expects pandas
                df_x = pd.DataFrame(x_numpy, columns=X.columns)
                return model.predict_proba(df_x)
                
            # Explain the instance — pass the USER's input row (X is the user DataFrame here)
            exp = explainer.explain_instance(
                data_row=X.values[0].astype(float),  # user's input, all numeric now
                predict_fn=predict_fn,
                num_features=10
            )
            
            # Display the LIME explanation in Streamlit using HTML component
            html = exp.as_html()
            components.html(html, height=800)


if __name__ == "__main__":
    main()
