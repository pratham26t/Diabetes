import streamlit as st
import numpy as np
import joblib
import os

# ------------------------------
# Load models (no imputer)
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    return joblib.load(os.path.join(BASE_DIR, filename))

models = {
    "Random Forest": load_model("random_forest.pkl"),
    "Bagging (DT)": load_model("bagging_dt.pkl"),
    "AdaBoost": load_model("adaboost.pkl"),
    "Voting Classifier": load_model("voting.pkl")
}

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(
    page_title="Diabetes Prediction | Ensemble Models",
    layout="centered"
)

st.title("Diabetes Prediction – Ensemble Models")
st.markdown(
    "This app demonstrates multiple ensemble ML models.\n"
    "Enter patient data and select a model to predict diabetes."
)

# Model selection
model_choice = st.selectbox("Select Model", list(models.keys()))

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    Glucose = st.number_input("Glucose", min_value=0.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0)

with col2:
    Insulin = st.number_input("Insulin", min_value=0.0)
    BMI = st.number_input("BMI", min_value=0.0)
    DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    Age = st.number_input("Age", min_value=0, step=1)

# Prediction
if st.button("Predict"):
    # Convert inputs to numpy array
    features = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness,
                         Insulin, BMI, DPF, Age], dtype=float).reshape(1, -1)

    # Prediction
    prediction = models[model_choice].predict(features)[0]

    # Show results with colored box
    if prediction == 1:
        st.markdown(
            f"<div style='background-color:#fdecea;padding:15px;border-radius:10px'>"
            f"<h3 style='color:#c0392b'>Prediction: Diabetic</h3>"
            f"<p>Model used: {model_choice}</p></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#eafaf1;padding:15px;border-radius:10px'>"
            f"<h3 style='color:#1e8449'>Prediction: Non-Diabetic</h3>"
            f"<p>Model used: {model_choice}</p></div>",
            unsafe_allow_html=True
        )
