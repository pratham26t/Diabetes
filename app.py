import streamlit as st
import numpy as np
import joblib
import os

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# ==============================
# Custom CSS for UI Enhancement
# ==============================
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.result-positive {
    background-color: #fdecea;
    padding: 20px;
    border-radius: 15px;
    border-left: 6px solid #c0392b;
}
.result-negative {
    background-color: #eafaf1;
    padding: 20px;
    border-radius: 15px;
    border-left: 6px solid #1e8449;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Models
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    return joblib.load(os.path.join(BASE_DIR, filename))

models = {
    "Random Forest": load_model("random_forest.pkl"),
    "Bagging (Decision Tree)": load_model("bagging_dt.pkl"),
    "AdaBoost": load_model("adaboost.pkl"),
    "Voting Classifier": load_model("voting.pkl")
}

# ==============================
# Sidebar
# ==============================
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Select the machine learning model for prediction.")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)

st.sidebar.info(
    "📌 **Note:** This tool assists in early risk screening and is not a medical diagnosis."
)

# ==============================
# Main Header
# ==============================
st.markdown(
    "<h1 style='text-align:center;'>🩺 Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>"
    "Enter patient health parameters to predict diabetes risk using ensemble ML models"
    "</p>",
    unsafe_allow_html=True
)

# ==============================
# Input Section
# ==============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧾 Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    Glucose = st.number_input("Glucose Level", min_value=0.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0)

with col2:
    Insulin = st.number_input("Insulin Level", min_value=0.0)
    BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0)
    DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    Age = st.number_input("Age", min_value=0, step=1)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    features = np.array([
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DPF, Age
    ], dtype=float).reshape(1, -1)

    prediction = models[model_choice].predict(features)[0]

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-positive">
                <h3>⚠️ Prediction: Diabetic</h3>
                <p><strong>Model Used:</strong> {model_choice}</p>
                <p>The patient shows a higher risk of diabetes. Medical consultation is advised.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-negative">
                <h3>✅ Prediction: Non-Diabetic</h3>
                <p><strong>Model Used:</strong> {model_choice}</p>
                <p>The patient shows a lower risk of diabetes.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================
# Footer
# ==============================
st.markdown(
    "<div class='footer'>"
    "© 2026 Diabetes Prediction App | Built with Streamlit & Machine Learning"
    "</div>",
    unsafe_allow_html=True
)
