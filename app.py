import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    csv_path = os.path.join(base_dir, "diabetes.csv")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(csv_path)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df.dropna(inplace=True)
    for col in df.columns[:-1]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
    X, y = df.drop("Outcome", axis=1), df["Outcome"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_model()

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .block-container { padding-top: 2rem; }
        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 700;
            margin-top: 1rem;
        }
        .diabetic { background-color: #ffe4e4; color: #c0392b; border: 2px solid #e74c3c; }
        .healthy  { background-color: #e4f9e4; color: #1e8449; border: 2px solid #27ae60; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter the patient's medical details below to predict diabetes risk using a trained **Decision Tree** model.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.divider()

if st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.markdown(f"""
            <div class="result-box diabetic">
                ⚠️ High Risk: Likely Diabetic<br>
                <small style="font-size:0.9rem; font-weight:400;">Confidence: {probability[1]*100:.1f}%</small>
            </div>
        """, unsafe_allow_html=True)
        st.warning("Please consult a healthcare professional for further evaluation.")
    else:
        st.markdown(f"""
            <div class="result-box healthy">
                ✅ Low Risk: Non-Diabetic<br>
                <small style="font-size:0.9rem; font-weight:400;">Confidence: {probability[0]*100:.1f}%</small>
            </div>
        """, unsafe_allow_html=True)
        st.success("The patient appears to be at low risk of diabetes. Stay healthy!")

st.divider()
st.caption("Model: Decision Tree Classifier | Dataset: Pima Indians Diabetes | Accuracy: ~75.32%")
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for medical advice.")
