import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessor (Baseline LR)",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Heart Disease Risk Prediction System")
st.subheader("Baseline Model: Logistic Regression")

# ---------------------------------------------------------
# 2. Load Model, Scaler & Metrics
# ---------------------------------------------------------
@st.cache_resource
def load_baseline_artifacts():
    model_path = 'logistic_regression_model/best_lr_model.pkl'
    scaler_path = 'logistic_regression_model/scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        st.error("Model or Scaler file missing! Please run the baseline model training script first.")
        return None, None

model, scaler = load_baseline_artifacts()

# Create tabs for UI navigation
tab1, tab2 = st.tabs(["📋 Patient Assessment (Predictor)", "📊 Baseline Performance Metrics"])

# ---------------------------------------------------------
# TAB 1: Live Patient Prediction Interface
# ---------------------------------------------------------
with tab1:
    st.write("### Input Patient Clinical Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        bp = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=220, value=120)
        chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        sleep = st.number_input("Sleep Hours per Day", min_value=2.0, max_value=14.0, value=7.0)
        
    with col2:
        exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Low", "Medium", "High"])
        stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
        smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        family_history = st.selectbox("Family Heart Disease History", ["No", "Yes"])

    with col3:
        diabetes = st.selectbox("Diabetes Status", ["No", "Yes"])
        high_bp = st.selectbox("High Blood Pressure Diagnosis", ["No", "Yes"])
        low_hdl = st.selectbox("Low HDL Cholesterol", ["No", "Yes"])
        high_ldl = st.selectbox("High LDL Cholesterol", ["No", "Yes"])
        triglycerides = st.number_input("Triglyceride Level", min_value=50, max_value=500, value=150)
        fbs = st.number_input("Fasting Blood Sugar", min_value=70, max_value=300, value=100)
        crp = st.number_input("CRP Level", min_value=0.0, max_value=30.0, value=2.0)
        homocysteine = st.number_input("Homocysteine Level", min_value=0.0, max_value=30.0, value=10.0)

    st.markdown("---")
    
    if st.button("🔴 Calculate Patient Risk Probability", type="primary", width="stretch"):
        if model is not None and scaler is not None:
            # 1. Map Categorical UI inputs to numerical encodings
            input_dict = {
                'Age': age,
                'Gender': 1 if gender == "Male" else 0,
                'Blood Pressure': bp,
                'Cholesterol Level': chol,
                'Exercise Habits': {"Low": 0, "Medium": 1, "High": 2}[exercise],
                'Smoking': 1 if smoking == "Yes" else 0,
                'Family Heart Disease': 1 if family_history == "Yes" else 0,
                'Diabetes': 1 if diabetes == "Yes" else 0,
                'BMI': bmi,
                'High Blood Pressure': 1 if high_bp == "Yes" else 0,
                'Low HDL Cholesterol': 1 if low_hdl == "Yes" else 0,
                'High LDL Cholesterol': 1 if high_ldl == "Yes" else 0,
                'Alcohol Consumption': {"None": 0, "Low": 1, "Medium": 2, "High": 3}[alcohol],
                'Stress Level': {"Low": 0, "Medium": 1, "High": 2}[stress],
                'Sleep Hours': sleep,
                'Sugar Consumption': {"Low": 0, "Medium": 1, "High": 2}[sugar],
                'Triglyceride Level': triglycerides,
                'Fasting Blood Sugar': fbs,
                'CRP Level': crp,
                'Homocysteine Level': homocysteine
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # 2. Scale Numerical Columns using saved Scaler
            num_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
                        'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            
            # 3. Predict Probability
            probability = model.predict_proba(input_df)[0][1]
            prediction = model.predict(input_df)[0]
            
            # 4. Display Output
            st.markdown("### Diagnosis Result:")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label="Calculated Disease Probability", value=f"{probability * 100:.1f}%")
                
            with res_col2:
                if prediction == 1 or probability >= 0.5:
                    st.error("⚠️ **HIGH RISK**: Patient shows symptoms/risk patterns for Heart Disease.")
                else:
                    st.success("✅ **LOW RISK**: Patient is unlikely to have Heart Disease.")

# ---------------------------------------------------------
# TAB 2: Model Metrics & Diagnostic Plots
# ---------------------------------------------------------
with tab2:
    st.write("### Baseline Model Diagnostic Reports & Plots")
    
    metrics_path = 'logistic_regression_model/lr_baseline_metrics.csv'
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        st.dataframe(df_metrics, width="stretch")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        cm_path = 'logistic_regression_model/lr_confusion_matrix.png'
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption="Baseline Confusion Matrix", width="stretch")
            
    with col_img2:
        roc_path = 'logistic_regression_model/lr_roc_curve.png'
        if os.path.exists(roc_path):
            st.image(Image.open(roc_path), caption="Baseline ROC Curve", width="stretch")