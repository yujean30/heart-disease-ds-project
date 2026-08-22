import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import random
from SVM_Model.SVM import SVMXGBHybrid

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Heart Disease Risk Prediction System")

# ---------------------------------------------------------
# 2. Load Models & Scalers
# ---------------------------------------------------------
@st.cache_resource
def load_baseline_artifacts():
    model_path = 'logistic_regression_model/best_lr_model.pkl'
    scaler_path = 'logistic_regression_model/scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    else:
        st.error("Baseline LR model or scaler missing!")
        return None, None

@st.cache_resource
def load_rf_artifacts():
    model_path = 'random_forest_model/random_forest_tuned.joblib'
    scaler_path = 'random_forest_model/scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    else:
        st.error("Random Forest model or scaler missing!")
        return None, None

@st.cache_resource
def load_svm_artifacts():
    model_path = 'SVM_Model/best_svm_xgb_hybrid_model.joblib'
    scaler_path = 'shared_scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

lr_model, lr_scaler = load_baseline_artifacts()
rf_model, rf_scaler = load_rf_artifacts()
svm_model, svm_scaler = load_svm_artifacts()

# ---------------------------------------------------------
# 3. Model Selection
# ---------------------------------------------------------
st.write("### Select Prediction Model")
model_choice = st.selectbox(
    "Choose a model:",
    ["Logistic Regression (Baseline)", "Random Forest", "SVM"]
)

if model_choice == "Logistic Regression (Baseline)":
    model, scaler = lr_model, lr_scaler
    metrics_path = 'logistic_regression_model/lr_baseline_metrics.csv'
    cm_path = 'logistic_regression_model/lr_confusion_matrix.png'
    roc_path = 'logistic_regression_model/lr_roc_curve.png'
    decision_threshold = 0.5
elif model_choice == "Random Forest":
    model, scaler = rf_model, rf_scaler
    metrics_path = 'random_forest_model/rf_metrics.csv'
    cm_path = 'random_forest_model/rf_confusion_matrix.png'
    roc_path = 'random_forest_model/rf_roc_curve.png'
    decision_threshold = 0.5
else:
    model, scaler = svm_model, svm_scaler
    metrics_path = 'SVM_Model/svm_xgb_metrics.csv'
    cm_path = 'SVM_Model/svm_xgb_confusion_matrix.png'
    roc_path = 'SVM_Model/svm_xgb_roc_curve.png'
    threshold_path = 'SVM_Model/svm_xgb_decision_threshold.joblib'
    decision_threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5
    if model is None:
        st.warning("SVM model file is missing. Run the SVM training script to enable SVM predictions.")

# ---------------------------------------------------------
# 3b. Model Comparison Section (Simplified) — moved above tabs
# ---------------------------------------------------------
st.markdown("---")
st.write("## 📊 Model Comparison")

display_metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
lr_metrics_path = 'logistic_regression_model/lr_baseline_metrics.csv'
rf_metrics_path = 'random_forest_model/rf_metrics.csv'
svm_metrics_path = 'SVM_Model/svm_xgb_metrics.csv'

if os.path.exists(lr_metrics_path) and os.path.exists(rf_metrics_path):
    df_lr = pd.read_csv(lr_metrics_path)
    df_rf = pd.read_csv(rf_metrics_path)

    df_lr['Model'] = 'Logistic Regression'
    df_rf['Model'] = 'Random Forest'

    comparison_frames = [df_lr, df_rf]
    if os.path.exists(svm_metrics_path):
        df_svm = pd.read_csv(svm_metrics_path)
        df_svm['Model'] = 'SVM'
        comparison_frames.append(df_svm)

    # Combine and reset index properly
    df_compare = pd.concat(comparison_frames, ignore_index=True)
    df_compare = df_compare[["Model", *display_metric_columns]]

    # Identify best model by F1-Score
    best_model_idx = df_compare['F1-Score'].idxmax()
    best_model_name = df_compare.loc[best_model_idx, 'Model']

    # Display plain table (no highlight)
    st.write("### Model Comparison Table")
    st.dataframe(df_compare, width="stretch")

    # Display result summary
    st.success(f"✅ Based on F1-Score, **{best_model_name}** performs better overall.")
else:
    st.info("Comparison metrics not available yet. Please ensure both models have metrics CSV files saved.")

st.markdown("---")

# ---------------------------------------------------------
# 4. Tabs for Prediction, Metrics, Samples, and EDA
# ---------------------------------------------------------
tab1, tab2, tab3, tab_eda = st.tabs([
    "📋 Patient Assessment (Predictor)",
    "📊 Model Performance Metrics",
    "🧪 Auto‑Generated Samples",
    "🔬 EDA"
])

# ---------------------------------------------------------
# TAB 1: Predictor
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

    if st.button("🔎 Calculate Patient Risk Probability", type="primary"):
        if model is not None and scaler is not None:
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
            num_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
                        'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            probability = model.predict_proba(input_df)[0][1]
            st.markdown(f"### Diagnosis Result ({model_choice})")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Calculated Disease Probability", value=f"{probability * 100:.1f}%")
            with res_col2:
                if probability >= decision_threshold:
                    st.error("⚠️ **HIGH RISK**: Patient shows symptoms/risk patterns for Heart Disease.")
                else:
                    st.success("✅ **LOW RISK**: Patient is unlikely to have Heart Disease.")

# ---------------------------------------------------------
# TAB 2: Performance Metrics
# ---------------------------------------------------------
with tab2:
    st.write(f"### {model_choice} Model Metrics & Diagnostic Plots")

    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        available_metrics = [
            column for column in display_metric_columns
            if column in df_metrics.columns
        ]
        st.dataframe(df_metrics[available_metrics], width="stretch")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption=f"{model_choice} Confusion Matrix", width="stretch")
    with col_img2:
        if os.path.exists(roc_path):
            st.image(Image.open(roc_path), caption=f"{model_choice} ROC Curve", width="stretch")

# ---------------------------------------------------------
# TAB 3: Auto‑Generated Samples (Validated)
# ---------------------------------------------------------
with tab3:
    st.write("### Auto‑Generated Example Data")

    if st.button("⚙️ Generate Sample Data"):
        feature_order = [
            'Age','Gender','Blood Pressure','Cholesterol Level','Exercise Habits',
            'Smoking','Family Heart Disease','Diabetes','BMI','High Blood Pressure',
            'Low HDL Cholesterol','High LDL Cholesterol','Alcohol Consumption',
            'Stress Level','Sleep Hours','Sugar Consumption','Triglyceride Level',
            'Fasting Blood Sugar','CRP Level','Homocysteine Level'
        ]

        num_cols = ['Age','Blood Pressure','Cholesterol Level','BMI','Sleep Hours',
                    'Triglyceride Level','Fasting Blood Sugar','CRP Level','Homocysteine Level']

        def generate_sample(label):
            samples = []
            for _ in range(5):
                if label == "High Risk":
                    samples.append({
                        'Age': random.randint(55, 80),
                        'Gender': 1,
                        'Blood Pressure': round(random.uniform(150, 220), 1),
                        'Cholesterol Level': round(random.uniform(250, 400), 1),
                        'Exercise Habits': 0,
                        'Smoking': 1,
                        'Family Heart Disease': 1,
                        'Diabetes': 1,
                        'BMI': round(random.uniform(28, 40), 1),
                        'High Blood Pressure': 1,
                        'Low HDL Cholesterol': 1,
                        'High LDL Cholesterol': 1,
                        'Alcohol Consumption': 3,
                        'Stress Level': 2,
                        'Sleep Hours': round(random.uniform(3, 6), 1),
                        'Sugar Consumption': 2,
                        'Triglyceride Level': round(random.uniform(200, 500), 1),
                        'Fasting Blood Sugar': round(random.uniform(150, 300), 1),
                        'CRP Level': round(random.uniform(10, 30), 1),
                        'Homocysteine Level': round(random.uniform(15, 30), 1)
                    })
                else:
                    samples.append({
                        'Age': random.randint(25, 45),
                        'Gender': 0,
                        'Blood Pressure': round(random.uniform(90, 130), 1),
                        'Cholesterol Level': round(random.uniform(150, 220), 1),
                        'Exercise Habits': random.choice([1, 2]),
                        'Smoking': 0,
                        'Family Heart Disease': 0,
                        'Diabetes': 0,
                        'BMI': round(random.uniform(18, 26), 1),
                        'High Blood Pressure': 0,
                        'Low HDL Cholesterol': 0,
                        'High LDL Cholesterol': 0,
                        'Alcohol Consumption': random.choice([0, 1]),
                        'Stress Level': 0,
                        'Sleep Hours': round(random.uniform(7, 9), 1),
                        'Sugar Consumption': 0,
                        'Triglyceride Level': round(random.uniform(50, 150), 1),
                        'Fasting Blood Sugar': round(random.uniform(70, 110), 1),
                        'CRP Level': round(random.uniform(0, 5), 1),
                        'Homocysteine Level': round(random.uniform(5, 12), 1)
                    })
            return pd.DataFrame(samples)

        # Generate both sets
        df_low = generate_sample("Low Risk")[feature_order]
        df_high = generate_sample("High Risk")[feature_order]

        # Validation: clip values to realistic ranges
        valid_ranges = {
            'Age': (18, 100),
            'Blood Pressure': (80, 220),
            'Cholesterol Level': (100, 400),
            'BMI': (10, 50),
            'Sleep Hours': (2, 14),
            'Triglyceride Level': (50, 500),
            'Fasting Blood Sugar': (70, 300),
            'CRP Level': (0, 30),
            'Homocysteine Level': (0, 30)
        }
        for col, (low, high) in valid_ranges.items():
            if col in df_low.columns:
                df_low[col] = df_low[col].clip(lower=low, upper=high)
            if col in df_high.columns:
                df_high[col] = df_high[col].clip(lower=low, upper=high)

        # Age must be integer
        df_low['Age'] = df_low['Age'].astype(int)
        df_high['Age'] = df_high['Age'].astype(int)

        # Scale numeric columns only for prediction
        scaled_low = df_low.copy()
        scaled_high = df_high.copy()
        scaled_low[num_cols] = scaler.transform(df_low[num_cols])
        scaled_high[num_cols] = scaler.transform(df_high[num_cols])

        # Predict
        df_low['Predicted Probability'] = model.predict_proba(scaled_low)[:, 1]
        df_low['Predicted Risk'] = np.where(df_low['Predicted Probability'] >= decision_threshold, "High Risk", "Low Risk")

        df_high['Predicted Probability'] = model.predict_proba(scaled_high)[:, 1]
        df_high['Predicted Risk'] = np.where(df_high['Predicted Probability'] >= decision_threshold, "High Risk", "Low Risk")

        # Filter mismatched predictions
        df_low = df_low[df_low['Predicted Risk'] == 'Low Risk']
        df_high = df_high[df_high['Predicted Risk'] == 'High Risk']

        # Map numeric codes back to readable labels
        mapping_dict = {
            'Gender': {0: 'Female', 1: 'Male'},
            'Exercise Habits': {0: 'Low', 1: 'Medium', 2: 'High'},
            'Alcohol Consumption': {0: 'None', 1: 'Low', 2: 'Medium', 3: 'High'},
            'Stress Level': {0: 'Low', 1: 'Medium', 2: 'High'},
            'Sugar Consumption': {0: 'Low', 1: 'Medium', 2: 'High'},
            'Smoking': {0: 'No', 1: 'Yes'},
            'Family Heart Disease': {0: 'No', 1: 'Yes'},
            'Diabetes': {0: 'No', 1: 'Yes'},
            'High Blood Pressure': {0: 'No', 1: 'Yes'},
            'Low HDL Cholesterol': {0: 'No', 1: 'Yes'},
            'High LDL Cholesterol': {0: 'No', 1: 'Yes'}
        }
        for col, mapping in mapping_dict.items():
            if col in df_low.columns:
                df_low[col] = df_low[col].map(mapping)
            if col in df_high.columns:
                df_high[col] = df_high[col].map(mapping)

        # Display results
        st.write("#### 🟢 Low‑Risk Samples (Predicted)")
        st.dataframe(df_low, width="stretch")

        st.write("#### 🔴 High‑Risk Samples (Predicted)")
        st.dataframe(df_high, width="stretch")

        st.success("✅ Synthetic samples generated and validated successfully.")

# ---------------------------------------------------------
# TAB 4: EDA
# ---------------------------------------------------------
with tab_eda:
    st.write("### 🔬 Exploratory Data Analysis")

    if "df_eda" not in st.session_state:
        st.session_state.df_eda = None

    # FIXED: Added directly relative path since CSV is alongside UI.py
    candidate_paths = [
        "heart_disease_cleaned_full.csv",
    ]
    auto_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    # Automatically load default CSV on initial page load or browser refresh
    if st.session_state.df_eda is None and auto_path is not None:
        st.session_state.df_eda = pd.read_csv(auto_path)

    uploaded_file = st.file_uploader("Or upload a dataset CSV", type=["csv"])

    if uploaded_file is not None:
        st.session_state.df_eda = pd.read_csv(uploaded_file)

    df_eda = st.session_state.df_eda

    if df_eda is not None:
        if uploaded_file is None and auto_path is not None:
            st.caption(f"Using default dataset loaded from local path: `{auto_path}`")
            
        st.write(f"**Shape:** {df_eda.shape[0]} rows × {df_eda.shape[1]} columns")

        st.write("#### Data Preview")
        st.dataframe(df_eda.head(20), width="stretch")

        st.write("#### Summary Statistics")
        st.dataframe(df_eda.describe(include="all").transpose(), width="stretch")

        st.write("#### Missing Values")
        missing = df_eda.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(
                missing.rename("Missing Count"),
                column_config={
                    "Missing Count": st.column_config.NumberColumn(
                        "Missing Count", alignment="left"
                    )
                },
                use_container_width=True,
            )

    numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.write("#### Feature Distribution")
        dist_col = st.selectbox("Select a numeric feature to view its distribution", numeric_cols)

        # 1. Use 6 to 8 bins max for clean visual grouping
        num_bins = 6
        min_val = df_eda[dist_col].min()
        max_val = df_eda[dist_col].max()

        # 2. Compute histogram with whole number rounded bin edges
        counts, bin_edges = np.histogram(df_eda[dist_col].dropna(), bins=num_bins)
        
        # 3. Format edges into clean integer ranges (e.g., "18 - 28", "28 - 38")
        bin_labels = [
            f"{int(round(bin_edges[i]))} - {int(round(bin_edges[i+1]))}" 
            for i in range(len(bin_edges)-1)
        ]

        # 4. Render clean bar chart
        chart_data = pd.DataFrame({
            "Range": bin_labels,
            "Count": counts
        }).set_index("Range")

        st.bar_chart(chart_data)
    else:
        st.info("No dataset found automatically — upload a CSV above to run EDA.")
