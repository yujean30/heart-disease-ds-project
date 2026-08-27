from importlib.resources import path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import random
from SVM_Model.SVM import SVMXGBHybrid
from pathlib import Path
import plotly.express as px

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Heart Disease Risk Prediction System")

@st.cache_resource
def load_baseline_artifacts():
    model_path = 'Logistic_Regression_Model/best_lr_model.pkl'
    scaler_path = 'Logistic_Regression_Model/scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    st.error("Baseline LR model or scaler missing!")
    return None, None

@st.cache_resource
def load_rf_artifacts():
    model_path = 'random_forest/random_forest_model/random_forest_tuned.joblib'
    scaler_path = 'shared_scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    st.error("Random Forest model or scaler missing!")
    return None, None

@st.cache_resource
def load_svm_artifacts():
    model_path = 'SVM_Model/best_svm_xgb_hybrid_model.joblib'
    scaler_path = 'shared_scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

@st.cache_resource
def load_knn_artifacts():
    model_path = 'KNN_Model/knn_model.joblib'
    scaler_path = 'KNN_Model/scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    st.error("KNN model or scaler missing!")
    return None, None

lr_model, lr_scaler = load_baseline_artifacts()
rf_model, rf_scaler = load_rf_artifacts()
svm_model, svm_scaler = load_svm_artifacts()
knn_model, knn_scaler = load_knn_artifacts()

# ---------------------------------------------------------
# 3b. Model Comparison Section (Simplified) — moved above tabs
# ---------------------------------------------------------
st.markdown("---")
st.write("## 📊 Model Comparison")

display_metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
lr_metrics_path = 'Logistic_Regression_Model/lr_baseline_metrics.csv'
rf_metrics_path = 'random_forest/random_forest_model/metrics.csv'
svm_metrics_path = 'SVM_Model/svm_xgb_metrics.csv'
knn_metrics_path = 'KNN_Model/knn_baseline_metrics.csv'

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
    if os.path.exists(knn_metrics_path):
        df_knn = pd.read_csv(knn_metrics_path)
        # ---
        # Rename columns to match display_metric_columns
        df_knn = df_knn.rename(columns={
            'F1 Score': 'F1-Score',
            'ROC AUC': 'ROC-AUC'
        })
        df_knn['Model'] = 'KNN'
        # ---
        comparison_frames.append(df_knn)

    # Combine and reset index properly
    df_compare = pd.concat(comparison_frames, ignore_index=True)
    df_compare = df_compare[["Model", *display_metric_columns]]

    # Identify best model by F1-Score
    best_model_idx = df_compare['F1-Score'].idxmax()
    best_model_name = df_compare.loc[best_model_idx, 'Model']

    # Display plain table (no highlight)
    st.write("### Model Comparison Table")
    st.dataframe(df_compare, width="stretch")

    # Reshape the shared metrics so every model can be compared in one chart.
    comparison_long = df_compare.melt(
        id_vars="Model",
        value_vars=display_metric_columns,
        var_name="Metric",
        value_name="Score"
    )
    comparison_long["Score (%)"] = comparison_long["Score"] * 100

    st.write("### Performance Comparison")
    comparison_fig = px.bar(
        comparison_long,
        x="Metric",
        y="Score (%)",
        color="Model",
        barmode="group",
        text="Score (%)",
        hover_data={"Score (%)": ":.2f"},
        color_discrete_map={
            "Logistic Regression": "#ecec98",
            "Random Forest": "#7dcfb6",
            "SVM": "#b5a4cb",
            "KNN": "#1C2B48"
        },
        title="Model Performance Across Evaluation Metrics"
    )
    comparison_fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False
    )
    comparison_fig.update_layout(
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
        xaxis_title="Evaluation metric",
        legend_title="Model"
    )
    st.plotly_chart(comparison_fig, width="stretch")

    # Display result summary
    st.success(f"✅ Based on F1-Score, **{best_model_name}** performs better overall.")
else:
    st.info("Comparison metrics not available yet. Please ensure both models have metrics CSV files saved.")

st.markdown("---")

# ---------------------------------------------------------
# 3. Model Selection
# ---------------------------------------------------------
st.write("### Select Prediction Model")
model_choice = st.selectbox(
    "Choose a model:",
    ["Logistic Regression (Baseline)", "Random Forest", "SVM", "KNN"]
)

if model_choice == "Logistic Regression (Baseline)":
    model, scaler = lr_model, lr_scaler
    metrics_path = 'Logistic_Regression_Model/lr_baseline_metrics.csv'
    cm_path = 'Logistic_Regression_Model/lr_confusion_matrix.png'
    roc_path = 'Logistic_Regression_Model/lr_roc_curve.png'
    decision_threshold = 0.5
elif model_choice == "Random Forest":
    model, scaler = rf_model, rf_scaler
    metrics_path = 'random_forest/random_forest_model/metrics.csv'
    cm_path = 'random_forest/random_forest_model/confusion_matrix.png'
    roc_path = 'random_forest/random_forest_model/roc_curve.png'
    model_dir = 'random_forest/random_forest_model'
    threshold_path = os.path.join(model_dir, 'decision_threshold.joblib')
    if os.path.exists(threshold_path):
        decision_threshold = joblib.load(threshold_path)
    else:
        st.warning(f"decision_threshold.joblib not found in {model_dir} -- falling back to 0.5.")
        decision_threshold = 0.5
elif model_choice == "KNN":
    model, scaler = knn_model, knn_scaler
    metrics_path = 'KNN_Model/knn_metrics.csv'
    cm_path = 'KNN_Model/knn_confusion_matrix.png'
    roc_path = 'KNN_Model/knn_roc_curve.png'
    decision_threshold = 0.5
    if model is None:
        st.warning("KNN model file is missing. Export the trained model/scaler from KNN_Heart_Disease_Model.ipynb to enable KNN predictions.")
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
        exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])

    with col2:
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Low", "Medium", "High"])
        stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
        smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        family_history = st.selectbox("Family Heart Disease History", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes Status", ["No", "Yes"])
        high_bp = st.selectbox("High Blood Pressure Diagnosis", ["No", "Yes"])

    with col3:
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
            #---
            # Scale the numerical features, but handle KNN separately 
            # since it may have been trained on a different feature set.
            if model_choice == "KNN":
                input_df = input_df.reindex(columns=scaler.feature_names_in_)
                input_df = pd.DataFrame(
                    scaler.transform(input_df),
                    columns=input_df.columns,
                    index=input_df.index
                )
            else:
                input_df[num_cols] = scaler.transform(input_df[num_cols])
            #---

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

        #---
        if model_choice == "KNN":
            knn_columns = scaler.feature_names_in_
            scaled_low = pd.DataFrame(
                scaler.transform(df_low.reindex(columns=knn_columns)),
                columns=knn_columns,
                index=df_low.index
            )
            scaled_high = pd.DataFrame(
                scaler.transform(df_high.reindex(columns=knn_columns)),
                columns=knn_columns,
                index=df_high.index
            )
        else:
            scaled_low[num_cols] = scaler.transform(df_low[num_cols])
            scaled_high[num_cols] = scaler.transform(df_high[num_cols])
        #---

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
    st.subheader("Exploratory Data Analysis")
    st.caption(
        "A focused view of data quality, disease prevalence, and the features most useful "
        "for understanding the prediction problem. These associations do not imply causation."
    )

    TARGET_COL = "Heart Disease Status"
    DATA_PATH = Path(__file__).resolve().parent / "heart_disease.csv"

    @st.cache_data
    def load_eda_data(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        data.columns = data.columns.str.strip()
        return data

    try:
        df_eda = load_eda_data(str(DATA_PATH))
    except FileNotFoundError:
        st.error("EDA dataset not found. Keep heart_disease.csv beside UI.py.")
        st.stop()

    if TARGET_COL not in df_eda.columns:
        st.error(f"Expected target column '{TARGET_COL}' was not found in the dataset.")
        st.stop()

    # 1. One concise data-quality overview (instead of a large raw-data table).
    positive_rate = (df_eda[TARGET_COL].astype(str).str.strip() == "Yes").mean() * 100
    audit_col1, audit_col2, audit_col3, audit_col4 = st.columns(4)
    audit_col1.metric("Patient records", f"{len(df_eda):,}")
    audit_col2.metric("Features", df_eda.shape[1] - 1)
    audit_col3.metric("Missing cells", f"{int(df_eda.isna().sum().sum()):,}")
    audit_col4.metric("Heart-disease prevalence", f"{positive_rate:.1f}%")

    with st.expander("Dataset audit and preview"):
            missing = (
                df_eda.isna().sum()
                .rename("Missing values")
                .to_frame()
                .query("`Missing values` > 0")
                .sort_values("Missing values", ascending=False)
            )
            st.write(f"Duplicate rows: **{df_eda.duplicated().sum():,}**")
            if missing.empty:
                st.success("No missing values detected.")
            else:
                st.dataframe(
                    missing,
                    column_config={
                        "Missing values": st.column_config.NumberColumn(
                            "Missing values",
                            alignment="left"
                        )
                    },
                    use_container_width=True
                )
            st.dataframe(df_eda.head(10), width="stretch")

    st.markdown("#### 1. Outcome balance")
    target_counts = (
        df_eda[TARGET_COL]
        .fillna("Missing")
        .value_counts()
        .rename_axis(TARGET_COL)
        .reset_index(name="Patients")
    )
    target_counts["Percentage"] = target_counts["Patients"] / len(df_eda) * 100
    fig_target = px.bar(
        target_counts,
        x=TARGET_COL,
        y="Patients",
        color=TARGET_COL,
        text=target_counts["Percentage"].map("{:.1f}%".format),
        color_discrete_map={"No": "#2E8B57", "Yes": "#D9534F", "Missing": "#6C757D"},
        title="Heart Disease Status",
    )
    fig_target.update_layout(showlegend=False, yaxis_title="Number of patients")
    fig_target.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_target, width="stretch")
    st.info(
        "Interpretation: the target is imbalanced, so evaluate models with recall, precision, "
        "F1-score, and ROC-AUC - not accuracy alone."
    )

    left, right = st.columns(2)

    # 2. Show one clinically meaningful numeric distribution at a time.
    with left:
        st.markdown("#### 2. Numeric feature by outcome")
        preferred_numeric = [
            "Age", "Blood Pressure", "Cholesterol Level", "BMI",
            "Fasting Blood Sugar", "Triglyceride Level", "CRP Level",
        ]
        numeric_choices = [c for c in preferred_numeric if c in df_eda.columns]
        numeric_feature = st.selectbox(
            "Clinical measure", numeric_choices, key="eda_numeric_feature"
        )
        fig_numeric = px.histogram(
            df_eda.dropna(subset=[numeric_feature, TARGET_COL]),
            x=numeric_feature,
            color=TARGET_COL,
            barmode="overlay",
            opacity=0.60,
            nbins=25,
            color_discrete_map={"No": "#2E8B57", "Yes": "#DE827F"},
            title=f"Distribution of {numeric_feature}",
        )
        fig_numeric.update_layout(legend_title_text="Heart disease")
        st.plotly_chart(fig_numeric, width="stretch")

    # 3. Compare disease rates, rather than counts, across a categorical risk factor.
    with right:
        st.markdown("#### 3. Disease rate by risk factor")
        preferred_categorical = [
            "Exercise Habits", "Smoking", "Diabetes", "Family Heart Disease",
            "High Blood Pressure", "Stress Level", "Alcohol Consumption",
        ]
        categorical_choices = [c for c in preferred_categorical if c in df_eda.columns]
        category_feature = st.selectbox(
            "Risk factor", categorical_choices, key="eda_category_feature"
        )
        rate_data = df_eda[[category_feature, TARGET_COL]].dropna().copy()
        rate_data["Disease rate (%)"] = (
            rate_data[TARGET_COL].astype(str).str.strip().eq("Yes").astype(int)
        )
        rate_data = (
            rate_data.groupby(category_feature, as_index=False)["Disease rate (%)"]
            .mean()
            .assign(**{"Disease rate (%)": lambda x: x["Disease rate (%)"] * 100})
            .sort_values("Disease rate (%)", ascending=False)
        )
        fig_rate = px.bar(
            rate_data,
            x=category_feature,
            y="Disease rate (%)",
            text=rate_data["Disease rate (%)"].map("{:.1f}%".format),
            color_discrete_sequence=["#F57970"],
            title=f"Heart-disease rate by {category_feature}",
        )
        fig_rate.update_traces(textposition="outside", cliponaxis=False)
        fig_rate.update_layout(yaxis_range=[0, min(100, rate_data["Disease rate (%)"].max() + 8)])
        st.plotly_chart(fig_rate, width="stretch")

    # 4. A small correlation matrix is more readable than an all-feature heatmap.
    st.markdown("#### 4. Relationships among core clinical measures")
    correlation_features = [
        "Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
        "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level",
    ]
    correlation_features = [c for c in correlation_features if c in df_eda.columns]
    corr = df_eda[correlation_features].corr(numeric_only=True).round(2)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Pearson correlation matrix (numeric clinical features)",
    )
    fig_corr.update_layout(coloraxis_colorbar_title="Correlation")
    st.plotly_chart(fig_corr, width="stretch")

    st.caption(
        "Presentation tip: explain one takeaway per chart. Keep the full descriptive statistics "
        "and every exploratory chart in the report or appendix, not on the live demo screen."
    )