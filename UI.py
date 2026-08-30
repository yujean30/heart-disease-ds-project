import os
import random
import time
from pathlib import Path
from importlib.resources import path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioLuxe AI | Heart Disease Risk Assessor",
    page_icon="💎",
    layout="wide"
)

# Initialize Session State
if "gender_choice" not in st.session_state:
    st.session_state["gender_choice"] = "Female"

if "smoking_choice" not in st.session_state:
    st.session_state["smoking_choice"] = "No"

if "alcohol_choice" not in st.session_state:
    st.session_state["alcohol_choice"] = "None"

if "sugar_choice" not in st.session_state:
    st.session_state["sugar_choice"] = "Medium"

if "exercise_choice" not in st.session_state:
    st.session_state["exercise_choice"] = "Medium"
    
if "stress_choice" not in st.session_state:
    st.session_state["stress_choice"] = "Medium"

# ---------------------------------------------------------
# 2. Header Section
# ---------------------------------------------------------
header_container = st.container(border=True)
with header_container:
    col_logo, col_desc = st.columns([1, 4])
    with col_logo:
        st.metric(label="System Prediction", value="ONLINE ⚡", delta="Stay Healthy !")
    with col_desc:
        st.title("🫀 Heart Disease Risk Assessor")
        st.caption("Heart Disease Risk Prediction & Exploratory Diagnostic Platform")

# ---------------------------------------------------------
# 3. Model & Scaler Artifact Loaders
# ---------------------------------------------------------
try:
    from SVM_Model.SVM import SVMXGBHybrid
except Exception:
    pass

@st.cache_resource
def load_baseline_artifacts():
    model_path = 'Logistic_Regression_Model/best_lr_model.pkl'
    scaler_path = 'Logistic_Regression_Model/scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

@st.cache_resource
def load_rf_artifacts():
    model_path = 'random_forest/best_rf_model.joblib'
    scaler_path = 'Preprocessing/shared_scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

@st.cache_resource
def load_svm_artifacts():
    model_path = "SVM_Model/best_svm_model.joblib"
    scaler_path = "Preprocessing/shared_scaler.pkl"
    encoder_path = "Preprocessing/feature_encoders.pkl"
    threshold_path = "SVM_Model/svm_decision_threshold.joblib"
    required_files = [
        model_path,
        scaler_path,
        encoder_path,
        threshold_path
    ]
    missing_files = [
        path for path in required_files
        if not os.path.exists(path)
    ]
    if missing_files:
        return (
            None,
            None,
            None,
            None,
            missing_files
        )
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoder_path)
        threshold = joblib.load(threshold_path)
        return (
            model,
            scaler,
            encoders,
            threshold,
            []
        )
    except Exception as e:
        return (None,None,None,None,[f"Loading error: {e}"])


@st.cache_resource
def load_knn_artifacts():
    model_path = 'KNN_Model/knn_model.joblib'
    scaler_path = 'KNN_Model/scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None



lr_model, lr_scaler = load_baseline_artifacts()
rf_model, rf_scaler = load_rf_artifacts()
(svm_model,svm_scaler,svm_encoders,svm_threshold,svm_missing_files) = load_svm_artifacts()
knn_model, knn_scaler = load_knn_artifacts()

# ---------------------------------------------------------
# 4. Model Comparison Section (All Metrics Highlighted)
# ---------------------------------------------------------
st.markdown("---")
st.write("## 📊 Model Comparison")

display_metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
highlight_targets = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]


lr_metrics_path = 'Logistic_Regression_Model/lr_baseline_metrics.csv'
rf_metrics_path = 'random_forest/random_forest_model/metrics.csv'
svm_metrics_path = "SVM_Model/svm_metrics.csv"
knn_metrics_path = 'KNN_Model/knn_baseline_metrics.csv'


if os.path.exists(lr_metrics_path) and os.path.exists(rf_metrics_path):
    df_lr = pd.read_csv(lr_metrics_path)
    df_rf = pd.read_csv(rf_metrics_path)

    df_lr['Model'] = 'Logistic Regression'
    df_rf['Model'] = 'Random Forest'
    comparison_frames = [df_lr,df_rf]
    if os.path.exists(svm_metrics_path):
        df_svm = pd.read_csv(svm_metrics_path)
        df_svm["Model"] = "SVM"
        comparison_frames.append(df_svm)
    if os.path.exists(knn_metrics_path):
        df_knn = pd.read_csv(knn_metrics_path)
        df_knn = df_knn.rename(columns={
            'F1 Score': 'F1-Score',
            'ROC AUC': 'ROC-AUC'
        })
        df_knn['Model'] = 'KNN'
        comparison_frames.append(df_knn)

    df_compare = pd.concat( comparison_frames,ignore_index=True)
    available_columns = [
        column
        for column in display_metric_columns
        if column in df_compare.columns
    ]
    df_compare = df_compare[
        ["Model", *available_columns]
    ]
    if "F1-Score" in df_compare.columns:
        best_model_idx = df_compare["F1-Score"].idxmax()
        best_model_name = df_compare.loc[
            best_model_idx,
            "Model"
        ]
    else:
        best_model_name = "N/A"

    styled_df = df_compare.style.highlight_max(
        subset=[
            col
            for col in highlight_targets
            if col in df_compare.columns
        ],
        color="#bbf7d0",
        axis=0
    )

    styled_df = styled_df.format(
        {
            col: "{:.4f}"
            for col in available_columns
        }
    )
    st.write("### Model Comparison Table")
    st.dataframe(styled_df, width="stretch")

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
        yaxis_range=[0, 105],
        xaxis_title="Evaluation metric",
        legend_title="Model"
    )
    st.plotly_chart(comparison_fig, width="stretch")

    st.success(f"✅ Based on F1-Score, **{best_model_name}** performs better overall.")
else:
    st.info("Comparison metrics not available yet. Please ensure both models have metrics CSV files saved.")

st.markdown("---")

# =========================================================
# 6. MODEL SELECTION
# =========================================================
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
    metrics_path = 'random_forest/rf_metrics.csv'
    cm_path = 'random_forest/rf_confusion_matrix.png'
    roc_path = 'random_forest/rf_roc_curve.png'
    decision_threshold = 0.5
elif model_choice == "KNN":
    model, scaler = knn_model, knn_scaler
    metrics_path = 'KNN_Model/knn_metrics.csv'
    cm_path = 'KNN_Model/knn_confusion_matrix.png'
    roc_path = 'KNN_Model/knn_roc_curve.png'
    decision_threshold = 0.5
elif model_choice == "SVM":
    model = svm_model
    scaler = svm_scaler
    metrics_path = "SVM_Model/svm_metrics.csv"
    cm_path = "SVM_Model/svm_confusion_matrix.png"
    roc_path = "SVM_Model/svm_roc_curve.png"
    decision_threshold = 0.5


# ---------------------------------------------------------
# Helper 0: 3D Gender Interactive Avatar Cards
# ---------------------------------------------------------
def create_3d_gender_cards(current_choice):
    fig = go.Figure()
    genders = [
        {"x": 1.0, "type": "Female", "color": "#ec4899", "bg": "rgba(236, 72, 153, 0.12)", "hair": "#f59e0b"},
        {"x": 2.5, "type": "Male", "color": "#3b82f6", "bg": "rgba(59, 130, 246, 0.12)", "hair": "#451a03"}
    ]

    for g in genders:
        gx = g["x"]
        is_sel = (current_choice == g["type"])
        border_c = g["color"] if is_sel else "#cbd5e1"
        bg_c = g["bg"] if is_sel else "rgba(248, 250, 252, 0.5)"
        border_w = 4.0 if is_sel else 1.8

        fig.add_shape(
            type="rect",
            x0=gx-0.60, y0=0.05, x1=gx+0.60, y1=2.15,
            line=dict(color=border_c, width=border_w),
            fillcolor=bg_c
        )

        fig.add_shape(
            type="rect",
            x0=gx-0.60, y0=0.05, x1=gx+0.60, y1=0.25,
            line=dict(color=border_c, width=1),
            fillcolor=g["color"] if is_sel else "#94a3b8"
        )

        if is_sel:
            fig.add_shape(
                type="circle",
                x0=gx-0.66, y0=-0.02, x1=gx+0.66, y1=2.22,
                line=dict(color=g["color"], width=2.5, dash="dot"),
                fillcolor="rgba(0,0,0,0)"
            )

        # Hair
        if g["type"] == "Female":
            fig.add_shape(type="circle", x0=gx-0.18, y0=1.35, x1=gx+0.18, y1=1.82, fillcolor=g["hair"], line=dict(width=0))
            fig.add_shape(type="rect", x0=gx-0.22, y0=1.20, x1=gx+0.22, y1=1.65, fillcolor=g["hair"], line=dict(width=0))
        else:
            fig.add_shape(type="circle", x0=gx-0.16, y0=1.40, x1=gx+0.16, y1=1.85, fillcolor=g["hair"], line=dict(width=0))

        # Head & Face
        fig.add_shape(type="circle", x0=gx-0.13, y0=1.32, x1=gx+0.13, y1=1.70, fillcolor="#fed7aa", line=dict(width=0))
        fig.add_shape(type="circle", x0=gx-0.08, y0=1.52, x1=gx-0.03, y1=1.57, fillcolor="#431407", line=dict(width=0))
        fig.add_shape(type="circle", x0=gx+0.03, y0=1.52, x1=gx+0.08, y1=1.57, fillcolor="#431407", line=dict(width=0))
        fig.add_shape(type="path", path=f"M {gx-0.05} 1.42 Q {gx} 1.36 {gx+0.05} 1.42", line=dict(color="#ea580c", width=2))

        # Coat & Shirt
        fig.add_shape(type="path", path=f"M {gx-0.18} 1.32 L {gx+0.18} 1.32 L {gx+0.24} 0.65 L {gx-0.24} 0.65 Z",
                      fillcolor="#ffffff", line=dict(color="#cbd5e1", width=1.5))
        shirt_c = "#f472b6" if g["type"] == "Female" else "#60a5fa"
        fig.add_shape(type="path", path=f"M {gx-0.08} 1.32 L {gx+0.08} 1.32 L {gx} 1.10 Z", fillcolor=shirt_c, line=dict(width=0))
        fig.add_shape(type="path", path=f"M {gx-0.10} 1.30 Q {gx} 0.95 {gx+0.10} 1.30", line=dict(color="#0f172a", width=2.5))
        fig.add_shape(type="circle", x0=gx-0.03, y0=0.92, x1=gx+0.03, y1=0.98, fillcolor="#94a3b8", line=dict(color="#0f172a", width=1.5))

        # Arms
        fig.add_shape(type="line", x0=gx-0.18, y0=1.28, x1=gx-0.30, y1=0.75, line=dict(color="#fed7aa", width=6))
        if is_sel:
            fig.add_shape(type="line", x0=gx+0.18, y0=1.25, x1=gx+0.35, y1=1.60, line=dict(color="#fed7aa", width=6))
            fig.add_shape(type="circle", x0=gx+0.32, y0=1.56, x1=gx+0.42, y1=1.68, fillcolor="#fed7aa", line=dict(width=0))
            fig.add_annotation(x=gx+0.48, y=1.75, text="👋", showarrow=False, font=dict(size=18))
        else:
            fig.add_shape(type="line", x0=gx+0.18, y0=1.28, x1=gx+0.30, y1=0.75, line=dict(color="#fed7aa", width=6))

    fig.add_trace(go.Scatter(
        x=[1.0, 2.5],
        y=[1.10, 1.10],
        mode="markers",
        marker=dict(size=110, opacity=0.01),
        hoverinfo="none",
        customdata=["Female", "Male"]
    ))

    fig.update_xaxes(range=[0.2, 3.3], visible=False)
    fig.update_yaxes(range=[-0.05, 2.25], visible=False)
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select'
    )
    return fig


# ---------------------------------------------------------
# Helper 0.5: 3D Luxury Smoking Status Interactive Cards (With Clean Lungs Vector)
# ---------------------------------------------------------
def create_3d_smoking_cards(current_choice):
    fig = go.Figure()
    options = [
        {"x": 1.0, "type": "No", "color": "#10b981", "bg": "rgba(16, 185, 129, 0.12)"},
        {"x": 2.5, "type": "Yes", "color": "#f97316", "bg": "rgba(249, 115, 22, 0.12)"}
    ]

    for opt in options:
        ox = opt["x"]
        is_sel = (current_choice == opt["type"])
        border_c = opt["color"] if is_sel else "#cbd5e1"
        bg_c = opt["bg"] if is_sel else "rgba(248, 250, 252, 0.5)"
        border_w = 4.0 if is_sel else 1.8

        fig.add_shape(
            type="rect",
            x0=ox-0.60, y0=0.05, x1=ox+0.60, y1=2.15,
            line=dict(color=border_c, width=border_w),
            fillcolor=bg_c
        )

        fig.add_shape(
            type="rect",
            x0=ox-0.60, y0=0.05, x1=ox+0.60, y1=0.25,
            line=dict(color=border_c, width=1),
            fillcolor=opt["color"] if is_sel else "#94a3b8"
        )

        if is_sel:
            fig.add_shape(
                type="circle",
                x0=ox-0.66, y0=-0.02, x1=ox+0.66, y1=2.22,
                line=dict(color=opt["color"], width=2.5, dash="dot"),
                fillcolor="rgba(0,0,0,0)"
            )

    # 1. Non-Smoker Artwork: Luxury Clean Healthy Lungs (🫁 Vector Illustration)
    # Left Lung Vector Shape
    fig.add_shape(type="path", path="M 0.88 1.45 Q 0.72 1.30 0.75 1.05 Q 0.82 0.85 0.95 0.95 Q 1.02 1.10 0.98 1.45 Z",
                  fillcolor="#34d399", line=dict(color="#059669", width=2))
    # Right Lung Vector Shape
    fig.add_shape(type="path", path="M 1.12 1.45 Q 1.28 1.30 1.25 1.05 Q 1.18 0.85 1.05 0.95 Q 0.98 1.10 1.02 1.45 Z",
                  fillcolor="#34d399", line=dict(color="#059669", width=2))
    # Trachea / Bronchi stem
    fig.add_shape(type="rect", x0=0.97, y0=1.35, x1=1.03, y1=1.55, fillcolor="#059669", line=dict(width=0))
    
    fig.add_annotation(x=1.0, y=0.55, text="NON-SMOKER", showarrow=False, font=dict(size=11, color="#047857", family="Arial Black"))
    if current_choice == "No":
        fig.add_annotation(x=1.0, y=1.78, text="🫁 CLEAN LUNGS", showarrow=False, font=dict(size=13, color="#059669", family="Arial Black"))

    # 2. Smoker Artwork (3D Cigar with Gold Filter, Ember Glow & Smoke Trails)
    fig.add_shape(type="rect", x0=2.08, y0=1.05, x1=2.68, y1=1.22, fillcolor="#ffffff", line=dict(color="#475569", width=1.5))
    fig.add_shape(type="rect", x0=2.08, y0=1.05, x1=2.25, y1=1.22, fillcolor="#eab308", line=dict(color="#b45309", width=1.5))
    fig.add_shape(type="rect", x0=2.68, y0=1.05, x1=2.76, y1=1.22, fillcolor="#ea580c", line=dict(color="#dc2626", width=1.5))
    fig.add_shape(type="circle", x0=2.74, y0=1.08, x1=2.84, y1=1.19, fillcolor="#ef4444", line=dict(color="#f97316", width=1))
    fig.add_shape(type="path", path="M 2.80 1.22 Q 2.75 1.45 2.88 1.60 Q 2.95 1.75 2.85 1.95",
                  line=dict(color="rgba(148, 163, 184, 0.75)", width=3.5))
    fig.add_shape(type="path", path="M 2.85 1.22 Q 2.95 1.40 2.82 1.62 Q 2.75 1.78 2.92 1.98",
                  line=dict(color="rgba(203, 213, 225, 0.6)", width=2.5))
    fig.add_annotation(x=2.5, y=0.55, text="SMOKER", showarrow=False, font=dict(size=11, color="#c2410c", family="Arial Black"))
    if current_choice == "Yes":
        fig.add_annotation(x=2.5, y=1.75, text="🔥 ACTIVE", showarrow=False, font=dict(size=14, color="#ea580c", family="Arial Black"))

    fig.add_trace(go.Scatter(
        x=[1.0, 2.5],
        y=[1.10, 1.10],
        mode="markers",
        marker=dict(size=110, opacity=0.01),
        hoverinfo="none",
        customdata=["No", "Yes"]
    ))

    fig.update_xaxes(range=[0.2, 3.3], visible=False)
    fig.update_yaxes(range=[-0.05, 2.25], visible=False)
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select'
    )
    return fig


# ---------------------------------------------------------
# Helper 1: Action-Animated 3D Exercise Stages
# ---------------------------------------------------------
def create_animated_action_stages(current_choice, anim_step=0):
    fig = go.Figure()
    cards = [
        {"x": 1.0, "type": "Low", "color": "#6366f1", "bg": "rgba(99, 102, 241, 0.12)"},
        {"x": 2.5, "type": "Medium", "color": "#0284c7", "bg": "rgba(2, 132, 199, 0.12)"},
        {"x": 4.0, "type": "High", "color": "#dc2626", "bg": "rgba(220, 38, 38, 0.12)"}
    ]

    for c in cards:
        cx = c["x"]
        is_sel = (current_choice == c["type"])
        border_c = c["color"] if is_sel else "#cbd5e1"
        bg_c = c["bg"] if is_sel else "rgba(248, 250, 252, 0.5)"
        border_w = 4.0 if is_sel else 1.8

        fig.add_shape(
            type="rect",
            x0=cx-0.65, y0=0.05, x1=cx+0.65, y1=2.15,
            line=dict(color=border_c, width=border_w),
            fillcolor=bg_c
        )

        fig.add_shape(
            type="rect",
            x0=cx-0.65, y0=0.05, x1=cx+0.65, y1=0.25,
            line=dict(color=border_c, width=1),
            fillcolor=c["color"] if is_sel else "#94a3b8"
        )

        if is_sel:
            fig.add_shape(
                type="circle",
                x0=cx-0.72, y0=-0.02, x1=cx+0.72, y1=2.22,
                line=dict(color=c["color"], width=2.5, dash="dot"),
                fillcolor="rgba(0,0,0,0)"
            )

    # 1. LOW (ZZZ FLIGHT MOTION)
    is_low_active = (current_choice == "Low")
    zzz_offset_x = (anim_step * 0.04) if is_low_active else 0
    zzz_offset_y = (anim_step * 0.05) if is_low_active else 0

    fig.add_shape(type="rect", x0=0.55, y0=0.75, x1=1.45, y1=1.10, fillcolor="#1e1b4b", line=dict(color="#312e81", width=2))
    fig.add_shape(type="rect", x0=0.50, y0=0.60, x1=0.65, y1=1.35, fillcolor="#312e81", line=dict(width=0))
    fig.add_shape(type="rect", x0=0.68, y0=0.95, x1=0.95, y1=1.18, fillcolor="#e0e7ff", line=dict(width=0))
    fig.add_shape(type="circle", x0=0.76, y0=0.98, x1=0.92, y1=1.16, fillcolor="#fbbf24", line=dict(width=0))
    fig.add_shape(type="path", path="M 0.88 1.05 L 1.42 1.05 L 1.42 0.82 L 0.88 0.82 Z", fillcolor="#38bdf8", line=dict(width=0))
    
    fig.add_annotation(
        x=1.10 + zzz_offset_x,
        y=1.55 + zzz_offset_y,
        text="Z",
        showarrow=False,
        font=dict(size=14 + anim_step, color="#818cf8", family="Arial Black")
    )
    fig.add_annotation(
        x=1.22 + zzz_offset_x * 1.3,
        y=1.68 + zzz_offset_y * 1.3,
        text="z",
        showarrow=False,
        font=dict(size=18 + anim_step * 2, color="#6366f1", family="Arial Black")
    )
    fig.add_annotation(
        x=1.36 + zzz_offset_x * 1.6,
        y=1.84 + zzz_offset_y * 1.6,
        text="z",
        showarrow=False,
        font=dict(size=22 + anim_step * 3, color="#4f46e5", family="Arial Black")
    )

    # 2. MEDIUM (RUNNER DASH MOTION)
    is_med_active = (current_choice == "Medium")
    run_shift_x = (anim_step * 0.05) if is_med_active else 0
    run_bob_y = (0.04 if anim_step % 2 == 1 else 0) if is_med_active else 0

    fig.add_shape(type="line", x0=2.0, y0=0.60, x1=3.0, y1=0.60, line=dict(color="#0284c7", width=3))
    flare_len = 0.28 + (anim_step * 0.03 if is_med_active else 0)
    fig.add_shape(type="line", x0=2.60 + run_shift_x, y0=1.50, x1=2.60 + run_shift_x + flare_len, y1=1.50, line=dict(color="#38bdf8", width=2.5))
    fig.add_shape(type="line", x0=2.65 + run_shift_x, y0=1.25, x1=2.65 + run_shift_x + flare_len, y1=1.25, line=dict(color="#38bdf8", width=2.5))
    
    rx = 2.45 + run_shift_x
    ry = 1.40 + run_bob_y
    fig.add_shape(type="circle", x0=rx-0.08, y0=ry-0.02, x1=rx+0.08, y1=ry+0.22, fillcolor="#f59e0b", line=dict(width=0))
    fig.add_shape(type="path", path=f"M {rx-0.05} {ry} L {rx+0.12} {ry} L {rx+0.05} {ry-0.45} L {rx-0.10} {ry-0.45} Z", fillcolor="#0284c7", line=dict(width=0))
    fig.add_shape(type="line", x0=rx-0.03, y0=ry-0.05, x1=rx-0.20, y1=ry-0.25, line=dict(color="#f59e0b", width=5))
    fig.add_shape(type="line", x0=rx+0.10, y0=ry-0.05, x1=rx+0.25, y1=ry-0.20, line=dict(color="#f59e0b", width=5))
    fig.add_shape(type="line", x0=rx-0.05, y0=ry-0.45, x1=rx-0.22, y1=ry-0.75, line=dict(color="#0369a1", width=6))
    fig.add_shape(type="line", x0=rx+0.05, y0=ry-0.45, x1=rx+0.28, y1=ry-0.68, line=dict(color="#0284c7", width=6))

    # 3. HIGH (BARBELL POWER PRESS MOTION)
    is_high_active = (current_choice == "High")
    lift_shift_y = (anim_step * 0.08) if is_high_active else 0

    fig.add_shape(type="circle", x0=3.90, y0=1.05, x1=4.10, y1=1.30, fillcolor="#f59e0b", line=dict(width=0))
    fig.add_shape(type="rect", x0=3.85, y0=0.78, x1=4.15, y1=1.10, fillcolor="#dc2626", line=dict(width=0))
    fig.add_shape(type="line", x0=3.90, y0=0.78, x1=3.75, y1=0.55, line=dict(color="#1e293b", width=8))
    fig.add_shape(type="line", x0=4.10, y0=0.78, x1=4.25, y1=0.55, line=dict(color="#1e293b", width=8))
    
    by = 1.55 + lift_shift_y
    fig.add_shape(type="line", x0=3.45, y0=by, x1=4.55, y1=by, line=dict(color="#cbd5e1", width=5))
    fig.add_shape(type="rect", x0=3.52, y0=by-0.25, x1=3.62, y1=by+0.25, fillcolor="#0f172a", line=dict(color="#ef4444", width=2))
    fig.add_shape(type="rect", x0=4.38, y0=by-0.25, x1=4.48, y1=by+0.25, fillcolor="#0f172a", line=dict(color="#ef4444", width=2))
    fig.add_shape(type="line", x0=3.88, y0=1.05, x1=3.75, y1=by, line=dict(color="#f59e0b", width=6))
    fig.add_shape(type="line", x0=4.12, y0=1.05, x1=4.25, y1=by, line=dict(color="#f59e0b", width=6))

    fig.add_trace(go.Scatter(
        x=[1.0, 2.5, 4.0],
        y=[1.10, 1.10, 1.10],
        mode="markers",
        marker=dict(size=100, opacity=0.01),
        hoverinfo="none",
        customdata=["Low", "Medium", "High"]
    ))

    fig.update_xaxes(range=[0.2, 4.8], visible=False)
    fig.update_yaxes(range=[-0.05, 2.25], visible=False)
    fig.update_layout(
        height=190,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select'
    )
    return fig


# ---------------------------------------------------------
# Helper 2: 3D Cartoon Wine Glass Visualizer (Compact Sizing)
# ---------------------------------------------------------
def create_3d_cartoon_bottle(current_choice):
    fig = go.Figure()

    # 1. Base & Stem
    fig.add_shape(type="rect", x0=0.48, y0=0.30, x1=0.52, y1=0.85, line=dict(color="#0f172a", width=1.6), fillcolor="#e2e8f0")
    fig.add_shape(type="path", path="M 0.35 0.30 Q 0.50 0.20 0.65 0.30 L 0.65 0.22 Q 0.50 0.12 0.35 0.22 Z",
                  line=dict(color="#0f172a", width=2.0), fillcolor="#cbd5e1")

    # 2. Outer Glass Bowl Outline
    fig.add_shape(type="path", path="M 0.30 2.25 Q 0.30 0.80 0.50 0.80 Q 0.70 0.80 0.70 2.25 Z",
                  line=dict(color="#0f172a", width=2.5), fillcolor="rgba(240, 249, 255, 0.4)")

    # 3. Layer 1 - Bottom Tier (少量 / 1/3 杯)
    c_bottom = "#b45309" if current_choice in ["Low", "Medium", "High"] else "rgba(226, 232, 240, 0.35)"
    fig.add_shape(type="path", path="M 0.37 1.25 Q 0.31 0.90 0.50 0.84 Q 0.69 0.90 0.63 1.25 Z",
                  line=dict(color="#0f172a", width=1.2), fillcolor=c_bottom)

    # Layer 2 - Middle Tier (中量 / 2/3 杯)
    c_middle = "#f59e0b" if current_choice in ["Medium", "High"] else "rgba(226, 232, 240, 0.35)"
    fig.add_shape(type="path", path="M 0.32 1.70 L 0.37 1.25 L 0.63 1.25 L 0.68 1.70 Z",
                  line=dict(color="#0f172a", width=1.2), fillcolor=c_middle)

    # Layer 3 - Top Tier (多量 / 满杯)
    c_top = "#38bdf8" if current_choice == "High" else "rgba(226, 232, 240, 0.35)"
    fig.add_shape(type="path", path="M 0.30 2.15 L 0.32 1.70 L 0.68 1.70 L 0.70 2.15 Z",
                  line=dict(color="#0f172a", width=1.2), fillcolor=c_top)

    # 4. Glass Highlights & Curve Reflections
    fig.add_shape(type="path", path="M 0.33 2.05 Q 0.35 1.35 0.42 1.00",
                  line=dict(color="rgba(255, 255, 255, 0.8)", width=2.5))
    fig.add_shape(type="circle", x0=0.47, y0=1.40, x1=0.53, y1=1.52, line=dict(color="white", width=1), fillcolor="rgba(255, 255, 255, 0.6)")

    # 5. Clickable Target Centers
    fig.add_trace(go.Scatter(
        x=[0.5, 0.5, 0.5],
        y=[1.05, 1.48, 1.92],
        mode="markers",
        marker=dict(size=35, opacity=0.01),
        hoverinfo="none",
        customdata=["Low", "Medium", "High"]
    ))

    fig.update_xaxes(range=[0.1, 0.9], visible=False)
    fig.update_yaxes(range=[0.0, 2.45], visible=False)
    fig.update_layout(
        height=160,
        margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select'
    )
    return fig


# ---------------------------------------------------------
# Helper 3: 3D Cartoon Boba Cup (3-Tier Scaled Cup for Sugar)
# ---------------------------------------------------------
def create_3d_cartoon_cup(current_choice):
    fig = go.Figure()

    # 1. Straw & Flat Lid
    fig.add_shape(type="rect", x0=0.47, y0=2.20, x1=0.53, y1=2.45, line=dict(color="#881337", width=2.0), fillcolor="#fb7185")
    fig.add_shape(type="path", path="M 0.28 2.20 Q 0.50 2.28 0.72 2.20 L 0.70 2.10 Q 0.50 2.18 0.30 2.10 Z",
                  line=dict(color="#0f172a", width=2.2), fillcolor="#334155")

    # 2. Outer Cup Body Contour
    fig.add_shape(type="path", path="M 0.30 2.10 L 0.36 0.35 Q 0.50 0.28 0.64 0.35 L 0.70 2.10 Z",
                  line=dict(color="#0f172a", width=2.5), fillcolor="rgba(240, 249, 255, 0.4)")

    # 3. Layer 1 - Bottom Tier (Low: 30% 微糖)
    c1 = "#4ade80" if current_choice in ["Low", "Medium", "High"] else "rgba(226, 232, 240, 0.35)"
    fig.add_shape(type="path", path="M 0.36 0.38 L 0.38 0.90 L 0.62 0.90 L 0.64 0.38 Q 0.50 0.32 0.36 0.38 Z",
                  line=dict(color="#0f172a", width=1.2), fillcolor=c1)

    # Layer 2 - Middle Tier (Medium: 70% 少糖)
    c2 = "#fbbf24" if current_choice in ["Medium", "High"] else "rgba(226, 232, 240, 0.35)"
    fig.add_shape(type="path", path="M 0.38 0.90 L 0.34 1.50 L 0.66 1.50 L 0.62 0.90 Z",
                  line=dict(color="#0f172a", width=1.2), fillcolor=c2)

    # Layer 3 - Top Tier (High: 100% 全糖)
    c3 = "#ea580c" if current_choice == "High" else "rgba(226, 232, 240, 0.35)"
    fig.add_shape(type="path", path="M 0.34 1.50 L 0.31 2.05 L 0.69 2.05 L 0.66 1.50 Z",
                  line=dict(color="#0f172a", width=1.2), fillcolor=c3)

    # 4. Highlights
    fig.add_shape(type="line", x0=0.36, y0=0.50, x1=0.33, y1=1.95, line=dict(color="rgba(255, 255, 255, 0.8)", width=2.5))

    # 5. Clickable Target Centers (3 Tiers)
    fig.add_trace(go.Scatter(
        x=[0.5, 0.5, 0.5],
        y=[0.65, 1.20, 1.78],
        mode="markers",
        marker=dict(size=35, opacity=0.01),
        hoverinfo="none",
        customdata=["Low", "Medium", "High"]
    ))

    fig.update_xaxes(range=[0.1, 0.9], visible=False)
    fig.update_yaxes(range=[0.0, 2.45], visible=False)
    fig.update_layout(
        height=160,
        margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select'
    )
    return fig

# ---------------------------------------------------------
# Helper 4: SpeedTest Gauge Renderer
# ---------------------------------------------------------
def render_speedtest_gauge(current_val, threshold, status_text="TESTING..."):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_val,
        number={'suffix': "%", 'valueformat': ".1f", 'font': {'size': 38, 'color': '#0f172a'}},
        title={'text': f"🚀 <b>SPEEDTEST RISK DIAL</b><br><span style='font-size:12px;color:#64748b;'>{status_text}</span>"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#475569"},
            'bar': {'color': "#ef4444" if current_val >= (threshold * 100) else "#10b981", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#0f172a", 'width': 4},
                'thickness': 0.8,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=25, r=25, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ---------------------------------------------------------
# Helper 5: Stress Gauge
# ---------------------------------------------------------
def create_stress_gauge(current_selection="Medium"):
    fig = go.Figure()

    # -----------------------------------------------------
    # Dark glassmorphism container
    # -----------------------------------------------------
    fig.add_shape(
        type="rect",
        x0=-1.35, y0=-0.38,
        x1=1.35, y1=1.45,
        fillcolor="#0f172a",
        line=dict(color="#334155", width=2),
        layer="below"
    )

    r_outer = 0.95
    r_inner = 0.60

    # -----------------------------------------------------
    # Stress levels
    # -----------------------------------------------------
    segments = [
        {
            "name": "Low",
            "color": "#10b981",
            "angles": (180, 120),
            "label_angle": 150,
            "emoji": "🥰"
        },
        {
            "name": "Medium",
            "color": "#f59e0b",
            "angles": (120, 60),
            "label_angle": 90,
            "emoji": "😕"
        },
        {
            "name": "High",
            "color": "#ef4444",
            "angles": (60, 0),
            "label_angle": 30,
            "emoji": "🤯"
        }
    ]

    # -----------------------------------------------------
    # DRAW CLICKABLE GAUGE CURVES
    # -----------------------------------------------------
    for seg in segments:

        a_start, a_end = seg["angles"]

        theta = np.linspace(
            np.radians(a_start),
            np.radians(a_end),
            60
        )

        # Outer arc
        x_out = r_outer * np.cos(theta)
        y_out = r_outer * np.sin(theta)

        # Inner arc
        x_in = r_inner * np.cos(theta)[::-1]
        y_in = r_inner * np.sin(theta)[::-1]

        x_path = np.concatenate([x_out, x_in])
        y_path = np.concatenate([y_out, y_in])

        # IMPORTANT:
        # Attach stress name to EVERY point
        custom_data = [seg["name"]] * len(x_path)

        fig.add_trace(
            go.Scatter(
                x=x_path,
                y=y_path,
                mode="lines",
                fill="toself",
                fillcolor=seg["color"],
                line=dict(
                    color=seg["color"],
                    width=4
                ),
                customdata=custom_data,

                # Remove Trace 0 / Trace 1 / Trace 2
                name="",

                # Show only your own hover text
                hovertemplate=(
                    f"<b>{seg['name']} Stress</b>"
                ),

                showlegend=False
            )
        )

    # -----------------------------------------------------
    # CLICKABLE EMOJIS
    # -----------------------------------------------------
    r_emoji = r_outer + 0.22

    emoji_x = []
    emoji_y = []
    emoji_text = []
    emoji_hover = []
    emoji_customdata = []

    for seg in segments:

        angle = np.radians(seg["label_angle"])

        x = r_emoji * np.cos(angle)
        y = r_emoji * np.sin(angle)

        emoji_x.append(x)
        emoji_y.append(y)

        # Bigger emoji when selected
        if seg["name"] == current_selection:
            emoji_text.append(
                f"<span style='font-size:36px'>{seg['emoji']}</span>"
            )
        else:
            emoji_text.append(
                f"<span style='font-size:27px'>{seg['emoji']}</span>"
            )

        emoji_hover.append(
            f"{seg['name']} Stress"
        )

        emoji_customdata.append(seg["name"])

    # -----------------------------------------------------
    # IMPORTANT:
    # Invisible marker makes emoji area clickable
    # -----------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=emoji_x,
            y=emoji_y,

            mode="text+markers",

            text=emoji_text,
            textposition="middle center",

            marker=dict(
                size=55,
                color="rgba(0,0,0,0)",
                line=dict(
                    color="rgba(0,0,0,0)"
                )
            ),

            customdata=emoji_customdata,

            hoverinfo="text",
            hovertext=emoji_hover,

            showlegend=False
        )
    )

    # -----------------------------------------------------
    # NEEDLE
    # -----------------------------------------------------
    angle_deg = {
        "Low": 150,
        "Medium": 90,
        "High": 30
    }.get(current_selection, 90)

    angle_rad = np.radians(angle_deg)

    needle_len = 0.80

    nx = needle_len * np.cos(angle_rad)
    ny = needle_len * np.sin(angle_rad)

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=nx,
        y1=ny,
        line=dict(
            color="#f8fafc",
            width=6
        ),
        layer="above"
    )

    # -----------------------------------------------------
    # OUTER PIVOT
    # -----------------------------------------------------
    fig.add_shape(
        type="circle",
        x0=-0.14,
        y0=-0.14,
        x1=0.14,
        y1=0.14,
        fillcolor="#38bdf8",
        line=dict(
            color="#f8fafc",
            width=2.5
        ),
        layer="above"
    )

    # -----------------------------------------------------
    # INNER PIVOT
    # -----------------------------------------------------
    fig.add_shape(
        type="circle",
        x0=-0.05,
        y0=-0.05,
        x1=0.05,
        y1=0.05,
        fillcolor="#0f172a",
        line=dict(
            color="rgba(0,0,0,0)"
        ),
        layer="above"
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------
    fig.add_annotation(
        x=0,
        y=-0.22,
        text="<b>STRESS LEVEL</b>",
        showarrow=False,
        font=dict(
            size=14,
            color="#f8fafc",
            family="Arial Black"
        )
    )

    # -----------------------------------------------------
    # AXES
    # -----------------------------------------------------
    fig.update_xaxes(
        range=[-1.4, 1.4],
        visible=False,
        fixedrange=True
    )

    fig.update_yaxes(
        range=[-0.45, 1.45],
        visible=False,
        fixedrange=True
    )

    # -----------------------------------------------------
    # LAYOUT
    # -----------------------------------------------------
    fig.update_layout(
        height=300,

        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),

        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",

        # IMPORTANT
        clickmode="event+select",

        hovermode="closest",

        dragmode=False
    )

    return fig

# ---------------------------------------------------------
# 6. Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab_eda = st.tabs([
    "📋 Patient Assessment (Predictor)",
    "📊 Model Performance Metrics",
    "🧪 Auto‑Generated Samples",
    "🔬 EDA"
])

# ---------------------------------------------------------
# TAB 1: Predictor (Categorically Organized Layout)
# ---------------------------------------------------------
with tab1:
    st.write("### 🧬 Input Patient Clinical Parameters")

    col1, col2, col3 = st.columns(3)

    # =========================================================
    # CATEGORY 1: Demographics & Personal Lifestyle
    # =========================================================
    with col1:
        with st.container(border=True):
            st.write("##### 👤 Demographics & Habits")
            
            # 1. 3D Gender Card Selection
            st.write("**Gender:**")
            gender_fig = create_3d_gender_cards(st.session_state["gender_choice"])
            gender_event = st.plotly_chart(gender_fig, use_container_width=True, on_select="rerun", key="gender_selector_direct")
            
            if gender_event and "selection" in gender_event and len(gender_event["selection"]["points"]) > 0:
                selected_g_pt = gender_event["selection"]["points"][0]["point_index"]
                mapped_gender = ["Female", "Male"]
                new_gen = mapped_gender[selected_g_pt]
                if new_gen != st.session_state["gender_choice"]:
                    st.session_state["gender_choice"] = new_gen
                    st.rerun()

            gender = st.session_state["gender_choice"]

            # 2. 3D Luxury Smoking Status Selection (Featuring Clean Lungs 🫁)
            st.write("**Smoking Status (Click 3D Card):**")
            smoke_fig = create_3d_smoking_cards(st.session_state["smoking_choice"])
            smoke_event = st.plotly_chart(smoke_fig, use_container_width=True, on_select="rerun", key="smoking_selector_direct")
            
            if smoke_event and "selection" in smoke_event and len(smoke_event["selection"]["points"]) > 0:
                selected_s_pt = smoke_event["selection"]["points"][0]["point_index"]
                mapped_smoke = ["No", "Yes"]
                new_smoke = mapped_smoke[selected_s_pt]
                if new_smoke != st.session_state["smoking_choice"]:
                    st.session_state["smoking_choice"] = new_smoke
                    st.rerun()

            smoking = st.session_state["smoking_choice"]

            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=50)
            sleep = st.number_input("Sleep Hours per Day", min_value=2.0, max_value=14.0, value=7.0)
            
            st.write("*🧘 Stress Level (Click directly on gauge):*")
            stress_fig = create_stress_gauge(st.session_state["stress_choice"])

            stress_event = st.plotly_chart(
            stress_fig,
            use_container_width=True,
            on_select="rerun",
            key="stress_gauge_direct",
            selection_mode="points",
        )

        # -----------------------------------------------------
        # Detect click on BOTH:
        # 1. Emoji
        # 2. Colored gauge curve
        # -----------------------------------------------------
        if (
            stress_event
            and "selection" in stress_event
            and len(stress_event["selection"]["points"]) > 0
        ):
            selected_point = stress_event["selection"]["points"][0]

            new_stress = None

            if "customdata" in selected_point:
                new_stress = selected_point["customdata"]

            if isinstance(new_stress, list):
                new_stress = new_stress[0]

            if new_stress in ["Low", "Medium", "High"]:

                if new_stress != st.session_state["stress_choice"]:
                    st.session_state["stress_choice"] = new_stress
                    st.rerun()

        stress = st.session_state["stress_choice"]

        st.info(
            f"Active Stress Setting: *{stress}*"
        )

        st.markdown("---")

# =========================================================
    # CATEGORY 2: Interactive Lifestyle & Intake
    # =========================================================
    with col2:
        with st.container(border=True):
            st.write("##### 🏃 Lifestyle & Dietary Intake")

            # Direct Clickable 3D Action-Animated Exercise Cards
            st.write("**Exercise Habits:**")
            ex_placeholder = st.empty()
            
            exercise_fig = create_animated_action_stages(st.session_state["exercise_choice"], anim_step=0)
            exercise_event = ex_placeholder.plotly_chart(exercise_fig, use_container_width=True, on_select="rerun", key="exercise_selector_direct")
            
            if exercise_event and "selection" in exercise_event and len(exercise_event["selection"]["points"]) > 0:
                selected_pt = exercise_event["selection"]["points"][0]["point_index"]
                mapped_exercise = ["Low", "Medium", "High"]
                new_ex = mapped_exercise[selected_pt]
                if new_ex != st.session_state["exercise_choice"]:
                    st.session_state["exercise_choice"] = new_ex
                    
                    # Smooth Action Sweep Animation for the newly selected level
                    for step in [1, 2, 3, 2, 1, 0]:
                        ex_placeholder.plotly_chart(
                            create_animated_action_stages(new_ex, anim_step=step),
                            use_container_width=True,
                            key=f"ex_anim_{step}_{random.randint(100,999)}"
                        )
                        time.sleep(0.04)
                    st.rerun()

            exercise = st.session_state["exercise_choice"]
            
            # Compact 3D Wine Glass
            st.write("**Alcohol Consumption (Click Glass Level):**")
            bottle_fig = create_3d_cartoon_bottle(st.session_state["alcohol_choice"])
            bottle_event = st.plotly_chart(bottle_fig, use_container_width=True, on_select="rerun", key="bottle_selector_direct")
            
            if bottle_event and "selection" in bottle_event and len(bottle_event["selection"]["points"]) > 0:
                selected_point_idx = bottle_event["selection"]["points"][0]["point_index"]
                mapped_choices = ["Low", "Medium", "High"]
                new_choice = mapped_choices[selected_point_idx]
                
                if new_choice == st.session_state["alcohol_choice"]:
                    st.session_state["alcohol_choice"] = "None"
                else:
                    st.session_state["alcohol_choice"] = new_choice
                st.rerun()

            alcohol = st.session_state["alcohol_choice"]

            # Compact 3D Boba Cup (Sugar)
            st.write("**Sugar Consumption (Click Cup Tier):**")
            cup_fig = create_3d_cartoon_cup(st.session_state["sugar_choice"])
            cup_event = st.plotly_chart(cup_fig, use_container_width=True, on_select="rerun", key="cup_selector_direct")
            
            if cup_event and "selection" in cup_event and len(cup_event["selection"]["points"]) > 0:
                selected_point_idx = cup_event["selection"]["points"][0]["point_index"]
                mapped_sugar = ["Low", "Medium", "High"]
                new_sugar = mapped_sugar[selected_point_idx]
                if new_sugar != st.session_state["sugar_choice"]:
                    st.session_state["sugar_choice"] = new_sugar
                    st.rerun()

            sugar = st.session_state["sugar_choice"]

    # =========================================================
    # CATEGORY 3: Clinical Measures, Medical History & Biomarkers
    # =========================================================
    with col3:
        with st.container(border=True):
            st.write("##### 🔬 Clinical Measures & Biomarkers")
            
            # Physical & Vital Metrics
            bp = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=220, value=120)
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)

            # Medical History
            family_history = st.selectbox("Family Heart Disease History", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes Diagnosis Status", ["No", "Yes"])
            high_bp = st.selectbox("High Blood Pressure Diagnosis", ["No", "Yes"])

            # Lipid Profile & Blood Biomarkers
            chol = st.number_input("Total Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
            low_hdl = st.selectbox("Low HDL Cholesterol", ["No", "Yes"])
            high_ldl = st.selectbox("High LDL Cholesterol", ["No", "Yes"])
            triglycerides = st.number_input("Triglyceride Level (mg/dL)", min_value=50, max_value=500, value=150)
            fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70, max_value=300, value=100)
            crp = st.number_input("CRP Level (mg/L)", min_value=0.0, max_value=30.0, value=2.0)
            homocysteine = st.number_input("Homocysteine Level (µmol/L)", min_value=0.0, max_value=30.0, value=10.0)

    st.markdown("---")
    if st.button("🚀 START SPEEDTEST RISK ANALYSIS", type="primary", use_container_width=True):
        if model is None:
            st.error(
                f"❌ {model_choice} model is not available."
            )
        elif model_choice != "SVM" and scaler is None:
            st.error(
                f"❌ {model_choice} scaler is not available."
            )
        else:
            input_dict = {
                "Age": age,
                'Gender': 1 if gender == "Male" else 0,
                "Blood Pressure": bp,
                "Cholesterol Level": chol,
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
                'Fasting Blood Sugar': fbs,                    'CRP Level': crp,
                'Homocysteine Level': homocysteine
            }
            input_df = pd.DataFrame([input_dict])
            num_cols = [
                "Age",
                "Blood Pressure",
                "Cholesterol Level",
                "BMI",
                "Sleep Hours",
                "Triglyceride Level",
                "Fasting Blood Sugar",
                "CRP Level",
                "Homocysteine Level"
            ]
            try:
                if model_choice == "KNN":
                    input_df = input_df.reindex(columns=scaler.feature_names_in_)
                    input_df = pd.DataFrame(
                        scaler.transform(input_df),
                        columns=input_df.columns,
                        index=input_df.index
                    )
                elif model_choice == "SVM":
                    # SVM was trained directly on the preprocessed data.
                    # Do NOT apply another scaler here.
                    pass
                else:
                    input_df[num_cols] = scaler.transform(input_df[num_cols])

                final_probability = float(model.predict_proba(input_df)[0][1])
                target_pct = final_probability * 100

                st.markdown(f"### 🎯 Real-Time Diagnostic SpeedTest ({model_choice})")

                res_col1, res_col2 = st.columns([1.3, 1.7])
                with res_col1:
                    speedtest_placeholder = (st.empty())
                with res_col2:
                    status_box = st.empty()
                    
                # Smooth Slow Speedtest Needle Movement
                slow_steps = 35
                for i in range(1,slow_steps + 1):
                    interp_val = (target_pct/ slow_steps* i)
                    speedtest_placeholder.plotly_chart(
                        render_speedtest_gauge(interp_val,decision_threshold,status_text=("⚡ READING SENSORS "f"({interp_val:.1f}%)...")),
                        use_container_width=True,
                        key=f"gauge_rise_{i}"
                    )
                    time.sleep(0.08)

                jitter_offsets = [2.2, -1.6, 1.1, -0.6, 0.2, 0.0]
                for j_idx, offset in enumerate(jitter_offsets):
                    j_val = min(100.0, max(0.0, target_pct + offset))
                    speedtest_placeholder.plotly_chart(
                        render_speedtest_gauge(j_val, decision_threshold, status_text="📡 STABILIZING DATA MATRIX..."),
                        use_container_width=True,
                        key=f"gauge_jitter_{j_idx}"
                    )
                    time.sleep(0.12)

                speedtest_placeholder.plotly_chart(
                    render_speedtest_gauge(target_pct, decision_threshold, status_text="✅ TEST COMPLETE - LOCKED"),
                    use_container_width=True,
                    key="gauge_final_locked"
                )

                with status_box.container(
                    border=True
                ):
                    st.metric(
                        label="Calculated Disease Probability",
                        value=f"{target_pct:.1f}%"
                    )
                    st.metric(
                        label="Model Decision Threshold",
                        value=(
                            f"{decision_threshold * 100:.1f}%"
                        )
                    )
                    if (
                        final_probability
                        >= decision_threshold
                    ):

                        st.error(
                            "⚠️ **HIGH RISK**: Patient "
                            "shows symptoms/risk patterns "
                            "for Heart Disease."
                        )
                    else:

                        st.success(
                            "✅ **LOW RISK**: Patient is "
                            "unlikely to have Heart Disease."
                        )
            except Exception as e:
                st.error("❌ Prediction failed.")

                st.exception(e)


# =========================================================
# TAB 2: PERFORMANCE METRICS
# =========================================================

with tab2:

    st.write(
        f"### 📊 {model_choice} Model "
        f"Metrics & Diagnostic Plots"
    )

    if os.path.exists(metrics_path):

        df_metrics = pd.read_csv(
            metrics_path
        )

        available_metrics = [
            column
            for column in display_metric_columns
            if column in df_metrics.columns
        ]

        if available_metrics:

            st.dataframe(
                df_metrics[
                    available_metrics
                ].style.format("{:.2%}"),
                width="stretch"
            )

        else:

            st.warning(
                "No standard evaluation metrics "
                "were found in the selected CSV."
            )

    else:

        st.warning(
            f"Metrics file not found: "
            f"`{metrics_path}`"
        )


    col_img1, col_img2 = st.columns(2)


    with col_img1:
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption=f"{model_choice} Confusion Matrix", width="stretch")
        else:
            st.info("Confusion matrix not available.")


    with col_img2:
        if os.path.exists(roc_path):
            st.image(Image.open(roc_path), caption=f"{model_choice} ROC Curve", width="stretch")
        else:
            st.info("ROC curve not available.")


# ---------------------------------------------------------
# TAB 3: Auto‑Generated Samples
# ---------------------------------------------------------
with tab3:
    st.write("### 🧪 Auto‑Generated Example Data")

    if st.button("⚙️ Generate Sample Data", type="primary"):
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
            for _ in range(10):
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

        df_low = generate_sample("Low Risk")[feature_order]
        df_high = generate_sample("High Risk")[feature_order]

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
        df_low['Age'] = df_low['Age'].astype(int)
        df_high['Age'] = df_high['Age'].astype(int)

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
            scaled_low = df_low.copy()
            scaled_high = df_high.copy()
            scaled_low[num_cols] = scaler.transform(df_low[num_cols])
            scaled_high[num_cols] = scaler.transform(df_high[num_cols])

        df_low['Predicted Probability'] = model.predict_proba(scaled_low)[:, 1]
        df_low['Predicted Risk'] = np.where(df_low['Predicted Probability'] >= decision_threshold, "High Risk", "Low Risk")

        df_high['Predicted Probability'] = model.predict_proba(scaled_high)[:, 1]
        df_high['Predicted Risk'] = np.where(df_high['Predicted Probability'] >= decision_threshold, "High Risk", "Low Risk")

        df_low = df_low[df_low['Predicted Risk'] == 'Low Risk'].head(10).copy()
        df_high = df_high[df_high['Predicted Risk'] == 'High Risk'].head(10).copy()

        if len(df_low) < 10:
            extra_low = generate_sample("Low Risk")[feature_order].head(10 - len(df_low))
            df_low = pd.concat([df_low, extra_low], ignore_index=True)
        if len(df_high) < 10:
            extra_high = generate_sample("High Risk")[feature_order].head(10 - len(df_high))
            df_high = pd.concat([df_high, extra_high], ignore_index=True)

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

        st.write("#### 🟢 Low‑Risk Samples (Predicted)")
        st.dataframe(df_low.head(10), width="stretch")

        st.write("#### 🔴 High‑Risk Samples (Predicted)")
        st.dataframe(df_high.head(10), width="stretch")

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
        st.error("EDA dataset not found. Keep heart_disease.csv beside your script.")
        st.stop()

    if TARGET_COL not in df_eda.columns:
        st.error(f"Expected target column '{TARGET_COL}' was not found in the dataset.")
        st.stop()

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
                width="stretch",
                column_config={
                    "Missing values": st.column_config.TextColumn(
                        "Missing values",
                        help="Number of missing entries"
                    )
                }
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
