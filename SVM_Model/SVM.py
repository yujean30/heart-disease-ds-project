
# =========================================================
# SVM HEART DISEASE PREDICTION
# Fine-tuned SVM with SMOTENC and Threshold Optimization
# =========================================================

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.svm import SVC


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    files = {
        "X_train": ROOT / "Preprocessing" / "X_train_preprocessed.csv",
        "X_test": ROOT / "Preprocessing" / "X_test_preprocessed.csv",
        "y_train": ROOT / "Preprocessing" / "y_train.csv",
        "y_test": ROOT / "Preprocessing" / "y_test.csv"
    }

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}. "
                "Please run preprocess.py first."
            )

    X_train = pd.read_csv(files["X_train"])
    X_test = pd.read_csv(files["X_test"])

    y_train = (
        pd.read_csv(files["y_train"])
        .squeeze("columns")
        .to_numpy()
    )

    y_test = (
        pd.read_csv(files["y_test"])
        .squeeze("columns")
        .to_numpy()
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# DATA CHECK
# =========================================================

def check_data(X_train, X_test, y_train, y_test):

    print("\nDATA CHECK")

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape : {X_test.shape}")

    print(
        "Training class balance:",
        pd.Series(y_train).value_counts().to_dict()
    )

    print(
        "Testing class balance :",
        pd.Series(y_test).value_counts().to_dict()
    )

    print(
        "Train/Test columns match:",
        list(X_train.columns) == list(X_test.columns)
    )

    print(
        "Training missing values:",
        X_train.isnull().sum().sum()
    )

    print(
        "Testing missing values :",
        X_test.isnull().sum().sum()
    )

    if list(X_train.columns) != list(X_test.columns):

        raise ValueError(
            "Training and testing columns do not match."
        )

    if (
        X_train.isnull().sum().sum() > 0
        or X_test.isnull().sum().sum() > 0
    ):

        raise ValueError(
            "Missing values remain after preprocessing."
        )


# =========================================================
# SMOTENC
# =========================================================

def create_smote(X_train):

    categorical_columns = [
        "Gender",
        "Exercise Habits",
        "Smoking",
        "Family Heart Disease",
        "Diabetes",
        "High Blood Pressure",
        "Low HDL Cholesterol",
        "High LDL Cholesterol",
        "Alcohol Consumption",
        "Stress Level",
        "Sugar Consumption"
    ]

    categorical_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_columns
        if col in X_train.columns
    ]

    return SMOTENC(
        categorical_features=categorical_indices,
        random_state=RANDOM_STATE,
        k_neighbors=5
    )


# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

def analyze_thresholds(y_true, probabilities):

    thresholds = np.linspace(0.20, 0.80, 61)

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append({

            "threshold": threshold,

            "accuracy": accuracy_score(
                y_true,
                predictions
            ),

            "precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "f1": f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "positive_rate": predictions.mean()
        })

    return pd.DataFrame(results)


# =========================================================
# CHOOSE BEST THRESHOLD
# =========================================================

def choose_threshold(results_df):

    reasonable = results_df[
        (results_df["positive_rate"] >= 0.10)
        &
        (results_df["positive_rate"] <= 0.50)
    ]

    if reasonable.empty:

        best = results_df.sort_values(
            ["f1", "precision", "recall"],
            ascending=False
        ).iloc[0]

    else:

        best = reasonable.sort_values(
            ["f1", "precision", "recall"],
            ascending=False
        ).iloc[0]

    print("\nOOF THRESHOLD ANALYSIS")

    print(
        f"Chosen threshold : {best['threshold']:.4f}"
    )

    print(
        f"OOF Accuracy     : {best['accuracy']:.4f}"
    )

    print(
        f"OOF Precision    : {best['precision']:.4f}"
    )

    print(
        f"OOF Recall       : {best['recall']:.4f}"
    )

    print(
        f"OOF F1           : {best['f1']:.4f}"
    )

    print(
        f"OOF Positive Rate: {best['positive_rate']:.4f}"
    )

    return float(best["threshold"])


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities
        )
    }

    return metrics, predictions


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    check_data(
        X_train,
        X_test,
        y_train,
        y_test
    )


    # -----------------------------------------------------
    # SMOTENC CONFIGURATION
    # -----------------------------------------------------

    print("\nSMOTENC CONFIGURATION")

    print(
        "SMOTENC is applied INSIDE CV."
    )

    print(
        "Test data will NOT be oversampled."
    )


    # -----------------------------------------------------
    # SVM PIPELINE
    # -----------------------------------------------------

    pipeline = Pipeline([

        (
            "smote",
            create_smote(X_train)
        ),

        (
            "svm",
            SVC(
                probability=True,
                cache_size=1000,
                random_state=RANDOM_STATE
            )
        )
    ])


    print("\nSVM MODEL")

    print(
        "Main algorithm: Support Vector Machine (SVM)"
    )

    print(
        "Combination method: None"
    )


    # =====================================================
    # BEFORE TUNING
    # =====================================================

    print(
        "\nBEFORE TUNING: SVM MODEL RESULTS"
    )

    baseline_model = Pipeline([

        (
            "smote",
            create_smote(X_train)
        ),

        (
            "svm",
            SVC(
                probability=True,
                cache_size=1000,
                random_state=RANDOM_STATE
            )
        )
    ])


    baseline_model.fit(
        X_train,
        y_train
    )


    baseline_probas = (
        baseline_model
        .predict_proba(X_test)[:, 1]
    )


    baseline_metrics, _ = calculate_metrics(
        y_test,
        baseline_probas,
        0.50
    )


    print(
        pd.DataFrame(
            [baseline_metrics]
        ).to_string(index=False)
    )


    # =====================================================
    # CROSS-VALIDATION
    # =====================================================

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=RANDOM_STATE
    )


    # =====================================================
    # SVM PARAMETER SEARCH
    # =====================================================

    param_distributions = {

        "svm__kernel": [
            "rbf",
            "linear"
        ],

        "svm__C": [
            0.1,
            1,
            10
        ],

        "svm__gamma": [
            "scale",
            0.001,
            0.01
        ]
    }


    print(
        "\nTUNING SVM MODEL"
    )

    print(
        "Searching SVM parameters..."
    )


    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_distributions,

        n_iter=9,

        scoring="f1",

        cv=cv,

        n_jobs=-1,

        verbose=1,

        random_state=RANDOM_STATE,

        return_train_score=True
    )


    search.fit(
        X_train,
        y_train
    )


    best_model = search.best_estimator_


    # =====================================================
    # BEST SVM
    # =====================================================

    print(
        "\nBEST SVM MODEL"
    )

    print(
        "Best Parameters:",
        search.best_params_
    )

    print(
        f"Best CV F1: {search.best_score_:.4f}"
    )


    # =====================================================
    # OVERFITTING CHECK
    # =====================================================

    train_pred = best_model.predict(
        X_train
    )


    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    train_precision = precision_score(
        y_train,
        train_pred,
        zero_division=0
    )

    train_recall = recall_score(
        y_train,
        train_pred,
        zero_division=0
    )

    train_f1 = f1_score(
        y_train,
        train_pred,
        zero_division=0
    )

    f1_gap = (
        train_f1 -
        search.best_score_
    )


    print(
        "\nOVERFITTING CHECK"
    )

    print(
        f"Training Accuracy : {train_accuracy:.4f}"
    )

    print(
        f"Training Precision: {train_precision:.4f}"
    )

    print(
        f"Training Recall   : {train_recall:.4f}"
    )

    print(
        f"Training F1       : {train_f1:.4f}"
    )

    print(
        f"CV F1             : {search.best_score_:.4f}"
    )

    print(
        f"F1 Gap            : {f1_gap:.4f}"
    )


    # =====================================================
    # OOF PROBABILITIES
    # =====================================================

    print(
        "\nGenerating out-of-fold probabilities..."
    )


    oof_probas = cross_val_predict(

        best_model,

        X_train,

        y_train,

        cv=cv,

        method="predict_proba",

        n_jobs=-1

    )[:, 1]


    # =====================================================
    # THRESHOLD OPTIMIZATION
    # =====================================================

    threshold_results = analyze_thresholds(

        y_train,

        oof_probas
    )


    decision_threshold = choose_threshold(
        threshold_results
    )


    threshold_results.to_csv(

        OUTPUT_DIR /
        "svm_threshold_comparison.csv",

        index=False
    )


    # =====================================================
    # AFTER TUNING
    # =====================================================

    print(
        "\nAFTER TUNING: SVM MODEL RESULTS"
    )


    test_probas = (
        best_model
        .predict_proba(X_test)[:, 1]
    )


    tuned_metrics, tuned_pred = calculate_metrics(

        y_test,

        test_probas,

        decision_threshold
    )


    tuning_comparison = pd.DataFrame([

        {
            "Stage": "Before Tuning",
            **baseline_metrics
        },

        {
            "Stage": "After Tuning",
            **tuned_metrics
        }

    ])


    print(
        f"\nTuned threshold = "
        f"{decision_threshold:.4f}"
    )


    print(
        "\nTUNING COMPARISON"
    )

    print(
        tuning_comparison.to_string(
            index=False
        )
    )


    tuning_comparison.to_csv(

        OUTPUT_DIR /
        "svm_tuning_comparison.csv",

        index=False
    )


    # =====================================================
    # FINAL METRICS
    # =====================================================

    y_pred = tuned_pred

    metrics = tuned_metrics


    final_metrics = {

        "Model":
            "SVM",

        "Accuracy":
            metrics["Accuracy"],

        "Precision":
            metrics["Precision"],

        "Recall":
            metrics["Recall"],

        "F1-Score":
            metrics["F1-Score"],

        "ROC-AUC":
            metrics["ROC-AUC"],

        "Default Threshold":
            0.50,

        "Decision Threshold":
            decision_threshold,

        "Best CV F1":
            search.best_score_,

        "Training F1":
            train_f1,

        "F1 Gap":
            f1_gap,

        "Best Params":
            str(search.best_params_)
    }


    pd.DataFrame(
        [final_metrics]
    ).to_csv(

        OUTPUT_DIR /
        "svm_metrics.csv",

        index=False
    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    print(
        "\nFINAL OVERALL RESULT"
    )

    print(
        pd.DataFrame(
            [tuned_metrics]
        ).to_string(index=False)
    )


    # =====================================================
    # SAVE MODEL
    # =====================================================

    joblib.dump(

        best_model,

        OUTPUT_DIR /
        "best_svm_model.joblib"
    )


    joblib.dump(

        decision_threshold,

        OUTPUT_DIR /
        "svm_decision_threshold.joblib"
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    print(
        "\nCLASSIFICATION REPORT"
    )

    print(
        classification_report(

            y_test,

            y_pred,

            target_names=[
                "No Disease",
                "Heart Disease"
            ],

            zero_division=0
        )
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    plt.figure(
        figsize=(6, 5)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Purples",

        cbar=False,

        xticklabels=[
            "No Heart Disease",
            "Heart Disease"
        ],

        yticklabels=[
            "No Heart Disease",
            "Heart Disease"
        ]
    )


    plt.title(
        "SVM Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR /
        "svm_confusion_matrix.png",

        dpi=300
    )


    plt.close()


    # =====================================================
    # ROC CURVE
    # =====================================================

    fpr, tpr, _ = roc_curve(

        y_test,

        test_probas
    )


    plt.figure(
        figsize=(6, 5)
    )


    plt.plot(

        fpr,

        tpr,

        lw=2,

        label=(
            f"SVM "
            f"(AUC = "
            f"{metrics['ROC-AUC']:.3f})"
        )
    )


    plt.plot(

        [0, 1],

        [0, 1],

        lw=1,

        linestyle="--"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate (Recall)"
    )

    plt.title(
        "SVM ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR /
        "svm_roc_curve.png",

        dpi=300
    )


    plt.close()


    # =====================================================
    # THRESHOLD ANALYSIS PLOT
    # =====================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(

        threshold_results["threshold"],

        threshold_results["accuracy"],

        label="Accuracy"
    )


    plt.plot(

        threshold_results["threshold"],

        threshold_results["precision"],

        label="Precision"
    )


    plt.plot(

        threshold_results["threshold"],

        threshold_results["recall"],

        label="Recall"
    )


    plt.plot(

        threshold_results["threshold"],

        threshold_results["f1"],

        label="F1-Score"
    )


    plt.axvline(

        decision_threshold,

        linestyle="--",

        label=(
            f"Chosen Threshold "
            f"({decision_threshold:.2f})"
        )
    )


    plt.ylim(
        0,
        1
    )


    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SVM Threshold Analysis"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR /
        "svm_threshold_metrics.png",

        dpi=300
    )


    plt.close()


    # =====================================================
    # COMPLETED
    # =====================================================

    print(
        "\nSVM TRAINING COMPLETED"
    )

    print(
        f"Artefacts saved in: "
        f"{OUTPUT_DIR}"
    )

    print(
        "Generated files:"
    )

    print(
        "- best_svm_model.joblib"
    )

    print(
        "- svm_decision_threshold.joblib"
    )

    print(
        "- svm_metrics.csv"
    )

    print(
        "- svm_tuning_comparison.csv"
    )

    print(
        "- svm_threshold_comparison.csv"
    )

    print(
        "- svm_threshold_metrics.png"
    )

    print(
        "- svm_confusion_matrix.png"
    )

    print(
        "- svm_roc_curve.png"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
