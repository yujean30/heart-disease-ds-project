from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def load_data():
    files = {
        "X_train": ROOT / "Preprocessing" / "X_train_preprocessed.csv",
        "X_test": ROOT / "Preprocessing" / "X_test_preprocessed.csv",
        "y_train": ROOT / "Preprocessing" / "y_train.csv",
        "y_test": ROOT / "Preprocessing" / "y_test.csv"
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}. Please run preprocess.py first.")

    X_train = pd.read_csv(files["X_train"])
    X_test = pd.read_csv(files["X_test"])
    y_train = pd.read_csv(files["y_train"]).squeeze("columns").to_numpy()
    y_test = pd.read_csv(files["y_test"]).squeeze("columns").to_numpy()

    return X_train, X_test, y_train, y_test


def check_data(X_train, X_test, y_train, y_test):
    print("\nDATA CHECK")
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape : {X_test.shape}")
    print("Training class balance:", pd.Series(y_train).value_counts().to_dict())
    print("Testing class balance :", pd.Series(y_test).value_counts().to_dict())
    print("Train/Test columns match:", list(X_train.columns) == list(X_test.columns))
    print("Training missing values:", X_train.isnull().sum().sum())
    print("Testing missing values :", X_test.isnull().sum().sum())

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing columns do not match.")

    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        raise ValueError("Missing values remain after preprocessing.")


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


class SVMXGBHybrid(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        svm_C=1.0,
        svm_gamma="scale",
        svm_kernel="rbf",
        xgb_n_estimators=100,
        xgb_max_depth=3,
        xgb_learning_rate=0.05,
        xgb_min_child_weight=3,
        xgb_subsample=0.8,
        xgb_colsample_bytree=0.8,
        xgb_gamma=0.0,
        svm_weight=1.0,
        xgb_weight=1.0
    ):
        self.svm_C = svm_C
        self.svm_gamma = svm_gamma
        self.svm_kernel = svm_kernel
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_min_child_weight = xgb_min_child_weight
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_gamma = xgb_gamma
        self.svm_weight = svm_weight
        self.xgb_weight = xgb_weight

    def fit(self, X, y):
        X = np.asarray(X)

        self.svm_ = SVC(
            kernel=self.svm_kernel,
            C=self.svm_C,
            gamma=self.svm_gamma,
            probability=True,
            cache_size=1000,
            random_state=RANDOM_STATE
        )

        self.xgb_ = XGBClassifier(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            min_child_weight=self.xgb_min_child_weight,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            gamma=self.xgb_gamma,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1
        )

        self.svm_.fit(X, y)
        self.xgb_.fit(X, y)
        self.classes_ = np.array([0, 1])

        return self

    def predict_proba(self, X):
        X = np.asarray(X)

        svm_prob = self.svm_.predict_proba(X)[:, 1]
        xgb_prob = self.xgb_.predict_proba(X)[:, 1]

        total_weight = self.svm_weight + self.xgb_weight
        hybrid_prob = (
            self.svm_weight * svm_prob +
            self.xgb_weight * xgb_prob
        ) / total_weight

        return np.column_stack([1 - hybrid_prob, hybrid_prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.50).astype(int)


def analyze_thresholds(y_true, probabilities):
    thresholds = np.linspace(0.20, 0.80, 61)
    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1": f1_score(y_true, predictions, zero_division=0),
            "positive_rate": predictions.mean()
        })

    return pd.DataFrame(results)


def choose_threshold(results_df):
    reasonable = results_df[
        (results_df["positive_rate"] >= 0.10) &
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
    print(f"Chosen threshold : {best['threshold']:.4f}")
    print(f"OOF Accuracy     : {best['accuracy']:.4f}")
    print(f"OOF Precision    : {best['precision']:.4f}")
    print(f"OOF Recall       : {best['recall']:.4f}")
    print(f"OOF F1           : {best['f1']:.4f}")
    print(f"OOF Positive Rate: {best['positive_rate']:.4f}")

    return float(best["threshold"])


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1-Score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities)
    }

    return metrics, predictions


def main():

    X_train, X_test, y_train, y_test = load_data()
    check_data(X_train, X_test, y_train, y_test)

    print("\nSMOTENC CONFIGURATION")
    print("SMOTENC is applied INSIDE CV.")
    print("Test data will NOT be oversampled.")

    pipeline = Pipeline([
        ("smote", create_smote(X_train)),
        ("hybrid", SVMXGBHybrid())
    ])

    print("\nHYBRID MODEL")
    print("Main algorithm       : SVM")
    print("Supporting algorithm : XGBoost")
    print("Combination method   : Weighted Soft Voting")

    print("\nBEFORE TUNING: HYBRID MODEL RESULTS")
    baseline_model = Pipeline([
        ("smote", create_smote(X_train)),
        ("hybrid", SVMXGBHybrid())
    ])
    baseline_model.fit(X_train, y_train)
    baseline_probas = baseline_model.predict_proba(X_test)[:, 1]
    baseline_metrics, _ = calculate_metrics(y_test, baseline_probas, 0.50)
    print(pd.DataFrame([baseline_metrics]).to_string(index=False))

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    param_distributions = {
        "hybrid__svm_kernel": ["rbf", "linear"],
        "hybrid__svm_C": [0.1, 1, 10],
        "hybrid__svm_gamma": ["scale", 0.001, 0.01],
        "hybrid__xgb_n_estimators": [100, 200],
        "hybrid__xgb_max_depth": [2, 3, 4],
        "hybrid__xgb_learning_rate": [0.02, 0.05, 0.1],
        "hybrid__xgb_min_child_weight": [3, 5],
        "hybrid__xgb_subsample": [0.8, 1.0],
        "hybrid__xgb_colsample_bytree": [0.8, 1.0],
        "hybrid__xgb_gamma": [0.0, 0.3],
        "hybrid__svm_weight": [1, 2, 3],
        "hybrid__xgb_weight": [1, 2, 3]
    }

    print("\nTUNING HYBRID MODEL")
    print("Searching SVM + XGBoost parameters...")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=True
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\nBEST HYBRID MODEL")
    print("Best Parameters:", search.best_params_)
    print(f"Best CV F1: {search.best_score_:.4f}")

    train_pred = best_model.predict(X_train)

    train_accuracy = accuracy_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred, zero_division=0)
    train_recall = recall_score(y_train, train_pred, zero_division=0)
    train_f1 = f1_score(y_train, train_pred, zero_division=0)
    f1_gap = train_f1 - search.best_score_

    print("\nOVERFITTING CHECK")
    print(f"Training Accuracy : {train_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Training Recall   : {train_recall:.4f}")
    print(f"Training F1       : {train_f1:.4f}")
    print(f"CV F1             : {search.best_score_:.4f}")
    print(f"F1 Gap            : {f1_gap:.4f}")

    print("\nGenerating out-of-fold probabilities...")

    oof_probas = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    threshold_results = analyze_thresholds(y_train, oof_probas)
    decision_threshold = choose_threshold(threshold_results)

    threshold_results.to_csv(
        OUTPUT_DIR / "svm_xgb_threshold_comparison.csv",
        index=False
    )

    print("\nAFTER TUNING: HYBRID MODEL RESULTS")

    test_probas = best_model.predict_proba(X_test)[:, 1]

    tuned_metrics, tuned_pred = calculate_metrics(
        y_test, test_probas, decision_threshold
    )

    tuning_comparison = pd.DataFrame([
        {"Stage": "Before Tuning", **baseline_metrics},
        {"Stage": "After Tuning", **tuned_metrics}
    ])
    print(f"\nTuned threshold = {decision_threshold:.4f}")
    print("\nTUNING COMPARISON")
    print(tuning_comparison.to_string(index=False))
    tuning_comparison.to_csv(
        OUTPUT_DIR / "svm_tuning_comparison.csv",
        index=False
    )

    y_pred = tuned_pred
    metrics = tuned_metrics

    final_metrics = {
        "Model": "SVM + XGBoost Hybrid",
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-Score": metrics["F1-Score"],
        "ROC-AUC": metrics["ROC-AUC"],
        "Default Threshold": 0.50,
        "Decision Threshold": decision_threshold,
        "Best CV F1": search.best_score_,
        "Training F1": train_f1,
        "F1 Gap": f1_gap,
        "Best Params": str(search.best_params_)
    }

    pd.DataFrame([final_metrics]).to_csv(
        OUTPUT_DIR / "svm_xgb_metrics.csv",
        index=False
    )

    print("\nFINAL OVERALL RESULT")
    print(pd.DataFrame([tuned_metrics]).to_string(index=False))

    joblib.dump(
        best_model,
        OUTPUT_DIR / "best_svm_xgb_hybrid_model.joblib"
    )

    joblib.dump(
        decision_threshold,
        OUTPUT_DIR / "svm_xgb_decision_threshold.joblib"
    )

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Disease", "Heart Disease"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=False,
        xticklabels=["No Heart Disease", "Heart Disease"],
        yticklabels=["No Heart Disease", "Heart Disease"]
    )
    plt.title("SVM + XGBoost Hybrid Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "svm_xgb_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, test_probas)

    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"SVM + XGBoost (AUC = {metrics['ROC-AUC']:.3f})"
    )
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("SVM + XGBoost Hybrid ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "svm_xgb_roc_curve.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 5))
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
        label=f"Chosen Threshold ({decision_threshold:.2f})"
    )
    plt.ylim(0, 1)
    plt.xlabel("Probability Threshold")
    plt.ylabel("Score")
    plt.title("SVM + XGBoost Threshold Analysis")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "svm_xgb_threshold_metrics.png",
        dpi=300
    )
    plt.close()

    print("\nSVM + XGBOOST HYBRID TRAINING COMPLETED")
    print(f"Artefacts saved in: {OUTPUT_DIR}")
    print("Generated files:")
    print("- best_svm_xgb_hybrid_model.joblib")
    print("- svm_xgb_decision_threshold.joblib")
    print("- svm_xgb_metrics.csv")
    print("- svm_tuning_comparison.csv")
    print("- svm_xgb_threshold_comparison.csv")
    print("- svm_xgb_threshold_metrics.png")
    print("- svm_xgb_confusion_matrix.png")
    print("- svm_xgb_roc_curve.png")


if __name__ == "__main__":
    main()
