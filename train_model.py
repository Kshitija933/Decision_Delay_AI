"""
DecisionDelay AI - Model Training Pipeline
Trains, evaluates, and saves the delay prediction models.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH   = "data/raw/decisiondelay_dataset.csv"
MODEL_DIR   = "models"
REPORT_DIR  = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

FEATURE_COLS = [
    "task_difficulty", "time_to_reward_days", "past_failure_loops",
    "self_efficacy_score", "emotional_valence", "social_pressure",
    "task_clarity", "time_available_hrs", "distraction_level",
    "perfectionism_score", "reward_proximity", "failure_weight", "clarity_gap",
]
TARGET = "delay_label"


# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────
def load_and_preprocess(path):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)   # High=0, Low=1, Medium=2 (alphabetical)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_enc, le, scaler, df


# ─────────────────────────────────────────────
# 2. DEFINE MODELS
# ─────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                      random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                           max_depth=5, random_state=42),
        "XGBoost":             xgb.XGBClassifier(n_estimators=200, learning_rate=0.1,
                                                  max_depth=6, use_label_encoder=False,
                                                  eval_metric="mlogloss", random_state=42,
                                                  verbosity=0),
        "SVM":                 SVC(kernel="rbf", probability=True, C=10, random_state=42),
    }


# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ─────────────────────────────────────────────
def train_evaluate(X, y, le):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = get_models()
    results = {}
    trained_models = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, model in models.items():
        print(f"\n▶ Training: {name}")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted")
        print(f"  CV F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full train set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else None

        print(f"  Test Accuracy: {acc:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f if auc else 'N/A'}")

        results[name] = {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "auc_ovr": round(auc, 4) if auc else None,
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "classification_report": classification_report(y_test, y_pred,
                                      target_names=le.classes_, output_dict=True),
        }
        trained_models[name] = model

    return trained_models, results, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 4. ENSEMBLE MODEL
# ─────────────────────────────────────────────
def build_ensemble(trained_models, X_train, y_train):
    estimators = [
        ("rf",  trained_models["Random Forest"]),
        ("xgb", trained_models["XGBoost"]),
        ("gb",  trained_models["Gradient Boosting"]),
    ]
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train, y_train)
    return ensemble


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, out_path):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="#4f46e5")
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Feature Importances – DecisionDelay AI")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved feature importance plot → {out_path}")


# ─────────────────────────────────────────────
# 6. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, le, name, out_path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(trained_models, ensemble, scaler, le, results):
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    best_model = trained_models[best_name]
    print(f"\n🏆 Best Model: {best_name}  (F1={results[best_name]['f1_weighted']})")

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(ensemble,   f"{MODEL_DIR}/ensemble_model.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le,         f"{MODEL_DIR}/label_encoder.pkl")

    with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    with open(f"{REPORT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    meta = {
        "best_model": best_name,
        "best_f1": results[best_name]["f1_weighted"],
        "trained_at": datetime.now().isoformat(),
        "n_features": len(FEATURE_COLS),
        "classes": list(le.classes_),
    }
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Models saved to /{MODEL_DIR}/")
    print(f"✅ Reports saved to /{REPORT_DIR}/")
    return best_model, best_name


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Load
    X, y, le, scaler, df = load_and_preprocess(DATA_PATH)

    # Step 2: Train
    trained_models, results, X_train, X_test, y_train, y_test = train_evaluate(X, y, le)

    # Step 3: Ensemble
    print("\n▶ Building Ensemble Model...")
    ensemble = build_ensemble(trained_models, X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    ens_f1   = f1_score(y_test, ens_pred, average="weighted")
    print(f"  Ensemble F1: {ens_f1:.4f}")
    results["Ensemble"] = {"f1_weighted": round(ens_f1, 4), "accuracy": round(accuracy_score(y_test, ens_pred), 4)}

    # Step 4: Plots
    plot_feature_importance(
        trained_models["XGBoost"], FEATURE_COLS,
        f"{REPORT_DIR}/feature_importance.png"
    )
    for name, model in trained_models.items():
        plot_confusion_matrix(model, X_test, y_test, le, name,
                              f"{REPORT_DIR}/cm_{name.replace(' ', '_')}.png")

    # Step 5: Save
    save_artifacts(trained_models, ensemble, scaler, le, results)

    # Summary Table
    print("\n" + "="*60)
    print("  FINAL COMPARISON TABLE")
    print("="*60)
    summary = pd.DataFrame({k: {"Accuracy": v["accuracy"], "F1 (weighted)": v["f1_weighted"]}
                             for k, v in results.items()}).T
    print(summary.sort_values("F1 (weighted)", ascending=False).to_string())