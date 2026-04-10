"""
DecisionDelay AI — Page: Model Performance
Place in: pages/2_🤖_Model_Performance.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Model Performance | DecisionDelay AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Syne:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Model Performance Dashboard")

# ── Load Results ──────────────────────────────────────────────
REPORT_PATH = Path("reports/results.json")

@st.cache_data
def load_results():
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            return json.load(f)
    # Simulated results for demo
    return {
        "Logistic Regression": {"accuracy": 0.7412, "f1_weighted": 0.7389, "auc_ovr": 0.8821, "cv_f1_mean": 0.7310, "cv_f1_std": 0.0183},
        "Random Forest":       {"accuracy": 0.8634, "f1_weighted": 0.8621, "auc_ovr": 0.9412, "cv_f1_mean": 0.8571, "cv_f1_std": 0.0121},
        "Gradient Boosting":   {"accuracy": 0.8721, "f1_weighted": 0.8699, "auc_ovr": 0.9488, "cv_f1_mean": 0.8641, "cv_f1_std": 0.0109},
        "XGBoost":             {"accuracy": 0.8855, "f1_weighted": 0.8831, "auc_ovr": 0.9563, "cv_f1_mean": 0.8792, "cv_f1_std": 0.0098},
        "SVM":                 {"accuracy": 0.8123, "f1_weighted": 0.8101, "auc_ovr": 0.9021, "cv_f1_mean": 0.8059, "cv_f1_std": 0.0145},
        "Ensemble":            {"accuracy": 0.8931, "f1_weighted": 0.8912, "auc_ovr": None,   "cv_f1_mean": None,   "cv_f1_std": None},
    }

results = load_results()
df_res = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

# ── Best Model Banner ─────────────────────────────────────────
best = df_res.loc[df_res["f1_weighted"].idxmax(), "Model"]
best_f1 = df_res["f1_weighted"].max()

st.markdown(f"""
<div style="background:linear-gradient(135deg,#064e3b,#065f46);border:1px solid #10b981;
            border-radius:12px;padding:1.2rem 2rem;margin-bottom:1.5rem">
    <div style="color:#6ee7b7;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px">🏆 Best Model</div>
    <div style="color:#fff;font-size:1.8rem;font-weight:700">{best}</div>
    <div style="color:#d1fae5">F1-Weighted Score: <b>{best_f1:.4f}</b></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Comparison Table ──────────────────────────────────────────
st.markdown("### Model Comparison Table")
st.dataframe(
    df_res.set_index("Model")[["accuracy","f1_weighted","auc_ovr","cv_f1_mean","cv_f1_std"]]
          .rename(columns={"accuracy":"Accuracy","f1_weighted":"F1 (Weighted)",
                            "auc_ovr":"AUC-OVR","cv_f1_mean":"CV F1 Mean","cv_f1_std":"CV F1 Std"})
          .style.background_gradient(cmap="Purples", subset=["Accuracy","F1 (Weighted)"])
          .format("{:.4f}", na_rep="—"),
    use_container_width=True,
)

st.markdown("---")

# ── Bar Charts ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### F1-Weighted Score")
    df_sorted = df_res.sort_values("f1_weighted", ascending=True)
    fig1 = go.Figure(go.Bar(
        x=df_sorted["f1_weighted"], y=df_sorted["Model"],
        orientation="h",
        marker=dict(
            color=df_sorted["f1_weighted"],
            colorscale="Blues",
            showscale=False,
        ),
        text=df_sorted["f1_weighted"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e0e7ff", height=350, xaxis_range=[0.6, 1.0],
                       margin=dict(t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### Accuracy vs AUC-OVR")
    df_scatter = df_res.dropna(subset=["auc_ovr"])
    fig2 = px.scatter(df_scatter, x="accuracy", y="auc_ovr", text="Model",
                      color="f1_weighted", color_continuous_scale="Purples",
                      size_max=20)
    fig2.update_traces(textposition="top center", marker_size=14)
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e0e7ff", height=350, margin=dict(t=10,b=10),
                       xaxis_title="Accuracy", yaxis_title="AUC-OVR",
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Feature Importance (image if exists) ──────────────────────
st.markdown("### Feature Importance (XGBoost)")

FI_PATH = Path("reports/feature_importance.png")
if FI_PATH.exists():
    st.image(str(FI_PATH), use_column_width=True)
else:
    # Simulated
    features = ["task_difficulty","distraction_level","perfectionism_score",
                "past_failure_loops","self_efficacy_score","clarity_gap",
                "time_to_reward_days","failure_weight","emotional_valence",
                "social_pressure","time_available_hrs","reward_proximity","task_clarity"]
    importances = np.array([0.19,0.15,0.12,0.11,0.10,0.09,0.07,0.06,0.05,0.04,0.03,0.02,0.01] + [0]*0)
    df_fi = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance")
    fig3 = px.bar(df_fi, x="Importance", y="Feature", orientation="h", color="Importance",
                  color_continuous_scale="Purples")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e0e7ff", height=420, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

# ── Model Architecture Explainer ──────────────────────────────
st.markdown("---")
st.markdown("### 🏗️ Model Architecture Summary")

arch_data = {
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "SVM", "Ensemble"],
    "Type": ["Linear", "Bagging", "Boosting", "Boosting", "Kernel", "Voting"],
    "Estimators": ["—", "200", "200", "200", "—", "3 models"],
    "Key Strength": [
        "Interpretable baseline",
        "Handles non-linearity well",
        "Sequential error correction",
        "Speed + regularization",
        "Good in high-dim space",
        "Reduces variance & bias",
    ],
}
st.table(pd.DataFrame(arch_data))