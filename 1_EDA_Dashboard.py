"""
DecisionDelay AI — Page: Exploratory Data Analysis
Place in: pages/1_📊_EDA_Dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="EDA Dashboard | DecisionDelay AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Syne:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background: #0f0e17; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Exploratory Data Analysis")

DATA_PATH = Path("data/raw/decisiondelay_dataset.csv")

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    # Generate synthetic preview if dataset not generated yet
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "delay_label":          np.random.choice(["Low","Medium","High"], n, p=[0.3,0.4,0.3]),
        "domain":               np.random.choice(["Fitness","Studying","Career Choice","Finance"], n),
        "task_difficulty":      np.random.uniform(1,10,n).round(2),
        "time_to_reward_days":  np.random.uniform(1,365,n).round(1),
        "past_failure_loops":   np.random.randint(0,12,n),
        "self_efficacy_score":  np.random.uniform(1,10,n).round(2),
        "emotional_valence":    np.random.uniform(-5,5,n).round(2),
        "distraction_level":    np.random.uniform(1,10,n).round(2),
        "perfectionism_score":  np.random.uniform(1,10,n).round(2),
        "delay_score":          np.random.uniform(0,100,n).round(1),
        "primary_delay_cause":  np.random.choice(
            ["Fear of Failure","Overwhelm / Complexity","Perfectionism","Low Self-Efficacy"], n),
        "acted":                np.random.randint(0,2,n),
    })
    return df

df = load_data()

# ── Summary Stats ─────────────────────────────────────────────
st.markdown("### Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Samples",    f"{len(df):,}")
c2.metric("Features",         f"{len(df.columns)}")
c3.metric("Avg Delay Score",  f"{df['delay_score'].mean():.1f}" if 'delay_score' in df.columns else "—")
c4.metric("Action Rate",      f"{df['acted'].mean()*100:.1f}%" if 'acted' in df.columns else "—")

st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# ── Label Distribution ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Delay Label Distribution")
    vc = df["delay_label"].value_counts()
    fig = go.Figure(go.Pie(
        labels=vc.index, values=vc.values,
        marker_colors=["#ef4444","#f59e0b","#10b981"],
        hole=0.4,
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e7ff", height=300, margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Delay Causes Distribution")
    if "primary_delay_cause" in df.columns:
        vc2 = df["primary_delay_cause"].value_counts()
        fig2 = px.bar(x=vc2.values, y=vc2.index, orientation="h",
                      color=vc2.values, color_continuous_scale="Purples")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e7ff", height=300, margin=dict(t=10,b=10),
                           coloraxis_showscale=False, yaxis_title="", xaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Feature Distributions ──────────────────────────────────────
st.markdown("#### Feature Distributions by Delay Label")
num_cols = ["task_difficulty", "self_efficacy_score", "emotional_valence",
            "perfectionism_score", "distraction_level", "time_to_reward_days"]
num_cols = [c for c in num_cols if c in df.columns]

sel = st.selectbox("Select Feature", num_cols)
fig3 = px.histogram(df, x=sel, color="delay_label",
                    color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
                    barmode="overlay", nbins=30, opacity=0.75)
fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font_color="#e0e7ff", height=350)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Correlation Heatmap ────────────────────────────────────────
st.markdown("#### Feature Correlation Matrix")
corr_cols = [c for c in num_cols + ["past_failure_loops","delay_score"] if c in df.columns]
corr = df[corr_cols].corr()
fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                 aspect="auto", zmin=-1, zmax=1)
fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e7ff", height=400)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Domain breakdown ──────────────────────────────────────────
st.markdown("#### Domain vs Delay Risk")
if "domain" in df.columns:
    grp = df.groupby(["domain","delay_label"]).size().reset_index(name="count")
    fig5 = px.bar(grp, x="domain", y="count", color="delay_label",
                  color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
                  barmode="group")
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e0e7ff", height=350)
    st.plotly_chart(fig5, use_container_width=True)