"""
DecisionDelay AI — Streamlit Application
Main entry point. Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DecisionDelay AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(99,102,241,0.3);
}
.hero-banner h1 { color: #fff; font-size: 2.8rem; margin: 0; letter-spacing: -1px; }
.hero-banner p  { color: #c7d2fe; font-size: 1.1rem; margin: 0.5rem 0 0; }

.metric-card {
    background: linear-gradient(135deg, #1e1b4b, #2d2a5e);
    border: 1px solid #4f46e5;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.3rem 0;
}
.metric-card .label { color: #a5b4fc; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }

.cause-card {
    border-left: 4px solid #4f46e5;
    background: #1e1b4b20;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.nudge-box {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.nudge-box h4 { color: #6ee7b7; margin: 0 0 0.8rem; }
.nudge-box p  { color: #d1fae5; margin: 0; }

.technique-item {
    background: #ffffff08;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    color: #c7d2fe;
    font-size: 0.9rem;
}

.delay-HIGH   { color: #f87171; font-weight: 700; }
.delay-MEDIUM { color: #fbbf24; font-weight: 700; }
.delay-LOW    { color: #34d399; font-weight: 700; }

.stSlider > div > div > div > div { background: #4f46e5 !important; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    color: #a5b4fc;
    border-bottom: 1px solid #312e81;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# NUDGE LIBRARY (inline so app runs without models)
# ─────────────────────────────────────────────
NUDGE_LIBRARY = {
    "Fear of Failure": {
        "emoji": "😨", "title": "Fear of Failure",
        "primary_nudge": "Reframe: every attempt is data. Start a 2-minute version RIGHT NOW.",
        "techniques": [
            "Implementation Intention: 'When X happens, I will do Y for 2 minutes.'",
            "Failure Pre-mortem: Write the worst case. Is it survivable?",
            "Past-win recall: Name one time you succeeded despite fear.",
        ],
        "quote": "You don't have to be great to start, but you have to start to be great. — Zig Ziglar",
    },
    "Overwhelm / Complexity": {
        "emoji": "🌊", "title": "Overwhelm / Complexity",
        "primary_nudge": "Break it into 3 micro-tasks. Focus ONLY on task #1.",
        "techniques": [
            "List every physical sub-action on paper.",
            "Pomodoro: commit to 25 minutes, then stop.",
            "Check off sub-tasks visibly for dopamine hits.",
        ],
        "quote": "Divide each difficulty into as many parts as possible. — Descartes",
    },
    "Lack of Immediate Reward": {
        "emoji": "⏳", "title": "Lack of Immediate Reward",
        "primary_nudge": "Attach an INSTANT reward after completion. Temptation-bundle it.",
        "techniques": [
            "Temptation bundling: Only do X (pleasure) while doing Y (work).",
            "Commit publicly to a deadline.",
            "Visualize the version of you who completed this.",
        ],
        "quote": "Act in the present; invest in the future. — Seneca",
    },
    "Perfectionism": {
        "emoji": "🎯", "title": "Perfectionism",
        "primary_nudge": "Set a 'good enough' bar. Ship the draft. Perfect kills done.",
        "techniques": [
            "Version 0.1 mindset: ugliest working version first.",
            "Strict time limit: 30 minutes then submit.",
            "Your work ≠ your worth. Decouple them.",
        ],
        "quote": "Done is better than perfect. — Sheryl Sandberg",
    },
    "Low Self-Efficacy": {
        "emoji": "💪", "title": "Low Self-Efficacy",
        "primary_nudge": "Recall a past win. Use it as proof you handle hard things.",
        "techniques": [
            "Start with an easier version to build confidence.",
            "Find someone similar who succeeded.",
            "Write 3 specific things you've accomplished this year.",
        ],
        "quote": "Whether you think you can or can't — you're right. — Henry Ford",
    },
    "External Distractions": {
        "emoji": "📵", "title": "External Distractions",
        "primary_nudge": "Environment design: phone away, noise-cancelling on, blockers active.",
        "techniques": [
            "Make distractions 20 seconds harder to access.",
            "Designate a specific deep-work spot.",
            "Airplane mode for 25-minute sprints.",
        ],
        "quote": "The ability to concentrate single-mindedly is the most valuable skill. — Brian Tracy",
    },
    "Ambiguity / Unclear Next Step": {
        "emoji": "🧭", "title": "Ambiguity / Unclear Next Step",
        "primary_nudge": "Write the VERY NEXT physical action in one sentence, right now.",
        "techniques": [
            "GTD: 'What is the next visible, physical action?'",
            "If-then: 'At 9am I will open file X and write Y.'",
            "5-min planning session before any work block.",
        ],
        "quote": "A good plan today beats a perfect plan tomorrow. — Patton",
    },
    "Emotional Avoidance": {
        "emoji": "🧠", "title": "Emotional Avoidance",
        "primary_nudge": "Name the emotion aloud. Do 5 minutes of exposure, then re-evaluate.",
        "techniques": [
            "Label it: 'I notice I feel ___ about this task.'",
            "RAIN: Recognize, Allow, Investigate, Nurture.",
            "Opposite action: walk toward the task gently.",
        ],
        "quote": "Between stimulus and response there is a space. — Viktor Frankl",
    },
}

DELAY_CAUSES = list(NUDGE_LIBRARY.keys())


# ─────────────────────────────────────────────
# RULE-BASED PREDICTOR (fallback when model unavailable)
# ─────────────────────────────────────────────
def rule_based_predict(d: dict) -> dict:
    score = (
        d["task_difficulty"]       * 3.5
        + d["time_to_reward_days"] * 0.08
        + d["past_failure_loops"]  * 2.5
        - d["self_efficacy_score"] * 3.0
        + max(0, -d["emotional_valence"]) * 2.0
        - d["social_pressure"]     * 1.5
        + (10 - d["task_clarity"]) * 2.0
        + d["distraction_level"]   * 2.0
        + d["perfectionism_score"] * 1.5
        - d["time_available_hrs"]  * 1.0
    )
    score = float(np.clip(score, 0, 100))

    if score < 30:
        label = "Low"
    elif score < 60:
        label = "Medium"
    else:
        label = "High"

    cause_scores = {
        "Fear of Failure":               d["past_failure_loops"] * 2 + max(0, -d["emotional_valence"]),
        "Overwhelm / Complexity":        d["task_difficulty"] + (10 - d["task_clarity"]),
        "Lack of Immediate Reward":      d["time_to_reward_days"] * 0.2,
        "Perfectionism":                 d["perfectionism_score"],
        "Low Self-Efficacy":             10 - d["self_efficacy_score"],
        "External Distractions":         d["distraction_level"],
        "Ambiguity / Unclear Next Step": (10 - d["task_clarity"]) * 1.5,
        "Emotional Avoidance":           max(0, -d["emotional_valence"]) * 2,
    }
    cause = max(cause_scores, key=cause_scores.get)

    # Probability-like scores
    high_p   = score / 100
    low_p    = max(0, (100 - score - 20) / 100)
    medium_p = max(0, 1 - high_p - low_p)

    return {
        "delay_label":   label,
        "delay_score":   round(score, 1),
        "probabilities": {"High": round(high_p, 2), "Medium": round(medium_p, 2), "Low": round(low_p, 2)},
        "primary_cause": cause,
        "nudge":         NUDGE_LIBRARY[cause],
        "confidence":    round(max(high_p, medium_p, low_p), 2),
        "model_used":    "Rule-Based Engine",
    }


# ─────────────────────────────────────────────
# TRY ML MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    try:
        sys.path.insert(0, ".")
        from inference.predict import DecisionDelayPredictor
        return DecisionDelayPredictor(), "ML Ensemble"
    except Exception:
        return None, "Rule-Based Engine"


# ─────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────
def delay_gauge(score: float, label: str):
    color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}.get(label, "#6366f1")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Delay Risk Score<br><b style='color:{color}'>{label}</b>",
               "font": {"color": "#e0e7ff", "size": 16}},
        number={"font": {"color": color, "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6366f1"},
            "bar":  {"color": color},
            "bgcolor": "#1e1b4b",
            "bordercolor": "#4f46e5",
            "steps": [
                {"range": [0, 30],  "color": "rgba(6, 78, 59, 0.19)"},
                {"range": [30, 60], "color": "rgba(120, 53, 15, 0.19)"},
                {"range": [60, 100],"color": "rgba(127, 29, 29, 0.19)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e7ff",
        height=280,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ─────────────────────────────────────────────
# PROBABILITY BAR CHART
# ─────────────────────────────────────────────
def prob_chart(probs: dict):
    colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
    fig = go.Figure()
    for label, p in probs.items():
        fig.add_trace(go.Bar(
            x=[label], y=[p * 100],
            marker_color=colors.get(label, "#6366f1"),
            text=[f"{p*100:.1f}%"], textposition="outside",
            textfont={"color": "#e0e7ff"},
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e7ff",
        showlegend=False,
        yaxis={"range": [0, 110], "gridcolor": "rgba(49, 46, 129, 0.31)", "title": "Probability (%)"},
        xaxis={"title": "Delay Level"},
        height=250,
        margin=dict(t=10, b=10, l=10, r=10),
        title={"text": "Class Probability Distribution", "font": {"size": 14, "color": "#a5b4fc"}},
    )
    return fig


# ─────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────
def radar_chart(d: dict):
    categories = ["Task Difficulty", "Distraction", "Perfectionism",
                  "Failure History", "Low Clarity", "Time Scarcity"]
    values = [
        d["task_difficulty"] / 10,
        d["distraction_level"] / 10,
        d["perfectionism_score"] / 10,
        min(d["past_failure_loops"] / 12, 1),
        (10 - d["task_clarity"]) / 10,
        1 - d["time_available_hrs"] / 6,
    ]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line_color="#818cf8",
        fillcolor="rgba(99,102,241,0.25)",
    ))
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "range": [0, 1], "color": "#6366f1"},
            "angularaxis": {"color": "#a5b4fc"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e7ff",
        height=300,
        margin=dict(t=30, b=30, l=30, r=30),
        title={"text": "Delay Risk Profile", "font": {"color": "#a5b4fc", "size": 14}},
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────
def sidebar_inputs():
    st.sidebar.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem'>🧠</div>
        <div style='font-family:Syne,sans-serif; color:#a5b4fc; font-size:1.1rem; font-weight:700'>
            DecisionDelay AI
        </div>
        <div style='color:#6366f180; font-size:0.75rem'>Behavioral Nudge Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    domain = st.sidebar.selectbox("📂 Domain",
        ["Fitness", "Studying", "Career Choice", "Finance", "Health Checkup", "Creative Project"])

    st.sidebar.markdown("### 🔢 Task Parameters")
    task_difficulty    = st.sidebar.slider("Task Difficulty (1=Easy, 10=Hard)", 1.0, 10.0, 6.0, 0.1)
    time_to_reward     = st.sidebar.slider("Days Until Meaningful Reward", 1, 365, 60)
    past_failures      = st.sidebar.slider("Past Failure Loops (attempts quit)", 0, 12, 3)

    st.sidebar.markdown("### 🧬 Personal State")
    self_efficacy      = st.sidebar.slider("Self-Efficacy (belief in ability)", 1.0, 10.0, 5.0, 0.1)
    emotional_valence  = st.sidebar.slider("Emotional Valence (−5=Dread → +5=Excited)", -5.0, 5.0, 0.0, 0.1)
    perfectionism      = st.sidebar.slider("Perfectionism Score", 1.0, 10.0, 6.0, 0.1)

    st.sidebar.markdown("### 🌍 Environment")
    social_pressure    = st.sidebar.slider("Social / External Accountability", 0.0, 10.0, 3.0, 0.1)
    task_clarity       = st.sidebar.slider("Task Clarity (next step clarity)", 1.0, 10.0, 5.0, 0.1)
    distraction        = st.sidebar.slider("Distraction Level", 1.0, 10.0, 5.0, 0.1)
    time_available     = st.sidebar.slider("Free Time Available Today (hrs)", 0.25, 8.0, 2.0, 0.25)

    st.sidebar.markdown("---")
    analyze = st.sidebar.button("🔍 Analyze Delay Risk", use_container_width=True, type="primary")

    return analyze, {
        "domain": domain,
        "task_difficulty":     task_difficulty,
        "time_to_reward_days": float(time_to_reward),
        "past_failure_loops":  past_failures,
        "self_efficacy_score": self_efficacy,
        "emotional_valence":   emotional_valence,
        "social_pressure":     social_pressure,
        "task_clarity":        task_clarity,
        "time_available_hrs":  time_available,
        "distraction_level":   distraction,
        "perfectionism_score": perfectionism,
    }


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <h1>🧠 DecisionDelay AI</h1>
        <p>Why do people know what to do — but still don't act?<br>
        <b>AI-powered behavioral analysis. No sensors. No surveillance.</b></p>
    </div>
    """, unsafe_allow_html=True)

    analyze, user_input = sidebar_inputs()
    ml_model, model_name = load_ml_model()

    # ── Default landing state ──────────────────
    if not analyze:
        col1, col2, col3 = st.columns(3)
        for col, emoji, title, desc in [
            (col1, "🎯", "Behavioral Analysis", "Identifies the real root cause of your delay"),
            (col2, "💡", "Personalized Nudges", "Evidence-based strategies tailored to your profile"),
            (col3, "📊", "Visual Risk Profile", "Radar, gauge & probability charts at a glance"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{emoji}</div>
                <div class="label">{title}</div>
                <div style="color:#c7d2fe; font-size:0.88rem; margin-top:0.3rem">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.info("👈  Adjust your parameters in the sidebar and click **Analyze Delay Risk**")
        return

    # ── Run Prediction ─────────────────────────
    if ml_model:
        result = ml_model.predict(user_input)
    else:
        result = rule_based_predict(user_input)

    label  = result["delay_label"]
    score  = result["delay_score"]
    cause  = result["primary_cause"]
    nudge  = result["nudge"]
    probs  = result["probabilities"]

    delay_color = {"High": "#ef4444", "Medium": "#fbbf24", "Low": "#34d399"}.get(label, "#6366f1")

    # ── TOP METRICS ───────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val, sub in [
        (m1, "Delay Risk", label, "Classification"),
        (m2, "Risk Score", f"{score}/100", "0=None, 100=Max"),
        (m3, "Confidence", f"{result['confidence']*100:.0f}%", "Model certainty"),
        (m4, "Model", result.get("model_used", model_name)[:12], "Engine used"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{lbl}</div>
            <div class="value" style="color:{delay_color if lbl=='Delay Risk' else '#fff'}">{val}</div>
            <div style="color:#6366f180; font-size:0.72rem">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── CHARTS ROW ────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.plotly_chart(delay_gauge(score, label), use_container_width=True)
    with c2:
        st.plotly_chart(prob_chart(probs), use_container_width=True)
    with c3:
        st.plotly_chart(radar_chart(user_input), use_container_width=True)

    st.markdown("---")

    # ── CAUSE & NUDGE ─────────────────────────
    left, right = st.columns([1, 1.2])

    with left:
        st.markdown('<div class="section-header">🔍 Root Cause Analysis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cause-card">
            <div style="font-size:1.8rem">{nudge.get('emoji','🔎')}</div>
            <div style="color:#a5b4fc; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px">Primary Cause</div>
            <div style="color:#e0e7ff; font-size:1.3rem; font-weight:700; margin:0.2rem 0">{cause}</div>
        </div>""", unsafe_allow_html=True)

        # Cause breakdown bars
        causes_raw = {
            "Fear of Failure": user_input["past_failure_loops"] * 2 + max(0, -user_input["emotional_valence"]),
            "Overwhelm":       user_input["task_difficulty"] + (10 - user_input["task_clarity"]),
            "No Reward":       user_input["time_to_reward_days"] * 0.2,
            "Perfectionism":   user_input["perfectionism_score"],
            "Low Self-Efficacy": 10 - user_input["self_efficacy_score"],
            "Distractions":    user_input["distraction_level"],
            "Unclear Steps":   (10 - user_input["task_clarity"]) * 1.5,
            "Emo. Avoidance":  max(0, -user_input["emotional_valence"]) * 2,
        }
        max_v = max(causes_raw.values()) or 1
        for k, v in sorted(causes_raw.items(), key=lambda x: -x[1])[:5]:
            pct = v / max_v * 100
            highlight = "background:#4f46e5;" if v == max(causes_raw.values()) else "background:#312e81;"
            st.markdown(f"""
            <div style="margin:0.35rem 0">
                <div style="display:flex;justify-content:space-between;color:#c7d2fe;font-size:0.8rem">
                    <span>{k}</span><span>{pct:.0f}%</span>
                </div>
                <div style="background:#1e1b4b;border-radius:4px;height:6px;margin-top:2px">
                    <div style="width:{pct}%;height:6px;border-radius:4px;{highlight}"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">💡 Behavioral Nudge Plan</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="nudge-box">
            <h4>⚡ Immediate Action</h4>
            <p>{nudge.get('primary_nudge','')}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Evidence-Based Techniques:**")
        for t in nudge.get("techniques", []):
            st.markdown(f'<div class="technique-item">→ {t}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1rem;padding:0.8rem;background:#1e1b4b;border-radius:8px;
                    border-left:3px solid #818cf8;color:#c7d2fe;font-style:italic;font-size:0.9rem">
            💬 {nudge.get('quote','')}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── DOMAIN-SPECIFIC TIPS ──────────────────
    st.markdown('<div class="section-header">🎯 Domain-Specific Strategies</div>', unsafe_allow_html=True)
    domain_tips = {
        "Fitness":         ["Schedule workouts like meetings — calendar block", "Find a workout buddy for accountability", "Start with just 10-minute sessions"],
        "Studying":        ["Active recall over passive reading", "Spaced repetition: Anki, Obsidian", "Study in 45-min blocks with 15-min breaks"],
        "Career Choice":   ["Informational interviews: talk to 3 people in target role", "Run a 30-day experiment before full commitment", "List your non-negotiables first"],
        "Finance":         ["Automate savings on payday", "Track expenses weekly, not monthly", "Name each savings bucket for motivation"],
        "Health Checkup":  ["Bundle the appointment with a reward afterward", "Book it right now — just open the calendar", "Normalize: preventive care is strength, not weakness"],
        "Creative Project":["Embrace ugly first drafts", "Create a 'minimum viable' version in 1 sitting", "Share publicly early for external accountability"],
    }
    tips = domain_tips.get(user_input["domain"], ["Define the very next action", "Set a specific deadline", "Find an accountability partner"])
    t1, t2, t3 = st.columns(3)
    for col, tip in zip([t1, t2, t3], tips[:3]):
        col.markdown(f"""
        <div class="metric-card" style="text-align:left;padding:1rem">
            <div style="color:#818cf8;font-size:1.2rem">✓</div>
            <div style="color:#c7d2fe;font-size:0.9rem;margin-top:0.3rem">{tip}</div>
        </div>""", unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#4f46e580;font-size:0.8rem;padding:1rem">
        🧠 DecisionDelay AI · No sensors · No surveillance · 100% behavioral science<br>
        Built for the <b>AI × Cognition</b> track
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()