"""
DecisionDelay AI - Inference Engine
Loads trained model artifacts and provides prediction utilities.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")

# ─────────────────────────────────────────────
# NUDGE LIBRARY
# ─────────────────────────────────────────────
NUDGE_LIBRARY = {
    "Fear of Failure": {
        "title": "Fear of Failure",
        "emoji": "😨",
        "primary_nudge": "Reframe: every attempt is data, not judgement. Start a 2-minute version RIGHT NOW.",
        "techniques": [
            "Implementation Intention: 'When X happens, I will do Y for 2 minutes.'",
            "Failure Pre-mortem: Write out the worst case. Is it survivable? Usually yes.",
            "Past-win recall: Name one time you succeeded despite fear.",
        ],
        "quote": "\"You don't have to be great to start, but you have to start to be great.\" — Zig Ziglar",
    },
    "Overwhelm / Complexity": {
        "title": "Overwhelm / Complexity",
        "emoji": "🌊",
        "primary_nudge": "Break it into 3 micro-tasks. Focus ONLY on micro-task #1.",
        "techniques": [
            "Task decomposition: List every physical sub-action.",
            "Time-boxing: Commit to 25 minutes (Pomodoro), then stop.",
            "Progress illusion: Check off sub-tasks visibly for dopamine.",
        ],
        "quote": "\"Divide each difficulty into as many parts as possible.\" — Descartes",
    },
    "Lack of Immediate Reward": {
        "title": "Lack of Immediate Reward",
        "emoji": "⏳",
        "primary_nudge": "Attach an INSTANT reward after completion. Temptation-bundle it.",
        "techniques": [
            "Temptation bundling: Only do X (guilty pleasure) while working on Y.",
            "Commitment device: Pre-commit publicly to a deadline.",
            "Visualize the future self who completed this task.",
        ],
        "quote": "\"Act in the present; invest in the future.\" — Seneca",
    },
    "Perfectionism": {
        "title": "Perfectionism",
        "emoji": "🎯",
        "primary_nudge": "Set a 'good enough' bar right now. Ship the draft. Perfect is the enemy of done.",
        "techniques": [
            "Version 0.1 mindset: Create the ugliest working version first.",
            "Time constraint: Give yourself exactly 30 min, then send/submit.",
            "Decoupling identity: Your work ≠ your worth.",
        ],
        "quote": "\"Done is better than perfect.\" — Sheryl Sandberg",
    },
    "Low Self-Efficacy": {
        "title": "Low Self-Efficacy",
        "emoji": "💪",
        "primary_nudge": "Recall a past win. Use it as evidence you can do hard things.",
        "techniques": [
            "Mastery experiences: Start with an easier version to build confidence.",
            "Vicarious learning: Find someone like you who succeeded.",
            "Verbal persuasion: Read testimonials or write affirmations.",
        ],
        "quote": "\"Whether you think you can or think you can't — you're right.\" — Henry Ford",
    },
    "External Distractions": {
        "title": "External Distractions",
        "emoji": "📵",
        "primary_nudge": "Environment design: phone in another room, noise-cancelling on, tab limiter active.",
        "techniques": [
            "Friction increase: Make distractions 20 seconds harder to access.",
            "Designated deep-work space: Only work at this desk/spot.",
            "Phone in airplane mode for 25-minute sprints.",
        ],
        "quote": "\"The ability to concentrate single-mindedly is the most valuable skill.\" — Brian Tracy",
    },
    "Ambiguity / Unclear Next Step": {
        "title": "Ambiguity / Unclear Next Step",
        "emoji": "🧭",
        "primary_nudge": "Write the VERY NEXT physical action in one sentence, right now.",
        "techniques": [
            "GTD Next Action: 'What is the next visible, physical action to move this forward?'",
            "If-then planning: 'If I sit down at 9am, I will open file X and write Y.'",
            "5-min planning session before starting any work block.",
        ],
        "quote": "\"A good plan today is better than a perfect plan tomorrow.\" — Patton",
    },
    "Emotional Avoidance": {
        "title": "Emotional Avoidance",
        "emoji": "🧠",
        "primary_nudge": "Name the emotion out loud. Do 5 minutes of exposure, then re-evaluate.",
        "techniques": [
            "Emotion labelling: 'I notice I feel ___ about this task.'",
            "RAIN technique: Recognize, Allow, Investigate, Nurture.",
            "Opposite action: If you feel like running, gently walk toward the task.",
        ],
        "quote": "\"Between stimulus and response, there is a space.\" — Viktor Frankl",
    },
}


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
class DecisionDelayPredictor:
    def __init__(self, model_type="ensemble"):
        self.model_type = model_type
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            model_file = "ensemble_model.pkl" if self.model_type == "ensemble" else "best_model.pkl"
            self.model   = joblib.load(MODEL_DIR / model_file)
            self.scaler  = joblib.load(MODEL_DIR / "scaler.pkl")
            self.le      = joblib.load(MODEL_DIR / "label_encoder.pkl")
            with open(MODEL_DIR / "feature_cols.json") as f:
                self.feature_cols = json.load(f)
            with open(MODEL_DIR / "model_meta.json") as f:
                self.meta = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                "Model files not found. Run training/train_model.py first."
            )

    def _engineer_features(self, raw: dict) -> pd.DataFrame:
        """Compute derived features from raw user input."""
        d = raw.copy()
        d["reward_proximity"] = round(1 / (d["time_to_reward_days"] + 1), 4)
        d["failure_weight"]   = d["past_failure_loops"] * (1 - d["self_efficacy_score"] / 10)
        d["clarity_gap"]      = 10 - d["task_clarity"]
        return pd.DataFrame([{col: d[col] for col in self.feature_cols}])

    def predict(self, raw_input: dict) -> dict:
        """
        raw_input: dict with keys matching FEATURE_COLS (pre-derived)
                   OR user-facing keys that get auto-derived.
        Returns full prediction payload.
        """
        X_df     = self._engineer_features(raw_input)
        X_scaled = self.scaler.transform(X_df)

        label_enc  = self.model.predict(X_scaled)[0]
        label      = self.le.inverse_transform([label_enc])[0]
        proba      = self.model.predict_proba(X_scaled)[0]
        proba_dict = {cls: round(float(p), 4)
                      for cls, p in zip(self.le.classes_, proba)}

        # Determine primary delay cause from feature magnitudes
        cause = self._infer_cause(raw_input)
        nudge_data = NUDGE_LIBRARY.get(cause, {})

        # Delay score (rule-based, consistent with dataset)
        delay_score = self._compute_delay_score(raw_input)

        return {
            "delay_label":      label,
            "delay_score":      round(delay_score, 1),
            "probabilities":    proba_dict,
            "primary_cause":    cause,
            "nudge":            nudge_data,
            "confidence":       round(max(proba), 4),
            "model_used":       self.meta.get("best_model", self.model_type),
        }

    def _compute_delay_score(self, d: dict) -> float:
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
        return float(np.clip(score, 0, 100))

    def _infer_cause(self, d: dict) -> str:
        scores = {
            "Fear of Failure":               d["past_failure_loops"] * 2 + max(0, -d["emotional_valence"]),
            "Overwhelm / Complexity":        d["task_difficulty"] + (10 - d["task_clarity"]),
            "Lack of Immediate Reward":      d["time_to_reward_days"] * 0.2,
            "Perfectionism":                 d["perfectionism_score"],
            "Low Self-Efficacy":             10 - d["self_efficacy_score"],
            "External Distractions":         d["distraction_level"],
            "Ambiguity / Unclear Next Step": (10 - d["task_clarity"]) * 1.5,
            "Emotional Avoidance":           max(0, -d["emotional_valence"]) * 2,
        }
        return max(scores, key=scores.get)


# ─────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────
def batch_predict(csv_path: str, output_path: str):
    predictor = DecisionDelayPredictor()
    df = pd.read_csv(csv_path)

    results = []
    for _, row in df.iterrows():
        pred = predictor.predict(row.to_dict())
        results.append({**row.to_dict(),
                        "predicted_label": pred["delay_label"],
                        "delay_score":     pred["delay_score"],
                        "primary_cause":   pred["primary_cause"],
                        "confidence":      pred["confidence"]})

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Batch predictions saved → {output_path}")
    return out_df


if __name__ == "__main__":
    # Quick smoke test
    test_input = {
        "task_difficulty":     8.0,
        "time_to_reward_days": 90.0,
        "past_failure_loops":  5,
        "self_efficacy_score": 3.5,
        "emotional_valence":   -3.0,
        "social_pressure":     2.0,
        "task_clarity":        4.0,
        "time_available_hrs":  1.0,
        "distraction_level":   7.0,
        "perfectionism_score": 8.0,
    }
    predictor = DecisionDelayPredictor()
    result = predictor.predict(test_input)
    print("\n🔍 Prediction Result:")
    for k, v in result.items():
        if k != "nudge":
            print(f"  {k}: {v}")
    print(f"\n💡 Nudge: {result['nudge'].get('primary_nudge', 'N/A')}")