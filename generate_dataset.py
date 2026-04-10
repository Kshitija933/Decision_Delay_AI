"""
DecisionDelay AI - Dataset Generation Script
Generates synthetic behavioral dataset for training the delay prediction model.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
N_SAMPLES = 5000
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAINS = ["Fitness", "Studying", "Career Choice", "Finance", "Health Checkup", "Creative Project"]

DELAY_CAUSES = [
    "Fear of Failure",
    "Overwhelm / Complexity",
    "Lack of Immediate Reward",
    "Perfectionism",
    "Low Self-Efficacy",
    "External Distractions",
    "Ambiguity / Unclear Next Step",
    "Emotional Avoidance",
]

NUDGE_MAP = {
    "Fear of Failure":               "Reframe failure as data. Start with a 2-minute version.",
    "Overwhelm / Complexity":        "Break into 3 micro-tasks. Focus only on Step 1.",
    "Lack of Immediate Reward":      "Attach an immediate reward after the task. Use temptation bundling.",
    "Perfectionism":                 "Set a 'good enough' threshold. Ship the draft.",
    "Low Self-Efficacy":             "Recall a past win. Use implementation intentions.",
    "External Distractions":         "Environment design: phone in another room, website blockers.",
    "Ambiguity / Unclear Next Step": "Write the very next physical action in one sentence.",
    "Emotional Avoidance":           "Name the emotion. Do a 5-minute exposure, then re-evaluate.",
}


def generate_sample(i):
    domain = random.choice(DOMAINS)

    # ── Core Inputs ──────────────────────────────
    task_difficulty      = round(random.uniform(1, 10), 2)   # 1=easy, 10=very hard
    time_to_reward       = round(random.uniform(0.5, 365), 1) # days until meaningful reward
    past_failure_loops   = random.randint(0, 12)              # times person tried & quit
    self_efficacy_score  = round(random.uniform(1, 10), 2)    # belief in own ability
    emotional_valence    = round(random.uniform(-5, 5), 2)    # -5=dread, +5=excited
    social_pressure      = round(random.uniform(0, 10), 2)    # external accountability
    task_clarity         = round(random.uniform(1, 10), 2)    # how clear the next step is
    time_available_hrs   = round(random.uniform(0.25, 6), 2)  # hours free today
    distraction_level    = round(random.uniform(1, 10), 2)    # environment noise/interruptions
    perfectionism_score  = round(random.uniform(1, 10), 2)    # tendency toward perfectionism

    # ── Derived Features ─────────────────────────
    reward_proximity     = round(1 / (time_to_reward + 1), 4) # hyperbolic discounting proxy
    failure_weight       = past_failure_loops * (1 - self_efficacy_score / 10)
    clarity_gap          = 10 - task_clarity

    # ── Delay Score (0-100, higher = more likely to delay) ──────────────
    delay_score = (
        task_difficulty       * 3.5
        + time_to_reward      * 0.08
        + past_failure_loops  * 2.5
        - self_efficacy_score * 3.0
        + max(0, -emotional_valence) * 2.0
        - social_pressure     * 1.5
        + clarity_gap         * 2.0
        + distraction_level   * 2.0
        + perfectionism_score * 1.5
        - time_available_hrs  * 1.0
        + np.random.normal(0, 5)  # noise
    )
    delay_score = round(np.clip(delay_score, 0, 100), 2)

    # ── Delay Label ──────────────────────────────
    if delay_score < 30:
        delay_label = "Low"
    elif delay_score < 60:
        delay_label = "Medium"
    else:
        delay_label = "High"

    # ── Primary Delay Cause (rule-based heuristic) ───────────────────────
    cause_scores = {
        "Fear of Failure":               past_failure_loops * 2 + max(0, -emotional_valence),
        "Overwhelm / Complexity":        task_difficulty + clarity_gap,
        "Lack of Immediate Reward":      time_to_reward * 0.2,
        "Perfectionism":                 perfectionism_score,
        "Low Self-Efficacy":             10 - self_efficacy_score,
        "External Distractions":         distraction_level,
        "Ambiguity / Unclear Next Step": clarity_gap * 1.5,
        "Emotional Avoidance":           max(0, -emotional_valence) * 2,
    }
    primary_cause = max(cause_scores, key=cause_scores.get)
    nudge = NUDGE_MAP[primary_cause]

    # ── Action Taken (simulated ground truth) ───────────────────────────
    action_prob = max(0.05, min(0.95, 1 - delay_score / 100))
    acted = int(np.random.random() < action_prob)

    return {
        "id":                   i,
        "domain":               domain,
        "task_difficulty":      task_difficulty,
        "time_to_reward_days":  time_to_reward,
        "past_failure_loops":   past_failure_loops,
        "self_efficacy_score":  self_efficacy_score,
        "emotional_valence":    emotional_valence,
        "social_pressure":      social_pressure,
        "task_clarity":         task_clarity,
        "time_available_hrs":   time_available_hrs,
        "distraction_level":    distraction_level,
        "perfectionism_score":  perfectionism_score,
        "reward_proximity":     reward_proximity,
        "failure_weight":       round(failure_weight, 4),
        "clarity_gap":          round(clarity_gap, 2),
        "delay_score":          delay_score,
        "delay_label":          delay_label,
        "primary_delay_cause":  primary_cause,
        "nudge_recommendation": nudge,
        "acted":                acted,
    }


if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} samples...")
    records = [generate_sample(i) for i in range(N_SAMPLES)]
    df = pd.DataFrame(records)

    out_path = os.path.join(OUTPUT_DIR, "decisiondelay_dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✅ Dataset saved → {out_path}")
    print(f"   Shape: {df.shape}")
    print(f"\nDelay Label Distribution:\n{df['delay_label'].value_counts()}")
    print(f"\nTop Delay Causes:\n{df['primary_delay_cause'].value_counts()}")
    print(f"\nSample rows:\n{df.head(3).T}")