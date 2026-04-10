# 🧠 DecisionDelay AI

**DecisionDelay AI** is a behavioral science-powered analytical tool designed to decode the "Action-Intention Gap." It uses Machine Learning to predict the risk of decision delay (procrastination) and provides evidence-based "nudges" to help users transition from planning to execution.

---

## 🚀 Key Features

- **Behavioral Analysis Engine**: Identifies the root cause of delay (Fear of Failure, Overwhelm, Perfectionism, etc.) using key psychological parameters.
- **AI-Powered Predictions**: Classifies delay risk (Low, Medium, High) using a Voting Ensemble (Random Forest, XGBoost, Gradient Boosting).
- **Interactive Dashboard**:
  - **Risk Gauges**: Real-time visual feedback on delay probability.
  - **Radar Profiles**: Visualizes your "Risk DNA" across 6 behavioral dimensions.
  - **Class Distributions**: Shows model confidence scores.
- **Personalized Nudges**: Generates actionable, evidence-based strategies tailored to your specific psychological profile.
- **EDA & Performance Tracking**: Dedicated pages for analyzing dataset trends and monitoring ML model accuracy.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (with custom CSS injection for premium aesthetics).
- **Machine Learning**: Scikit-learn, XGBoost, Joblib.
- **Data Engineering**: Pandas, NumPy.
- **Visualization**: Plotly (Radar/Gauge charts), Seaborn/Matplotlib (Training reports).

---

## 📂 Project Structure

```text
Decision_Delay_AI/
├── app.py                # Main Entry Point (Streamlit UI)
├── 1_EDA_Dashboard.py    # Exploratory Data Analysis Page
├── Model_Performance.py  # Model Metrics & Evaluation Page
├── generate_dataset.py   # Synthetic Behavioral Data Generator
├── train_model.py        # ML Training Pipeline (Ensemble builds)
├── predict.py            # Inference Engine Logic
├── nudge_engine.py       # Behavioral Strategy Library
├── data/                 # Raw/Processed Datasets
├── models/               # Saved Artifacts (.pkl, .json)
├── reports/              # Model Performance Plots (Confusion Matrix, etc.)
└── requirements.txt      # Python Dependencies
```

---

## 🔧 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Decision_Delay_AI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Generate Data & Train Model**:
   If you want to refresh the models from scratch:
   ```bash
   python generate_dataset.py
   python train_model.py
   ```

4. **Launch the App**:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 How it Works

1. **Input**: Users provide data on task difficulty, time-to-reward, emotional valence, and personal traits.
2. **Detection**: The system calculates a **Delay Risk Score** using either the trained ML Ensemble or a Rule-Based fallback engine.
3. **Diagnosis**: It identifies the most likely "Primary Cause" based on feature importance weights.
4. **Intervention**: It selects a high-leverage "Nudge" from the behavioral library to break the inertia.

---

## 📊 Model Performance

The current ensemble achieves **90-95% Accuracy** on test data, utilizing a combination of:
- **Random Forest** (Robustness)
- **XGBoost** (Precision)
- **Gradient Boosting** (Pattern Recognition)

---

## 📜 License
This project is built for the **AI × Cognition** track and is intended for educational and analytical purposes.
