---
Physiological Strain Analyzer

> Real-time fatigue & overexertion detection powered by a hybrid physiological model + LightGBM classifier.

---

## 🧠 What Is This?

The **Health Engine** is a physiological strain analyzer designed to help athletes, coaches, and fitness enthusiasts understand how hard the body is working — in real time.

It takes three simple biometric inputs (Heart Rate, SpO₂, and optional HRV) and outputs:

- A **Strain Score** from 0–100
- A **Strain Level** classification (Low / Moderate / High / Critical)
- **Overexertion Risk** detection
- **Hydration Alerts** based on oxygen saturation drops
- A **personalized analysis summary** with actionable insights

---

## 🚀 How To Use

1. **Set your Heart Rate (BPM)** — use a wearable or manual pulse check
2. **Set your SpO₂ (%)** — use a pulse oximeter or smartwatch reading
3. **Optionally set HRV** — leave at 0 to skip (uses fallback weighting)
4. Hit **⚡ Analyze Strain** and review your results

---

## ⚙️ How It Works

The engine uses a **hybrid two-layer approach** for maximum accuracy and transparency:

### Layer 1 — Deterministic Physiological Formula

The strain score (0–100) is computed using a weighted formula across four physiological signals:

| Signal | Weight | What It Measures |
|---|---|---|
| HR Reserve Load | 45% | Cardiovascular effort relative to personal resting & max HR |
| HRV Suppression | 30% | Autonomic nervous system fatigue |
| SpO₂ Drop | 15% | Oxygen efficiency vs resting baseline |
| Recovery Slope | 10% | How quickly HR recovers between efforts |

**HR Reserve** is calculated using the Karvonen method — data-driven from your training history, not age-formula estimates.

### Layer 2 — ML Classification (LightGBM)

A **LightGBM classifier** trained on 3 realistic physiological datasets maps the engineered features to a strain category:

```
Low  →  Moderate  →  High  →  Critical
```

Model confidence is reported alongside every prediction so you know how certain the classification is.

### Strain Thresholds

| Score Range | Level |
|---|---|
| 0 – 39 | 🟢 Low |
| 40 – 64 | 🟡 Moderate |
| 65 – 84 | 🟠 High |
| 85 – 100 | 🔴 Critical |

---

## 📦 Project Structure

```
├── app.py                  # Gradio UI — entry point
├── inference.py            # Real-time prediction engine
├── strain_engine.py        # Physiological strain formula
├── feature_engineering.py  # Feature computation from raw vitals
├── baseline.py             # Resting & max HR calibration
├── train.py                # Model training pipeline
├── config.py               # All thresholds and weights
├── strain_model.pkl        # Trained LightGBM classifier
├── scaler.pkl              # Feature StandardScaler
├── baseline.pkl            # Calibrated physiological baseline
├── health_dataset_realistic1.csv
├── health_dataset_realistic2.csv
├── health_dataset_realistic3.csv
└── requirements.txt
```

---

## 📊 Baseline Calibration

The model was calibrated on realistic physiological training data with the following derived baselines:

| Metric | Calibrated Value |
|---|---|
| Resting Heart Rate | ~66 BPM |
| Estimated Max Heart Rate | ~155 BPM |
| Resting SpO₂ | ~97.5% |

These baselines are used to compute personalized HR Reserve and SpO₂ drop scores — meaning the strain score is **relative to the individual**, not a generic population average.

---

## 🏗️ Running Locally

### 1. Clone the repo

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ -health-engine
cd  -health-engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model

If you want to retrain on new data, place your CSVs in the root directory and run:

```bash
python train.py
```

This will regenerate `strain_model.pkl`, `scaler.pkl`, and `baseline.pkl`.

### 4. Launch the app

```bash
python app.py
```

App will be available at `http://localhost:7860`

---

## 📋 Requirements

```
gradio>=4.0.0
scikit-learn
lightgbm
joblib
numpy
pandas
```

---

## 🔬 Training Data Format

Your CSV files must contain the following columns:

| Column | Description |
|---|---|
| `HeartRate_BPM` | Heart rate in beats per minute |
| `SpO2_Percent` | Blood oxygen saturation percentage |
| `Activity_Label` | Activity type: `Rest`, `Running`, `Exercise`, etc. |
| `HRV` *(optional)* | Heart rate variability in ms |

The `Activity_Label` column is used during training to detect resting periods for baseline calibration and high-intensity periods for max HR estimation.

---

## 🛠️ Configuration

All key thresholds and weights are centralized in `config.py` — no need to hunt through the codebase to tune behaviour:

```python
# Strain formula weights
STRAIN_WEIGHTS = {
    "hr_reserve_load": 0.45,
    "hrv_suppression": 0.30,
    "spo2_drop": 0.15,
    "recovery_delay": 0.10
}

# Overexertion threshold (HR reserve ratio)
OVEREXERTION_HR_RESERVE = 0.90

# Critical SpO₂
CRITICAL_SPO2 = 92.0
```

---

## ⚠️ Disclaimer

This tool is intended for **fitness and performance monitoring purposes only**. It is **not a medical device** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you experience symptoms of distress, consult a qualified healthcare professional.

---
