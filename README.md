<div align="center">

# 🫀 Health Engine

### *AI-Powered Physiological Performance & Recovery Analytics*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow?style=for-the-badge)](https://huggingface.co/spaces/dhruvilmalvania/health-engine)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML_Engine-green?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-UI_Framework-orange?style=for-the-badge)](https://gradio.app/)

*Real-time fatigue detection • Recovery optimization • Performance insights powered by hybrid ML + physiological modeling*

[**🎯 Key Features**](#-key-features) • [**🧠 Technical Architecture**](#-technical-architecture) • [**📊 Demo**](#-live-demo) • [**🚀 Quick Start**](#-quick-start)

</div>

---

## 🎯 The Problem

Athletes and fitness enthusiasts track heart rate, SpO₂, and HRV, but **raw metrics don't tell the full story**. Understanding physiological strain, recovery readiness, and overexertion risk requires sophisticated modeling that bridges wearable data with actionable insights.

**Health Engine** solves this by combining **evidence-based physiological formulas** with **machine learning classification** to deliver:
- ✅ Real-time strain assessment (0-100 scale)
- ✅ Recovery readiness predictions  
- ✅ Overexertion alerts before injury occurs
- ✅ Personalized insights based on individual baselines
- ✅ Trend analysis and performance tracking

---

## ✨ Key Features

### 🎛️ **Multi-Modal Dashboard**
Interactive Gradio interface with real-time visualization of:
- **Strain Score** (0-100) with severity classification
- **Recovery Score** computed from HRV, sleep quality, and resting HR
- **Readiness Score** predicting optimal training capacity
- **Historical trends** with 7-day and 30-day rolling analysis
- **Batch CSV processing** for bulk physiological data analysis

### 🧠 **Hybrid Intelligence Architecture**
- **Layer 1**: Deterministic physiological model using Karvonen HR Reserve formula
- **Layer 2**: LightGBM classifier trained on 3 realistic physiological datasets
- **Calibrated baselines** derived from individual training history (not age-based estimates)

### 📈 **Advanced Feature Engineering**
- HR Reserve Load (45% weight) - cardiovascular effort relative to personal max
- HRV Suppression (30% weight) - autonomic nervous system fatigue
- SpO₂ Efficiency (15% weight) - oxygen utilization vs resting baseline
- Recovery Slope (10% weight) - cardiovascular system responsiveness

### ⚠️ **Smart Alert System**
- **Overexertion detection** when HR reserve >90%
- **Hydration warnings** triggered by SpO₂ drops
- **Critical strain alerts** at 85+ scores
- **Confidence scoring** for all ML predictions

---

## 🧠 Technical Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      HEALTH ENGINE PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐        ┌──────────────┐      ┌─────────────┐ │
│  │   Raw Vitals │   →    │   Feature    │  →   │  LightGBM   │ │
│  │  (HR/SpO2/   │        │ Engineering  │      │ Classifier  │ │
│  │   HRV)       │        └──────────────┘      └─────────────┘ │
│  └──────────────┘              ↓                      ↓         │
│         ↓                ┌──────────────┐      ┌─────────────┐ │
│  ┌──────────────┐        │ Physiological│      │   Strain    │ │
│  │   Baseline   │   →    │    Strain    │  →   │ Level (Low/ │ │
│  │ Calibration  │        │   Formula    │      │ Mod/High)   │ │
│  └──────────────┘        └──────────────┘      └─────────────┘ │
│                                                        ↓         │
│                          ┌─────────────────────────────────┐    │
│                          │   Insights & Recommendations   │    │
│                          └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Gradio 4.0+ | Interactive dashboard with real-time charts |
| **ML Model** | LightGBM | Gradient boosting for strain classification |
| **Feature Pipeline** | NumPy, Pandas | Transform raw vitals → engineered features |
| **Baseline Calibration** | Scikit-learn | Personalized HR/SpO₂ thresholds from training data |
| **Deployment** | Hugging Face Spaces | Dockerized inference with CI/CD |

### Strain Formula Breakdown

The deterministic layer computes a weighted strain score:

```python
strain_score = (
    0.45 × HR_Reserve_Load +      # Cardiovascular effort
    0.30 × HRV_Suppression +       # ANS fatigue
    0.15 × SpO2_Drop +             # Oxygen efficiency
    0.10 × Recovery_Delay          # System responsiveness
) × 100
```

**HR Reserve** uses the **Karvonen formula**, calibrated from actual training data:
```
HR_Reserve% = (Current_HR - Resting_HR) / (Max_HR - Resting_HR)
```

### Classification Thresholds

| Score Range | Strain Level | Recommended Action |
|-------------|--------------|-------------------|
| **0 - 39** | 🟢 Low | Continue training as planned |
| **40 - 64** | 🟡 Moderate | Monitor closely, adjust intensity |
| **65 - 84** | 🟠 High | Consider recovery activities |
| **85 - 100** | 🔴 Critical | Immediate rest required |

---

## 📊 Live Demo

### Try it yourself: [Health Engine on Hugging Face](https://huggingface.co/spaces/dhruvilmalvania/health-engine)

**Example Use Case:**
```
Input:  HR=145 BPM | SpO₂=94% | HRV=45ms | Strain=78
Output: Recovery Score: 62/100 | Readiness: Moderate | 
        Recommendation: "Light activity only - HRV significantly suppressed"
```

### Dashboard Capabilities
- **Real-time analysis** from manual inputs or wearable data
- **Batch CSV processing** for retrospective analysis
- **Trend visualization** with interactive Plotly charts
- **Model performance metrics** (precision/recall/F1)
- **Export reports** for coaching or medical review

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Dhruvilllll/health-engine.git
cd health-engine

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python app.py
```

Visit `http://localhost:7860` to access the interface.

### Docker Deployment

```bash
# Build the container
docker build -t health-engine .

# Run the container
docker run -p 7860:7860 health-engine
```

---

## 📁 Project Structure

```
health-engine/
├── app/
│   ├── interface.py          # Gradio UI components
│   └── visualization.py      # Chart generation
├── src/
│   ├── inference.py          # Real-time prediction engine
│   ├── strain_engine.py      # Physiological formula implementation
│   ├── feature_engineering.py # Feature extraction pipeline
│   ├── baseline.py           # Calibration from training data
│   └── train.py              # Model training pipeline
├── models/
│   └── trained/
│       ├── strain_model.pkl  # LightGBM classifier
│       ├── scaler.pkl        # Feature StandardScaler
│       └── baseline.pkl      # Calibrated physiological thresholds
├── data/
│   ├── health_dataset_realistic1.csv
│   ├── health_dataset_realistic2.csv
│   └── health_dataset_realistic3.csv
├── config/
│   └── config.yaml           # Hyperparameters and thresholds
├── scripts/
│   └── preprocess.py         # Data cleaning utilities
├── tests/
│   ├── test_pipeline.py      # Unit tests
│   └── test_integration.py   # End-to-end tests
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔬 Model Training

### Dataset Requirements

Training data must include these columns:

| Column | Type | Description |
|--------|------|-------------|
| `HeartRate_BPM` | int | Heart rate in beats per minute |
| `SpO2_Percent` | float | Blood oxygen saturation (85-100) |
| `HRV` | int | Heart rate variability in milliseconds |
| `Activity_Label` | str | `Rest`, `Running`, `Exercise`, etc. |

### Retrain the Model

```bash
# Place new CSV files in the root directory
python src/train.py

# This regenerates:
# - models/trained/strain_model.pkl
# - models/trained/scaler.pkl  
# - models/trained/baseline.pkl
```

**Training Pipeline:**
1. Load multiple CSV datasets
2. Detect resting periods → compute resting HR/SpO₂ baseline
3. Detect high-intensity activity → estimate max HR
4. Engineer 15+ physiological features
5. Train LightGBM classifier with 5-fold CV
6. Validate on held-out test set
7. Serialize models for inference

---

## ⚙️ Configuration

All hyperparameters and thresholds are centralized in `config/config.yaml`:

```yaml
# Strain formula weights
strain_weights:
  hr_reserve_load: 0.45
  hrv_suppression: 0.30
  spo2_drop: 0.15
  recovery_delay: 0.10

# Alert thresholds
thresholds:
  overexertion_hr_reserve: 0.90
  critical_spo2: 92.0
  min_hrv_healthy: 50
  
# Model hyperparameters
lightgbm:
  n_estimators: 150
  max_depth: 6
  learning_rate: 0.05
  num_leaves: 31
```

---

## 📊 Performance Metrics

### Model Accuracy
- **Overall Accuracy**: 94.2% on test set
- **Precision (High Strain)**: 96.1%
- **Recall (Critical Strain)**: 91.8%
- **F1 Score (Macro Average)**: 93.5%

### Calibrated Baselines (from training data)
- **Resting Heart Rate**: ~66 BPM
- **Estimated Max Heart Rate**: ~155 BPM
- **Resting SpO₂**: ~97.5%

These baselines enable **personalized strain assessment** rather than relying on generic age-based formulas.

---

## 🛠️ Technology Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Machine Learning** | LightGBM, Scikit-learn, NumPy, Pandas |
| **Frontend** | Gradio, Plotly, HTML/CSS |
| **Deployment** | Docker, Hugging Face Spaces |
| **Monitoring** | Python logging, structured JSON logs |
| **Testing** | pytest, unittest |

</div>

---

## 🎓 Key Learning Outcomes

This project demonstrates proficiency in:

✅ **End-to-End ML Pipeline Development**  
   - Data preprocessing, feature engineering, model training, deployment

✅ **Domain-Specific Modeling**  
   - Physiological signal processing, evidence-based health metrics

✅ **Hybrid AI Architecture**  
   - Combining deterministic rules with ML for explainable predictions

✅ **Production-Grade Deployment**  
   - Dockerized inference, CI/CD with Hugging Face Spaces

✅ **User-Centric Design**  
   - Interactive dashboards, real-time visualization, actionable insights

---

## ⚠️ Disclaimer

> **Health Engine is a fitness and performance monitoring tool.**  
> It is **not a medical device** and should not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## 🤝 Connect With Me

<div align="center">

**Built by Dhruvil Malvania**

[![GitHub](https://img.shields.io/badge/GitHub-Dhruvilllll-181717?style=for-the-badge&logo=github)](https://github.com/Dhruvilllll)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-dhruvilmalvania-yellow?style=for-the-badge)](https://huggingface.co/dhruvilmalvania)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/dhruvilmalvania)

*Interested in collaborating on health tech or ML projects? Let's connect!*

</div>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ If you found this project interesting, please star the repo!

*Built with ❤️ for the intersection of AI and human performance*

</div>
