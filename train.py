# ==========================================
# Statmize Health Engine - Training Pipeline
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import config
from baseline import BaselineCalibrator
from feature_engineering import FeatureEngineer
from strain_engine import StrainEngine


# --------------------------------------------------
# 1️⃣ Load All Datasets
# --------------------------------------------------

def load_datasets():
    df1 = pd.read_csv("health_dataset_realistic1.csv")
    df2 = pd.read_csv("health_dataset_realistic2.csv")
    df3 = pd.read_csv("health_dataset_realistic3.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)
    return df


# --------------------------------------------------
# 2️⃣ Prepare Training Data
# --------------------------------------------------

def prepare_training_data(df):

    # Baseline calibration
    calibrator = BaselineCalibrator()
    calibrator.compute_resting_baseline(df)
    calibrator.estimate_max_hr(df)

    # Feature engineering
    engineer = FeatureEngineer(calibrator)
    df = engineer.build_features(df)

    # Generate pseudo labels from deterministic strain
    strain_engine = StrainEngine()
    df = strain_engine.evaluate_dataframe(df)

    # Feature columns for ML
    feature_cols = [
        "hr_reserve_load",
        "spo2_drop",
        "hrv_suppression",
        "recovery_slope",
        "rolling_hr_mean"
    ]

    X = df[feature_cols]
    y = df["strain_level"]

    return X, y, calibrator


# --------------------------------------------------
# 3️⃣ Train Model
# --------------------------------------------------

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    if config.USE_FEATURE_SCALING:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    # Choose model
    if config.MODEL_TYPE == "lightgbm" and LIGHTGBM_AVAILABLE:
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=config.RANDOM_STATE
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=config.RANDOM_STATE
        )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("\n===== Model Evaluation =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, scaler


# --------------------------------------------------
# 4️⃣ Save Model
# --------------------------------------------------

def save_artifacts(model, scaler, calibrator):

    joblib.dump(model, "strain_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(calibrator.get_baseline_dict(), "baseline.pkl")

    print("\nModel and artifacts saved successfully.")


# --------------------------------------------------
# 5️⃣ Main Training Flow
# --------------------------------------------------

if __name__ == "__main__":

    df = load_datasets()

    X, y, calibrator = prepare_training_data(df)

    model, scaler = train_model(X, y)

    save_artifacts(model, scaler, calibrator)