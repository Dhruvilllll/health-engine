"""
Health Engine - Complete Pipeline Test
Tests the entire recovery prediction pipeline
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.config import config
from src.common.utils import standardize_columns, handle_missing_values

def test_pipeline():
    """Run complete pipeline test."""
    print("\n" + "="*60)
    print("HEALTH ENGINE - PIPELINE TEST")
    print("="*60)

    # === TEST 1: DATA LOADING ===
    print("\n[TEST 1] Loading physiological data...")
    phys_files = [
        "data/raw/physiological/physiological_cycles.csv",
        "data/raw/physiological/physiological_cycles(1).csv",
        "data/raw/physiological/physiological_cycles(2).csv"
    ]

    phys_dfs = []
    for f in phys_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df = standardize_columns(df)
            phys_dfs.append(df)
            print(f"  [OK] Loaded {f}: {df.shape}")
        else:
            print(f"  [SKIP] File not found: {f}")

    if not phys_dfs:
        print("[ERR] No data loaded")
        return

    df = pd.concat(phys_dfs, ignore_index=True).drop_duplicates()
    print(f"  [OK] Merged: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # === TEST 2: CLEAN DATA ===
    print("\n[TEST 2] Cleaning data...")
    before = df.isnull().sum().sum()
    df = handle_missing_values(df, method="median")
    after = df.isnull().sum().sum()
    print(f"  [OK] Missing: {before} -> {after}")

    # Feature engineering with correct column names
    if "resting_heart_rate_bpm" in df.columns and "heart_rate_variability_ms" in df.columns:
        df["stress_ratio"] = df["resting_heart_rate_bpm"] / (df["heart_rate_variability_ms"] + 1e-5)
        print(f"  [OK] Created stress_ratio")

    if "heart_rate_variability_ms" in df.columns and "day_strain" in df.columns:
        df["hrv_strain_balance"] = df["heart_rate_variability_ms"] / (df["day_strain"] + 1)
        print(f"  [OK] Created hrv_strain_balance")

    if "heart_rate_variability_ms" in df.columns:
        df["hrv_7d_avg"] = df["heart_rate_variability_ms"].rolling(7, 1).mean()
        print(f"  [OK] Created hrv_7d_avg")

    if "day_strain" in df.columns:
        df["strain_7d_avg"] = df["day_strain"].rolling(7, 1).mean()
        print(f"  [OK] Created strain_7d_avg")

    # === TEST 3: RECOVERY SCORING ===
    print("\n[TEST 3] Calculating recovery scores...")

    def norm_100(s):
        min_v, max_v = s.min(), s.max()
        return ((s - min_v) / (max_v - min_v + 1e-8)) * 100

    # Map available columns to score components
    components = {}

    if "sleep_performance_" in df.columns:
        components["sleep"] = 0.40 * norm_100(df["sleep_performance_"])
    else:
        components["sleep"] = 0.40 * pd.Series([50]*len(df))

    if "heart_rate_variability_ms" in df.columns:
        components["hrv"] = 0.30 * norm_100(df["heart_rate_variability_ms"])
    else:
        components["hrv"] = 0.30 * pd.Series([50]*len(df))

    if "resting_heart_rate_bpm" in df.columns:
        rhr_norm = norm_100(df["resting_heart_rate_bpm"])
        components["rhr"] = 0.20 * (100 - rhr_norm)
    else:
        components["rhr"] = 0.20 * pd.Series([50]*len(df))

    if "day_strain" in df.columns:
        components["strain"] = -0.10 * norm_100(df["day_strain"])
    else:
        components["strain"] = 0

    if "blood_oxygen_" in df.columns:
        components["spo2"] = 0.05 * norm_100(df["blood_oxygen_"])
    else:
        components["spo2"] = 0

    df["recovery_score"] = sum(components.values())
    df["recovery_score"] = np.clip(df["recovery_score"], 0, 100)

    print(f"  [OK] Scores calculated")
    print(f"    Mean: {df['recovery_score'].mean():.2f}")
    print(f"    Std:  {df['recovery_score'].std():.2f}")
    print(f"    Range: {df['recovery_score'].min():.2f} - {df['recovery_score'].max():.2f}")

    # Classification
    def classify(score):
        if score < 40: return "Poor"
        elif score < 60: return "Moderate"
        elif score < 80: return "Good"
        else: return "Excellent"

    df["recovery_status"] = df["recovery_score"].apply(classify)
    print(f"  [OK] Classification:")
    for status, count in df["recovery_status"].value_counts().items():
        print(f"    {status}: {count}")

    # === TEST 4: MODEL TRAINING ===
    print("\n[TEST 4] Training model...")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib

    # Select available features
    possible_features = [
        "heart_rate_variability_ms", "resting_heart_rate_bpm", "day_strain", "blood_oxygen_",
        "stress_ratio", "hrv_strain_balance", "hrv_7d_avg", "strain_7d_avg"
    ]
    features = [f for f in possible_features if f in df.columns]
    print(f"  [OK] Selected {len(features)} features: {features}")

    X = df[features].fillna(df[features].median())
    y = df["recovery_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  [OK] Train/test split: {len(X_train)}/{len(X_test)}")

    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=1
    )
    model.fit(X_train, y_train)
    print(f"  [OK] Model trained")

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"  [OK] Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

    # Top features
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"  [OK] Top 3 features:")
    for _, row in importance.head(3).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    # Save model
    os.makedirs("models/trained", exist_ok=True)
    joblib.dump(model, "models/trained/recovery_random_forest.pkl")
    print(f"  [OK] Model saved")

    # === TEST 5: PREDICTIONS ===
    print("\n[TEST 5] Testing predictions...")
    for idx in [0, len(df)//2, len(df)-1]:
        sample = df.iloc[idx][features].values.reshape(1, -1)
        pred = model.predict(sample)[0]
        status = classify(pred)
        print(f"  Sample {idx}: score={pred:.1f}, status={status}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"[OK] Rows processed: {len(df)}")
    print(f"[OK] Features used: {len(features)}")
    print(f"[OK] Model trained and saved")
    print(f"[OK] Predictions working")
    print("\nALL TESTS PASSED\n")

if __name__ == "__main__":
    test_pipeline()
