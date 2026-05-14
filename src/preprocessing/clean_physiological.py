"""
===========================================================
Health Engine — Physiological Cleaning Pipeline
===========================================================

Purpose:
--------
Clean and engineer physiological wearable data
for recovery prediction.

This pipeline:
- handles missing values
- removes outliers
- parses timestamps
- engineers physiological features
- creates rolling trends
- normalizes health metrics

Author:
-------
Dhruvil Malvania

===========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/processed/physiological_merged.csv"

OUTPUT_PATH = "data/processed"

OUTPUT_FILE = "physiological_cleaned.csv"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\n Loading merged physiological dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f" Dataset Shape: {df.shape}")

    return df


# =========================================================
# DETECT TIMESTAMP COLUMN
# =========================================================

def detect_timestamp_column(df):

    possible_timestamp_cols = [
        "date",
        "created_at",
        "timestamp",
        "cycle_start",
        "start_time"
    ]

    for col in possible_timestamp_cols:

        if col in df.columns:
            return col

    return None


# =========================================================
# PARSE TIMESTAMPS
# =========================================================

def parse_timestamps(df):

    timestamp_col = detect_timestamp_column(df)

    if timestamp_col:

        print(f"\n Parsing timestamp column: {timestamp_col}")

        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col],
            errors="coerce"
        )

        df = df.sort_values(timestamp_col)

    else:
        print("\n No timestamp column found.")

    return df


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

def handle_missing_values(df):

    print("\n Handling missing values...")

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        median_value = df[col].median()

        df[col] = df[col].fillna(
            median_value
        )

    return df


# =========================================================
# REMOVE PHYSIOLOGICAL OUTLIERS
# =========================================================

def remove_outliers(df):

    print("\n Removing physiological outliers...")

    physiological_limits = {
        "heart_rate_variability_ms": (10, 250),
        "resting_heart_rate_bpm": (30, 120),
        "blood_oxygen_": (70, 100),
        "respiratory_rate_rpm": (5, 40),
        "day_strain": (0, 25),
    }

    for col, (lower, upper) in physiological_limits.items():

        if col in df.columns:

            before = df.shape[0]

            df = df[
                (df[col] >= lower) &
                (df[col] <= upper)
            ]

            after = df.shape[0]

            print(
                f"{col}: removed {before - after} outliers"
            )

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def feature_engineering(df):

    print("\n Creating physiological features...")

    # -----------------------------------------------------
    # STRESS RATIO
    # -----------------------------------------------------

    if (
        "resting_heart_rate_bpm" in df.columns and
        "heart_rate_variability_ms" in df.columns
    ):

        df["stress_ratio"] = (
            df["resting_heart_rate_bpm"] /
            (df["heart_rate_variability_ms"] + 1e-5)
        )

    # -----------------------------------------------------
    # SLEEP DEFICIT
    # -----------------------------------------------------

    if "sleep_performance_" in df.columns:

        df["sleep_deficit"] = (
            100 -
            df["sleep_performance_"]
        )

    # -----------------------------------------------------
    # HRV RECOVERY BALANCE
    # -----------------------------------------------------

    if (
        "heart_rate_variability_ms" in df.columns and
        "day_strain" in df.columns
    ):

        df["hrv_strain_balance"] = (
            df["heart_rate_variability_ms"] /
            (df["day_strain"] + 1)
        )

    return df


# =========================================================
# ROLLING FEATURES
# =========================================================

def rolling_features(df):

    print("\n Creating rolling physiological trends...")

    # -----------------------------------------------------
    # HRV ROLLING MEAN
    # -----------------------------------------------------

    if "heart_rate_variability_ms" in df.columns:

        df["hrv_7d_avg"] = (
            df["heart_rate_variability_ms"]
            .rolling(window=7, min_periods=1)
            .mean()
        )

    # -----------------------------------------------------
    # STRAIN ROLLING MEAN
    # -----------------------------------------------------

    if "day_strain" in df.columns:

        df["strain_7d_avg"] = (
            df["day_strain"]
            .rolling(window=7, min_periods=1)
            .mean()
        )

    # -----------------------------------------------------
    # RECOVERY TREND
    # -----------------------------------------------------

    if "recovery_score" in df.columns:

        df["recovery_trend"] = (
            df["recovery_score"]
            .diff()
        )

    return df


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_features(df):

    print("\n Keeping physiological metrics on raw input scale...")

    return df


# =========================================================
# FINAL DATASET SUMMARY
# =========================================================

def final_summary(df):

    print("\n FINAL DATASET SUMMARY")
    print("=" * 50)

    print(f"\n Final Shape: {df.shape}")

    print("\n Missing Values:")
    print(df.isnull().sum().sum())

    print("\n Columns:")
    print(df.columns.tolist())


# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset(df):

    output_path = os.path.join(
        OUTPUT_PATH,
        OUTPUT_FILE
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n Cleaned dataset saved:")
    print(output_path)


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n================================================")
    print(" HEALTH ENGINE — CLEANING PIPELINE")
    print("================================================")

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    df = load_data()

    # -----------------------------------------------------
    # TIMESTAMPS
    # -----------------------------------------------------

    df = parse_timestamps(df)

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    df = handle_missing_values(df)

    # -----------------------------------------------------
    # OUTLIERS
    # -----------------------------------------------------

    df = remove_outliers(df)

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    df = feature_engineering(df)

    # -----------------------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------------------

    df = rolling_features(df)

    # -----------------------------------------------------
    # NORMALIZATION
    # -----------------------------------------------------

    df = normalize_features(df)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    final_summary(df)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_dataset(df)

    print("\n CLEANING PIPELINE COMPLETE")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
