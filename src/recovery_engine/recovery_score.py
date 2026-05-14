"""
===========================================================
Health Engine — Recovery Score Engine
===========================================================

Purpose:
--------
Generate scientifically weighted recovery scores
using wearable physiological metrics.

This module:
- calculates recovery score
- classifies recovery state
- generates workout recommendations
- creates final ML target

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


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/processed/physiological_cleaned.csv"

OUTPUT_PATH = "data/processed"

OUTPUT_FILE = "recovery_engineered.csv"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    print("\n Loading cleaned physiological dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f" Dataset Shape: {df.shape}")

    return df


# =========================================================
# NORMALIZE TO 100 SCALE
# =========================================================

def normalize_100(series):

    min_val = series.min()
    max_val = series.max()

    return (
        (series - min_val) /
        (max_val - min_val + 1e-8)
    ) * 100


# =========================================================
# RECOVERY SCORE CALCULATION
# =========================================================

def calculate_recovery_score(df):

    print("\n Calculating recovery scores...")

    # -----------------------------------------------------
    # CHECK AVAILABLE FEATURES
    # -----------------------------------------------------

    available_features = []

    possible_features = [
        "sleep_performance_",
        "heart_rate_variability_ms",
        "resting_heart_rate_bpm",
        "day_strain",
        "blood_oxygen_"
    ]

    for feature in possible_features:

        if feature in df.columns:
            available_features.append(feature)

    print("\n Available recovery features:")
    print(available_features)

    # -----------------------------------------------------
    # DYNAMIC FEATURE SCORING
    # -----------------------------------------------------

    sleep_score = (

        normalize_100(
            df["sleep_performance_"]
        )

        if "sleep_performance_" in df.columns

        else 50
    )

    hrv_score = (

        normalize_100(
            df["heart_rate_variability_ms"]
        )

        if "heart_rate_variability_ms" in df.columns

        else 50
    )

    strain_score = (

        normalize_100(
            df["day_strain"]
        )

        if "day_strain" in df.columns

        else 50
    )

    # -----------------------------------------------------
    # RHR INVERSE SCORING
    # -----------------------------------------------------

    if "resting_heart_rate_bpm" in df.columns:

        rhr_norm = normalize_100(
            df["resting_heart_rate_bpm"]
        )

        rhr_score = 100 - rhr_norm

    else:

        rhr_score = 50

    # -----------------------------------------------------
    # OPTIONAL SPO2 BONUS
    # -----------------------------------------------------

    if "blood_oxygen_" in df.columns:

        spo2_score = normalize_100(
            df["blood_oxygen_"]
        )

    else:

        spo2_score = 0

    # -----------------------------------------------------
    # FINAL RECOVERY FORMULA
    # -----------------------------------------------------

    df["recovery_score"] = (

        (0.40 * sleep_score) +

        (0.30 * hrv_score) +

        (0.20 * rhr_score) -

        (0.10 * strain_score) +

        (0.05 * spo2_score)

    )

    # -----------------------------------------------------
    # CLIP TO VALID RANGE
    # -----------------------------------------------------

    df["recovery_score"] = np.clip(
        df["recovery_score"],
        0,
        100
    )

    print("\n Recovery scores generated successfully.")

    return df


# =========================================================
# RECOVERY CLASSIFICATION
# =========================================================

def classify_recovery(df):

    print("\n Classifying recovery states...")

    conditions = [

        (df["recovery_score"] < 40),

        (
            (df["recovery_score"] >= 40) &
            (df["recovery_score"] < 60)
        ),

        (
            (df["recovery_score"] >= 60) &
            (df["recovery_score"] < 80)
        ),

        (df["recovery_score"] >= 80)

    ]

    labels = [

        "Poor Recovery",
        "Moderate Recovery",
        "Good Recovery",
        "Excellent Recovery"
    ]

    df["recovery_status"] = np.select(
        conditions,
        labels,
        default="Unknown"
    )

    return df


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

def generate_recommendations(df):

    print("\n Generating recommendations...")

    recommendations = []

    for score in df["recovery_score"]:

        if score < 40:

            recommendations.append(
                "Complete rest or light walking recommended"
            )

        elif score < 60:

            recommendations.append(
                "Light workout and recovery focus recommended"
            )

        elif score < 80:

            recommendations.append(
                "Moderate intensity workout recommended"
            )

        else:

            recommendations.append(
                "High intensity training readiness detected"
            )

    df["recommendation"] = recommendations

    return df


# =========================================================
# READINESS SCORE
# =========================================================

def calculate_readiness(df):

    print("\n Calculating readiness score...")

    if "hrv_7d_avg" in df.columns:

        readiness_component = normalize_100(
            df["hrv_7d_avg"]
        )

    else:

        readiness_component = 50

    df["readiness_score"] = (

        (df["recovery_score"] * 0.7) +

        (readiness_component * 0.3)

    )

    df["readiness_score"] = np.clip(
        df["readiness_score"],
        0,
        100
    )

    return df


# =========================================================
# FINAL SUMMARY
# =========================================================

def final_summary(df):

    print("\n FINAL RECOVERY ENGINE SUMMARY")
    print("=" * 50)

    print(f"\n Final Shape: {df.shape}")

    print("\n Recovery Status Distribution:")

    print(
        df["recovery_status"]
        .value_counts()
    )

    print("\n Average Recovery Score:")

    print(
        round(
            df["recovery_score"].mean(),
            2
        )
    )


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

    print("\n Recovery dataset saved:")
    print(output_path)


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n================================================")
    print(" HEALTH ENGINE — RECOVERY ENGINE")
    print("================================================")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_dataset()

    # -----------------------------------------------------
    # RECOVERY SCORE
    # -----------------------------------------------------

    df = calculate_recovery_score(df)

    # -----------------------------------------------------
    # RECOVERY CLASSIFICATION
    # -----------------------------------------------------

    df = classify_recovery(df)

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    df = generate_recommendations(df)

    # -----------------------------------------------------
    # READINESS SCORE
    # -----------------------------------------------------

    df = calculate_readiness(df)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    final_summary(df)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_dataset(df)

    print("\n RECOVERY ENGINE COMPLETE")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
