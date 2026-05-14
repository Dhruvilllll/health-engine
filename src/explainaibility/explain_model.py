"""
===========================================================
Health Engine — Model Explainability
===========================================================

Purpose:
--------
Generate explainability insights for recovery prediction.

This module:
- loads trained model
- computes SHAP values
- visualizes feature importance
- creates recovery trend charts
- explains physiological drivers

Author:
-------
Dhruvil Malvania

===========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import joblib
import shap

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px


# =========================================================
# CONFIGURATION
# =========================================================

DATA_FILE = "data/processed/recovery_engineered.csv"

MODEL_FILE = (
    "models/trained/recovery_random_forest.pkl"
)

OUTPUT_PATH = "reports/figures"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\n Loading recovery dataset...")

    df = pd.read_csv(DATA_FILE)

    print(f" Dataset Shape: {df.shape}")

    return df


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    print("\n Loading trained model...")

    model = joblib.load(MODEL_FILE)

    print("\n Model loaded successfully.")

    return model


# =========================================================
# SELECT FEATURES
# =========================================================

def select_features(df):

    possible_features = [
        # Map to actual column names in the data
        "heart_rate_variability_ms",
        "resting_heart_rate_bpm",
        "day_strain",
        "blood_oxygen_",
        "stress_ratio",
        "hrv_strain_balance",
        "hrv_7d_avg",
        "strain_7d_avg"
    ]

    features = [
        feature for feature in possible_features
        if feature in df.columns
    ]

    X = df[features]

    return X, features


# =========================================================
# FEATURE IMPORTANCE PLOT
# =========================================================

def feature_importance_plot(model, features):

    print("\n Generating feature importance plot...")

    importance_df = pd.DataFrame({

        "feature": features,

        "importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        "Recovery Prediction Feature Importance"
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_PATH,
        "feature_importance.png"
    )

    plt.savefig(output_file)

    print(f"\n Saved: {output_file}")


# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

def shap_explainability(model, X):

    print("\n Computing SHAP explainability...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # -----------------------------------------------------
    # SUMMARY PLOT
    # -----------------------------------------------------

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    output_file = os.path.join(
        OUTPUT_PATH,
        "shap_summary.png"
    )

    plt.savefig(
        output_file,
        bbox_inches="tight"
    )

    print(f"\n Saved: {output_file}")


# =========================================================
# RECOVERY TREND VISUALIZATION
# =========================================================

def recovery_trend_plot(df):

    print("\n Generating recovery trend plot...")

    if "recovery_score" not in df.columns:

        print("\n recovery_score not found.")
        return

    fig = px.line(

        df,

        y="recovery_score",

        title="Recovery Score Trend"

    )

    output_file = os.path.join(
        OUTPUT_PATH,
        "recovery_trend.html"
    )

    fig.write_html(output_file)

    print(f"\n Saved: {output_file}")


# =========================================================
# HRV TREND VISUALIZATION
# =========================================================

def hrv_trend_plot(df):

    print("\n Generating HRV trend plot...")

    if "heart_rate_variability_ms" not in df.columns:

        print("\n HRV not found.")
        return

    fig = px.line(

        df,

        y="heart_rate_variability_ms",

        title="HRV Trend"

    )

    output_file = os.path.join(
        OUTPUT_PATH,
        "hrv_trend.html"
    )

    fig.write_html(output_file)

    print(f"\n Saved: {output_file}")


# =========================================================
# STRAIN VS RECOVERY
# =========================================================

def strain_vs_recovery(df):

    print("\n Generating strain vs recovery analysis...")

    if (
        "day_strain" not in df.columns or
        "recovery_score" not in df.columns
    ):

        print("\n Required columns missing.")
        return

    fig = px.scatter(

        df,

        x="day_strain",

        y="recovery_score",

        title="Strain vs Recovery"

    )

    output_file = os.path.join(
        OUTPUT_PATH,
        "strain_vs_recovery.html"
    )

    fig.write_html(output_file)

    print(f"\n Saved: {output_file}")


# =========================================================
# PHYSIOLOGICAL INSIGHTS
# =========================================================

def physiological_insights(df):

    print("\n Physiological Insights")
    print("=" * 50)

    if "heart_rate_variability_ms" in df.columns:
        print(
            f"\n Average HRV: "
            f"{round(df['heart_rate_variability_ms'].mean(), 2)}"
        )

    if "resting_heart_rate_bpm" in df.columns:
        print(
            f"\n Average Resting HR: "
            f"{round(df['resting_heart_rate_bpm'].mean(), 2)}"
        )

    if "recovery_score" in df.columns:
        print(
            f"\n Average Recovery Score: "
            f"{round(df['recovery_score'].mean(), 2)}"
        )

    if "day_strain" in df.columns:
        print(
            f"\n Average Day Strain: "
            f"{round(df['day_strain'].mean(), 2)}"
        )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n================================================")
    print(" HEALTH ENGINE — MODEL EXPLAINABILITY")
    print("================================================")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_data()

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = load_model()

    # -----------------------------------------------------
    # SELECT FEATURES
    # -----------------------------------------------------

    X, features = select_features(df)

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    feature_importance_plot(
        model,
        features
    )

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    shap_explainability(
        model,
        X
    )

    # -----------------------------------------------------
    # RECOVERY TREND
    # -----------------------------------------------------

    recovery_trend_plot(df)

    # -----------------------------------------------------
    # HRV TREND
    # -----------------------------------------------------

    hrv_trend_plot(df)

    # -----------------------------------------------------
    # STRAIN ANALYSIS
    # -----------------------------------------------------

    strain_vs_recovery(df)

    # -----------------------------------------------------
    # INSIGHTS
    # -----------------------------------------------------

    physiological_insights(df)

    print("\n EXPLAINABILITY PIPELINE COMPLETE")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()