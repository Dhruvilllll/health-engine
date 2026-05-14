"""
===========================================================
Health Engine — Recovery Model Training
===========================================================

Purpose:
--------
Train Random Forest model for recovery prediction.

This pipeline:
- loads recovery dataset
- selects features
- splits train/test data
- trains Random Forest Regressor
- evaluates performance
- saves trained model

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
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/processed/recovery_engineered.csv"

MODEL_OUTPUT_PATH = "models/trained"

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    print("\n Loading recovery dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f" Dataset Shape: {df.shape}")

    return df


# =========================================================
# FEATURE SELECTION
# =========================================================

def select_features(df):

    print("\n Selecting model features...")

    possible_features = [

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

    print("\n Selected Features:")
    print(features)

    X = df[features]

    y = df["recovery_score"]

    return X, y, features


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

def split_data(X, y):

    print("\n Splitting train/test data...")

    return train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42

    )


# =========================================================
# MODEL TRAINING
# =========================================================

def train_model(X_train, y_train):

    print("\n Training Random Forest Regressor...")

    model = RandomForestRegressor(

        n_estimators=300,

        max_depth=12,

        min_samples_split=5,

        min_samples_leaf=2,

        random_state=42

    )

    model.fit(X_train, y_train)

    print("\n Model training complete.")

    return model


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(model, X_test, y_test):

    print("\n Evaluating model...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n MODEL PERFORMANCE")
    print("=" * 40)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")

    return predictions


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def feature_importance(model, features):

    print("\n Feature Importance")
    print("=" * 40)

    importance_df = pd.DataFrame({

        "feature": features,

        "importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print(importance_df)

    return importance_df


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):

    model_path = os.path.join(

        MODEL_OUTPUT_PATH,

        "recovery_random_forest.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\n Model saved:")
    print(model_path)


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n================================================")
    print(" HEALTH ENGINE — MODEL TRAINING")
    print("================================================")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_dataset()

    # -----------------------------------------------------
    # FEATURE SELECTION
    # -----------------------------------------------------

    X, y, features = select_features(df)

    # -----------------------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # EVALUATE MODEL
    # -----------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    feature_importance(
        model,
        features
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    save_model(model)

    print("\n MODEL TRAINING PIPELINE COMPLETE")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
