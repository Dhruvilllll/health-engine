"""
===================================================
Health Engine - Prediction Service
===================================================

Service for making predictions on new data.
Handles model loading, data validation, and inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib

from config.config import config
from src.common.utils import (
    load_model,
    validate_required_columns,
    normalize_100,
    minmax_scale,
    select_available_features,
    handle_missing_values
)
from src.common.logger import setup_logger


logger = setup_logger(__name__)


class PredictionService:
    """Service for recovery predictions."""

    def __init__(self, model_path: Path = None):
        """
        Initialize prediction service.

        Args:
            model_path: Path to trained model pickle file
        """
        self.model_path = model_path or config.MODEL_PATH
        self.model = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Load trained model from disk."""
        try:
            self.model = load_model(self.model_path)
            trained_features = getattr(self.model, "feature_names_in_", None)
            if trained_features is not None:
                self.feature_names = list(trained_features)
            logger.info(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model not found at {self.model_path}")
            raise

    def predict(
        self,
        input_data: Dict
    ) -> Dict:
        """
        Make prediction on single record.

        Args:
            input_data: Dictionary with physiological metrics

        Returns:
            Dictionary with prediction, recovery status, and recommendation
        """
        try:
            # Create DataFrame from input
            df = pd.DataFrame([input_data])

            # Validate and prepare
            df_prepared = self._prepare_data(df)

            # Get features
            features = self._get_features(df_prepared)

            # Make prediction
            prediction = self.model.predict(features)[0]

            # Get recovery status and recommendation
            status = self._classify_recovery(prediction)
            recommendation = self._get_recommendation(prediction)

            return {
                "predicted_score": float(round(prediction, 2)),
                "recovery_score": float(round(prediction, 2)),
                "recovery_status": status,
                "recommendation": recommendation,
                "confidence": self._estimate_confidence(prediction)
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def predict_batch(
        self,
        input_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions on multiple records.

        Args:
            input_data: DataFrame with multiple records

        Returns:
            DataFrame with predictions added
        """
        try:
            # Validate and prepare
            df_prepared = self._prepare_data(input_data.copy())

            # Get features
            features = self._get_features(df_prepared)

            # Make predictions
            predictions = self.model.predict(features)

            # Add results to DataFrame
            df_prepared["predicted_recovery_score"] = predictions
            df_prepared["recovery_status"] = df_prepared["predicted_recovery_score"].apply(
                self._classify_recovery
            )
            df_prepared["recommendation"] = df_prepared["predicted_recovery_score"].apply(
                self._get_recommendation
            )

            logger.info(f"Batch prediction completed for {len(df_prepared)} records")
            return df_prepared

        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise ValueError(f"Batch prediction failed: {str(e)}")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate input data."""
        df = self._normalize_column_aliases(df)

        # Handle missing values
        df = handle_missing_values(df, method="median")

        # Apply feature engineering
        df = self._engineer_features(df)

        return df

    @staticmethod
    def _normalize_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
        """Accept older internal names and map them to API/training names."""
        aliases = {
            "hrv": "heart_rate_variability_ms",
            "resting_heart_rate": "resting_heart_rate_bpm",
            "spo2": "blood_oxygen_",
            "sleep_performance_percentage": "sleep_performance_",
            "respiratory_rate": "respiratory_rate_rpm",
        }

        for source, target in aliases.items():
            if source in df.columns and target not in df.columns:
                df[target] = df[source]

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw inputs."""
        # Stress ratio
        if "resting_heart_rate_bpm" in df.columns and "heart_rate_variability_ms" in df.columns:
            df["stress_ratio"] = df["resting_heart_rate_bpm"] / (df["heart_rate_variability_ms"] + 1e-5)
        elif "resting_heart_rate" in df.columns and "hrv" in df.columns:
            df["stress_ratio"] = df["resting_heart_rate"] / (df["hrv"] + 1e-5)

        # HRV Strain Balance
        if "heart_rate_variability_ms" in df.columns and "day_strain" in df.columns:
            df["hrv_strain_balance"] = df["heart_rate_variability_ms"] / (df["day_strain"] + 1)
        elif "hrv" in df.columns and "day_strain" in df.columns:
            df["hrv_strain_balance"] = df["hrv"] / (df["day_strain"] + 1)

        # Sleep Deficit
        if "sleep_performance_" in df.columns:
            df["sleep_deficit"] = 100 - df["sleep_performance_"]
        elif "sleep_performance_percentage" in df.columns:
            df["sleep_deficit"] = 100 - df["sleep_performance_percentage"]

        # Rolling averages (use single values if not available)
        if "heart_rate_variability_ms" in df.columns:
            if "hrv_7d_avg" not in df.columns:
                df["hrv_7d_avg"] = df["heart_rate_variability_ms"]
        elif "hrv" in df.columns:
            if "hrv_7d_avg" not in df.columns:
                df["hrv_7d_avg"] = df["hrv"]

        if "day_strain" in df.columns:
            if "strain_7d_avg" not in df.columns:
                df["strain_7d_avg"] = df["day_strain"]

        return df

    def _get_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract features needed for prediction."""
        trained_features = getattr(self.model, "feature_names_in_", None)
        if trained_features is not None:
            required_features = list(trained_features)
        else:
            required_features = [
                "heart_rate_variability_ms",
                "resting_heart_rate_bpm",
                "day_strain",
                "blood_oxygen_",
                "stress_ratio",
                "hrv_strain_balance",
                "hrv_7d_avg",
                "strain_7d_avg"
            ]

        missing_features = [f for f in required_features if f not in df.columns]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}. "
                f"Expected: {required_features}"
            )

        self.feature_names = required_features

        # Return only the selected features in the correct order
        return df[required_features]

    @staticmethod
    def _classify_recovery(score: float) -> str:
        """Classify recovery status from score."""
        if score < 40:
            return "Poor Recovery"
        elif score < 60:
            return "Moderate Recovery"
        elif score < 80:
            return "Good Recovery"
        else:
            return "Excellent Recovery"

    @staticmethod
    def _get_recommendation(score: float) -> str:
        """Get recommendation based on recovery score."""
        if score < 40:
            return "Complete rest or light walking recommended"
        elif score < 60:
            return "Light workout and recovery focus recommended"
        elif score < 80:
            return "Moderate intensity workout recommended"
        else:
            return "High intensity training readiness detected"

    @staticmethod
    def _estimate_confidence(score: float) -> str:
        """Estimate confidence level in prediction."""
        # Confidence is higher for extreme scores
        distance_from_middle = abs(score - 50)
        if distance_from_middle > 30:
            return "high"
        elif distance_from_middle > 15:
            return "medium"
        else:
            return "low"


def create_prediction_service() -> PredictionService:
    """Factory function to create prediction service."""
    return PredictionService()
