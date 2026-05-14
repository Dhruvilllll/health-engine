"""
===================================================
Health Engine - Data Transformer
===================================================

Transform raw input data to engineered features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.common.utils import handle_missing_values
from src.common.logger import setup_logger


logger = setup_logger(__name__)


class DataTransformer:
    """Transform and engineer features from raw inputs."""

    # Feature mapping for flexibility
    FEATURE_MAPPING = {
        "hrv": ["hrv", "heart_rate_variability", "heart_rate_variability_ms"],
        "resting_heart_rate": ["resting_heart_rate", "rhr", "resting_heart_rate_bpm"],
        "day_strain": ["day_strain", "strain"],
        "spo2": ["spo2", "blood_oxygen", "blood_oxygen_"],
        "sleep_performance_percentage": [
            "sleep_performance_percentage",
            "sleep_performance_",
            "sleep_quality"
        ],
        "average_hr_bpm": [
            "average_hr_bpm",
            "avg_hr",
            "average_heart_rate"
        ],
        "asleep_duration_min": [
            "asleep_duration_min",
            "sleep_duration",
            "total_sleep"
        ]
    }

    def __init__(self):
        """Initialize transformer."""
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit transformer and transform data."""
        self.fitted = True
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data to engineered features."""
        df = df.copy()

        # Standardize column names
        df = self._standardize_column_names(df)

        # Handle missing values
        df = handle_missing_values(df, method="median")

        # Engineer features
        df = self._engineer_features(df)

        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names by mapping variants to canonical names."""
        for canonical_name, aliases in self.FEATURE_MAPPING.items():
            for alias in aliases:
                if alias in df.columns and canonical_name not in df.columns:
                    df.rename(columns={alias: canonical_name}, inplace=True)
                    logger.debug(f"Mapped {alias} → {canonical_name}")

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        # Stress Ratio: RHR / HRV
        if "resting_heart_rate" in df.columns and "hrv" in df.columns:
            df["stress_ratio"] = df["resting_heart_rate"] / (df["hrv"] + 1e-5)
            logger.debug("Created stress_ratio feature")

        # HRV Strain Balance
        if "hrv" in df.columns and "day_strain" in df.columns:
            df["hrv_strain_balance"] = df["hrv"] / (df["day_strain"] + 1)
            logger.debug("Created hrv_strain_balance feature")

        # Sleep Deficit
        if "sleep_performance_percentage" in df.columns:
            df["sleep_deficit"] = 100 - df["sleep_performance_percentage"]
            logger.debug("Created sleep_deficit feature")

        # 7-day rolling average (for single records, use the value itself)
        if "hrv" in df.columns:
            if len(df) > 1:
                df["hrv_7d_avg"] = df["hrv"].rolling(window=7, min_periods=1).mean()
            else:
                df["hrv_7d_avg"] = df["hrv"]
            logger.debug("Created hrv_7d_avg feature")

        if "day_strain" in df.columns:
            if len(df) > 1:
                df["strain_7d_avg"] = df["day_strain"].rolling(window=7, min_periods=1).mean()
            else:
                df["strain_7d_avg"] = df["day_strain"]
            logger.debug("Created strain_7d_avg feature")

        return df

    def get_feature_info(self) -> Dict:
        """Get information about features."""
        return {
            "primary_features": [f for f in self.FEATURE_MAPPING.keys()],
            "engineered_features": [
                "stress_ratio",
                "hrv_strain_balance",
                "sleep_deficit",
                "hrv_7d_avg",
                "strain_7d_avg"
            ]
        }
