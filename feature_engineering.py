# ==========================================
# Statmize Health Engine - Feature Engineering
# ==========================================

import numpy as np
import pandas as pd

import config
from baseline import BaselineCalibrator


class FeatureEngineer:
    """
    Generates physiological features for hybrid strain model.
    """

    def __init__(self, calibrator: BaselineCalibrator):
        self.calibrator = calibrator

    # --------------------------------------------------
    # 1️⃣ HR Reserve Load (Vectorized)
    # --------------------------------------------------
    def compute_hr_reserve_load(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes HR reserve ratio for entire dataframe.
        """

        resting_hr = self.calibrator.resting_hr
        max_hr = self.calibrator.estimated_max_hr

        denominator = max_hr - resting_hr
        if denominator <= 0:
            return pd.Series(np.zeros(len(df)))

        hr_reserve = (df["HeartRate_BPM"] - resting_hr) / denominator
        hr_reserve = hr_reserve.clip(0, 1)

        return hr_reserve

    # --------------------------------------------------
    # 2️⃣ SpO2 Drop
    # --------------------------------------------------
    def compute_spo2_drop(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes oxygen saturation drop from resting baseline.
        """

        baseline_spo2 = self.calibrator.resting_spo2

        spo2_drop = baseline_spo2 - df["SpO2_Percent"]
        spo2_drop = spo2_drop.clip(lower=0)

        return spo2_drop

    # --------------------------------------------------
    # 3️⃣ HRV Suppression (Optional / Future Ready)
    # --------------------------------------------------
    def compute_hrv_suppression(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes HRV suppression relative to baseline.
        If HRV column not present, returns zeros.
        """

        if "HRV" not in df.columns or self.calibrator.resting_hrv is None:
            return pd.Series(np.zeros(len(df)))

        baseline_hrv = self.calibrator.resting_hrv
        hrv_drop = (baseline_hrv - df["HRV"]) / baseline_hrv
        hrv_drop = hrv_drop.clip(lower=0)

        return hrv_drop

    # --------------------------------------------------
    # 4️⃣ Recovery Slope
    # --------------------------------------------------
    def compute_recovery_slope(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes recovery slope using rolling window.
        Measures how quickly HR drops.
        """

        window_size = int(config.RECOVERY_ANALYSIS_WINDOW / 10)  # assuming 10s step

        hr_diff = df["HeartRate_BPM"].diff()

        recovery_slope = (
            hr_diff.rolling(window=window_size, min_periods=1)
            .mean()
            .abs()
        )

        return recovery_slope

    # --------------------------------------------------
    # 5️⃣ Rolling HR Mean
    # --------------------------------------------------
    def compute_rolling_hr_mean(self, df: pd.DataFrame) -> pd.Series:
        """
        Smooth HR to detect sustained effort.
        """

        window_size = int(config.ROLLING_WINDOW_SECONDS / 10)

        rolling_hr = (
            df["HeartRate_BPM"]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )

        return rolling_hr

    # --------------------------------------------------
    # 6️⃣ Master Feature Builder
    # --------------------------------------------------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds all engineered features.
        """

        df = df.copy()

        df["hr_reserve_load"] = self.compute_hr_reserve_load(df)
        df["spo2_drop"] = self.compute_spo2_drop(df)
        df["hrv_suppression"] = self.compute_hrv_suppression(df)
        df["recovery_slope"] = self.compute_recovery_slope(df)
        df["rolling_hr_mean"] = self.compute_rolling_hr_mean(df)

        return df