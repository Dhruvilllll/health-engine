# ==========================================
# Statmize Health Engine - Baseline Module
# ==========================================

import numpy as np
import pandas as pd
from typing import Dict

import config


class BaselineCalibrator:
    """
    Calibrates physiological baselines for a player
    using data-driven estimation (no age assumption).
    """

    def __init__(self):
        self.resting_hr = None
        self.resting_spo2 = None
        self.resting_hrv = None
        self.estimated_max_hr = None

    # --------------------------------------------------
    # 1️⃣ Detect Resting Baseline
    # --------------------------------------------------
    def compute_resting_baseline(self, df: pd.DataFrame) -> None:
        """
        Computes resting HR, SpO2, and HRV baseline
        from rows labeled as REST_LABEL.
        """

        rest_df = df[df["Activity_Label"] == config.REST_LABEL]

        if rest_df.empty:
            raise ValueError("No resting data found for baseline calibration.")

        self.resting_hr = rest_df["HeartRate_BPM"].mean()
        self.resting_spo2 = rest_df["SpO2_Percent"].mean()

        # HRV optional (future-ready)
        if "HRV" in df.columns:
            self.resting_hrv = rest_df["HRV"].mean()

    # --------------------------------------------------
    # 2️⃣ Estimate Max HR (Data-Driven)
    # --------------------------------------------------
    def estimate_max_hr(self, df: pd.DataFrame) -> None:
        """
        Estimates max HR using percentile from high-intensity periods.
        """

        high_intensity_df = df[
            df["Activity_Label"].isin(config.HIGH_INTENSITY_LABELS)
        ]

        if high_intensity_df.empty:
            raise ValueError("No high-intensity data found for max HR estimation.")

        self.estimated_max_hr = np.percentile(
            high_intensity_df["HeartRate_BPM"],
            config.MAX_HR_PERCENTILE
        )

    # --------------------------------------------------
    # 3️⃣ HR Reserve Calculation
    # --------------------------------------------------
    def compute_hr_reserve_ratio(self, current_hr: float) -> float:
        """
        Computes heart rate reserve ratio.
        """

        if self.resting_hr is None or self.estimated_max_hr is None:
            raise ValueError("Baseline not calibrated.")

        denominator = self.estimated_max_hr - self.resting_hr

        if denominator <= 0:
            return 0.0

        hr_reserve = (current_hr - self.resting_hr) / denominator

        # Clamp between 0 and 1
        return float(np.clip(hr_reserve, 0.0, 1.0))

    # --------------------------------------------------
    # 4️⃣ Export Baseline Values
    # --------------------------------------------------
    def get_baseline_dict(self) -> Dict:
        """
        Returns calibrated physiological baseline.
        """

        return {
            "resting_hr": self.resting_hr,
            "resting_spo2": self.resting_spo2,
            "resting_hrv": self.resting_hrv,
            "estimated_max_hr": self.estimated_max_hr
        }