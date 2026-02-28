# ==========================================
# Statmize Health Engine - Strain Engine
# ==========================================

import numpy as np
import pandas as pd

import config


class StrainEngine:
    """
    Computes final physiological strain score
    and health risk outputs.
    """

    def __init__(self):
        self.weights = config.STRAIN_WEIGHTS

    # --------------------------------------------------
    # 1️⃣ Compute Raw Strain Score (0–1 scale)
    # --------------------------------------------------
    def compute_raw_strain(self, df: pd.DataFrame) -> pd.Series:
        """
        Applies hybrid weighted strain formula.
        """

        raw_strain = (
            self.weights["hr_reserve_load"] * df["hr_reserve_load"] +
            self.weights["hrv_suppression"] * df["hrv_suppression"] +
            self.weights["spo2_drop"] * (df["spo2_drop"] / 5.0) +  # normalize approx
            self.weights["recovery_delay"] * (df["recovery_slope"] / 10.0)
        )

        # Ensure no negatives
        raw_strain = raw_strain.clip(lower=0)

        return raw_strain

    # --------------------------------------------------
    # 2️⃣ Scale Strain to 0–100
    # --------------------------------------------------
    def scale_strain(self, raw_strain: pd.Series) -> pd.Series:
        """
        Scales raw strain to 0–100.
        """

        scaled = raw_strain.clip(0, 1) * 100

        return scaled

    # --------------------------------------------------
    # 3️⃣ Assign Strain Level
    # --------------------------------------------------
    def assign_strain_level(self, strain_score: float) -> str:
        """
        Converts numeric strain into category.
        """

        thresholds = config.STRAIN_THRESHOLDS

        if strain_score >= thresholds["critical"]:
            return "Critical"
        elif strain_score >= thresholds["high"]:
            return "High"
        elif strain_score >= thresholds["moderate"]:
            return "Moderate"
        else:
            return "Low"

    # --------------------------------------------------
    # 4️⃣ Overexertion Detection
    # --------------------------------------------------
    def detect_overexertion(self, hr_reserve_value: float) -> str:
        """
        Detects overexertion based on HR reserve.
        """

        if hr_reserve_value >= config.OVEREXERTION_HR_RESERVE:
            return "High"
        elif hr_reserve_value >= 0.75:
            return "Medium"
        else:
            return "Low"

    # --------------------------------------------------
    # 5️⃣ Hydration Alert
    # --------------------------------------------------
    def hydration_alert(self, spo2_drop_value: float) -> bool:
        """
        Simple oxygen-based hydration warning.
        """

        return spo2_drop_value > 2.0

    # --------------------------------------------------
    # 6️⃣ Full Evaluation For One Row
    # --------------------------------------------------
    def evaluate_row(self, row: pd.Series) -> dict:
        """
        Generates full health intelligence output for one data point.
        """

        raw_strain = (
            self.weights["hr_reserve_load"] * row["hr_reserve_load"] +
            self.weights["hrv_suppression"] * row["hrv_suppression"] +
            self.weights["spo2_drop"] * (row["spo2_drop"] / 5.0) +
            self.weights["recovery_delay"] * (row["recovery_slope"] / 10.0)
        )

        raw_strain = max(0, min(raw_strain, 1))
        strain_score = raw_strain * 100

        return {
            "strain_score": round(strain_score, 2),
            "strain_level": self.assign_strain_level(strain_score),
            "overexertion_risk": self.detect_overexertion(row["hr_reserve_load"]),
            "hydration_alert": self.hydration_alert(row["spo2_drop"])
        }

    # --------------------------------------------------
    # 7️⃣ Evaluate Full Dataset
    # --------------------------------------------------
    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies strain evaluation to entire dataframe.
        """

        df = df.copy()

        raw_strain = self.compute_raw_strain(df)
        df["strain_score"] = self.scale_strain(raw_strain)

        df["strain_level"] = df["strain_score"].apply(self.assign_strain_level)
        df["overexertion_risk"] = df["hr_reserve_load"].apply(self.detect_overexertion)
        df["hydration_alert"] = df["spo2_drop"].apply(self.hydration_alert)

        return df