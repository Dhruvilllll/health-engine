# ==========================================
# Statmize Health Engine - Inference Module
# ==========================================

import os
import joblib
import numpy as np
import pandas as pd

from strain_engine import StrainEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class HealthInferenceEngine:
    """
    Loads trained artifacts and performs real-time prediction.
    """

    def __init__(self):
        self.model = joblib.load(os.path.join(BASE_DIR, "strain_model.pkl"))
        self.scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        self.baseline = joblib.load(os.path.join(BASE_DIR, "baseline.pkl"))
        self.strain_engine = StrainEngine()

    def compute_hr_reserve(self, current_hr):
        resting_hr = self.baseline["resting_hr"]
        max_hr = self.baseline["estimated_max_hr"]
        denominator = max_hr - resting_hr
        if denominator <= 0:
            return 0.0
        hr_reserve = (current_hr - resting_hr) / denominator
        return float(np.clip(hr_reserve, 0.0, 1.0))

    def compute_spo2_drop(self, current_spo2):
        baseline_spo2 = self.baseline["resting_spo2"]
        drop = baseline_spo2 - current_spo2
        return max(0.0, drop)

    def build_feature_vector(self, hr, spo2, hrv=None):
        hr_reserve = self.compute_hr_reserve(hr)
        spo2_drop = self.compute_spo2_drop(spo2)

        if hrv is not None and self.baseline.get("resting_hrv"):
            baseline_hrv = self.baseline["resting_hrv"]
            hrv_suppression = max(0.0, (baseline_hrv - hrv) / baseline_hrv)
        else:
            hrv_suppression = 0.0

        feature_dict = {
            "hr_reserve_load": hr_reserve,
            "spo2_drop": spo2_drop,
            "hrv_suppression": hrv_suppression,
            "recovery_slope": 0.0,
            "rolling_hr_mean": hr
        }

        features_df = pd.DataFrame([feature_dict])

        if self.scaler:
            features_scaled = self.scaler.transform(features_df)
        else:
            features_scaled = features_df.values

        return features_scaled, hr_reserve, spo2_drop

    def predict(self, hr, spo2, hrv=None):
        features, hr_reserve, spo2_drop = self.build_feature_vector(hr, spo2, hrv)

        predicted_level = self.model.predict(features)[0]

        if hasattr(self.model, "predict_proba"):
            probability = np.max(self.model.predict_proba(features))
        else:
            probability = None

        strain_dict = self.strain_engine.evaluate_row({
            "hr_reserve_load": hr_reserve,
            "hrv_suppression": 0.0,
            "spo2_drop": spo2_drop,
            "recovery_slope": 0.0
        })

        return {
            "predicted_strain_level": predicted_level,
            "model_confidence": round(float(probability), 3) if probability else None,
            "strain_score": strain_dict["strain_score"],
            "overexertion_risk": strain_dict["overexertion_risk"],
            "hydration_alert": strain_dict["hydration_alert"]
        }