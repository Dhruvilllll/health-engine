import pandas as pd
from baseline import BaselineCalibrator
from feature_engineering import FeatureEngineer
from strain_engine import StrainEngine

# Load datasets
df1 = pd.read_csv("health_dataset_realistic1.csv")
df2 = pd.read_csv("health_dataset_realistic2.csv")
df3 = pd.read_csv("health_dataset_realistic3.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)

# Baseline calibration
calibrator = BaselineCalibrator()
calibrator.compute_resting_baseline(df)
calibrator.estimate_max_hr(df)

# Feature engineering
engineer = FeatureEngineer(calibrator)
df_features = engineer.build_features(df)

# Strain engine
strain_engine = StrainEngine()
df_result = strain_engine.evaluate_dataframe(df_features)

print(df_result[[
    "HeartRate_BPM",
    "hr_reserve_load",
    "strain_score",
    "strain_level",
    "overexertion_risk"
]].head(20))