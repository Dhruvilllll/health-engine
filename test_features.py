import pandas as pd
from baseline import BaselineCalibrator
from feature_engineering import FeatureEngineer

# Load datasets
df1 = pd.read_csv("health_dataset_realistic1.csv")
df2 = pd.read_csv("health_dataset_realistic2.csv")
df3 = pd.read_csv("health_dataset_realistic3.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)

# Calibrate baseline
calibrator = BaselineCalibrator()
calibrator.compute_resting_baseline(df)
calibrator.estimate_max_hr(df)

# Feature engineering
engineer = FeatureEngineer(calibrator)
df_features = engineer.build_features(df)

print(df_features.head())