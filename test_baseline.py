import pandas as pd
from baseline import BaselineCalibrator

# -----------------------------------
# 1️⃣ Load All Health Datasets
# -----------------------------------

df1 = pd.read_csv("health_dataset_realistic1.csv")
df2 = pd.read_csv("health_dataset_realistic2.csv")
df3 = pd.read_csv("health_dataset_realistic3.csv")

# Combine into single dataframe
df = pd.concat([df1, df2, df3], ignore_index=True)

print("Total rows loaded:", len(df))


# -----------------------------------
# 2️⃣ Initialize Baseline Calibrator
# -----------------------------------

calibrator = BaselineCalibrator()

calibrator.compute_resting_baseline(df)
calibrator.estimate_max_hr(df)

baseline_values = calibrator.get_baseline_dict()


# -----------------------------------
# 3️⃣ Print Baseline Results
# -----------------------------------

print("\n===== Baseline Calibration Results =====")
print("Resting HR:", round(baseline_values["resting_hr"], 2))
print("Resting SpO2:", round(baseline_values["resting_spo2"], 2))
print("Estimated Max HR:", round(baseline_values["estimated_max_hr"], 2))

# Example HR reserve calculation
example_hr = 150
reserve = calibrator.compute_hr_reserve_ratio(example_hr)

print(f"\nHR Reserve Ratio for HR={example_hr}: {round(reserve, 3)}")