"""
===================================================
Health Engine - Shared Utilities
===================================================

Common functions used across all modules:
- Data standardization
- Normalization
- Model loading
- Feature validation
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ===================================================
# PATH UTILITIES
# ===================================================

def get_project_root() -> Path:
    """Get absolute path to project root."""
    return Path(__file__).parent.parent.parent


def get_data_path(filename: str, data_type: str = "processed") -> Path:
    """Get absolute path to data file."""
    root = get_project_root()
    return root / "data" / data_type / filename


def get_model_path(model_name: str = "recovery_random_forest.pkl") -> Path:
    """Get absolute path to model file."""
    root = get_project_root()
    return root / "models" / "trained" / model_name


# ===================================================
# COLUMN STANDARDIZATION
# ===================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across all datasets.

    - lowercase
    - remove spaces
    - replace hyphens with underscores
    - remove special characters
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    return df


# ===================================================
# NORMALIZATION UTILITIES
# ===================================================

def normalize_100(series: pd.Series, name: str = None) -> pd.Series:
    """
    Normalize series to 0-100 scale.

    Formula: (x - min) / (max - min) * 100
    """
    min_val = series.min()
    max_val = series.max()

    normalized = (
        (series - min_val) /
        (max_val - min_val + 1e-8)
    ) * 100

    if name:
        normalized.name = name

    return normalized


def minmax_scale(series: pd.Series) -> Tuple[pd.Series, float, float]:
    """
    Min-max normalize to 0-1 scale.

    Returns: normalized series, min value, max value
    """
    min_val = series.min()
    max_val = series.max()

    normalized = (series - min_val) / (max_val - min_val + 1e-8)

    return normalized, min_val, max_val


# ===================================================
# MODEL UTILITIES
# ===================================================

def load_model(model_path: Path):
    """Load trained model from pickle file."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return joblib.load(model_path)


def save_model(model, model_path: Path):
    """Save model to pickle file."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


# ===================================================
# DATA VALIDATION
# ===================================================

def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Check if DataFrame has all required columns."""
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return True


def validate_numeric_columns(df: pd.DataFrame, columns: List[str]):
    """Ensure columns are numeric."""
    for col in columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as e:
                raise ValueError(f"Cannot convert {col} to numeric: {e}")

    return df


def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """Remove duplicate rows."""
    initial_shape = df.shape[0]
    df_clean = df.drop_duplicates(subset=subset)
    removed = initial_shape - df_clean.shape[0]

    if removed > 0:
        print(f"Removed {removed} duplicate rows")

    return df_clean


# ===================================================
# FEATURE UTILITIES
# ===================================================

def select_available_features(
    df: pd.DataFrame,
    feature_list: List[str]
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select only features that exist in DataFrame.

    Returns: list of available features, DataFrame with only those features
    """
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]

    if missing:
        print(f"Warning: Missing features {missing}")

    return available, df[available]


def get_feature_stats(df: pd.DataFrame, features: List[str]) -> Dict:
    """Get summary statistics for features."""
    return {
        f: {
            "mean": df[f].mean(),
            "std": df[f].std(),
            "min": df[f].min(),
            "max": df[f].max(),
            "missing": df[f].isna().sum()
        }
        for f in features if f in df.columns
    }


# ===================================================
# MISSING VALUE HANDLING
# ===================================================

def handle_missing_values(
    df: pd.DataFrame,
    method: str = "median"
) -> pd.DataFrame:
    """
    Handle missing values in numeric columns.

    Args:
        df: DataFrame
        method: "median", "mean", or "drop"

    Returns: DataFrame with missing values handled
    """
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            if method == "median":
                fill_value = df[col].median()
            elif method == "mean":
                fill_value = df[col].mean()
            else:
                continue

            df[col] = df[col].fillna(fill_value)

    return df


# ===================================================
# OUTLIER DETECTION
# ===================================================

def remove_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    before = df.shape[0]
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    after = df.shape[0]

    if before - after > 0:
        print(f"Removed {before - after} outliers from {column}")

    return df


def check_physiological_ranges(df: pd.DataFrame) -> Dict:
    """
    Check if physiological values are within reasonable ranges.

    Returns: Dictionary of valid/invalid status for each metric
    """
    ranges = {
        "hrv": (10, 250),
        "resting_heart_rate": (30, 120),
        "spo2": (70, 100),
        "respiratory_rate": (5, 40),
        "day_strain": (0, 25),
    }

    status = {}
    for metric, (lower, upper) in ranges.items():
        if metric in df.columns:
            valid = df[(df[metric] >= lower) & (df[metric] <= upper)].shape[0]
            total = df.shape[0]
            status[metric] = {
                "valid": valid,
                "invalid": total - valid,
                "percentage": round(100 * valid / total, 2)
            }

    return status


# ===================================================
# SUMMARY STATISTICS
# ===================================================

def print_data_summary(df: pd.DataFrame, title: str = "Data Summary"):
    """Print summary of DataFrame."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Stats:\n{df.describe()}")
