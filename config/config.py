"""
===================================================
Health Engine - Configuration Management
===================================================

Centralized configuration for all modules.
Uses environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import List


class Config:
    """Base configuration class."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models" / "trained"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Data files
    PHYSIOLOGICAL_MERGED = DATA_PROCESSED / "physiological_merged.csv"
    PHYSIOLOGICAL_CLEANED = DATA_PROCESSED / "physiological_cleaned.csv"
    RECOVERY_ENGINEERED = DATA_PROCESSED / "recovery_engineered.csv"
    SLEEP_MERGED = DATA_PROCESSED / "sleep_merged.csv"
    WORKOUT_MERGED = DATA_PROCESSED / "workout_merged.csv"

    # Model files
    MODEL_PATH = MODELS_DIR / "recovery_random_forest.pkl"
    SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_TITLE = "Health Engine API"
    API_DESCRIPTION = "Recovery & Readiness Prediction System"
    API_VERSION = "1.0.0"

    # Model Configuration
    MODEL_N_ESTIMATORS = 300
    MODEL_MAX_DEPTH = 12
    MODEL_MIN_SAMPLES_SPLIT = 5
    MODEL_MIN_SAMPLES_LEAF = 2
    MODEL_RANDOM_STATE = 42

    # Feature Configuration
    PRIMARY_FEATURES = [
        "day_strain",
        "resting_heart_rate",
        "hrv",
        "sleep_performance_percentage",
        "average_hr_bpm",
        "spo2"
    ]

    ENGINEERED_FEATURES = [
        "stress_ratio",
        "sleep_deficit",
        "hrv_strain_balance",
        "hrv_7d_avg",
        "strain_7d_avg"
    ]

    # Recovery thresholds
    RECOVERY_POOR_THRESHOLD = 40
    RECOVERY_MODERATE_THRESHOLD = 60
    RECOVERY_GOOD_THRESHOLD = 80

    RECOVERY_THRESHOLDS = {
        "poor": 40,
        "moderate": 60,
        "good": 80,
        "excellent": 100
    }

    # Physiological limits for outlier detection
    PHYSIOLOGICAL_LIMITS = {
        "hrv": (10, 250),
        "resting_heart_rate": (30, 120),
        "spo2": (70, 100),
        "respiratory_rate": (5, 40),
        "day_strain": (0, 25),
    }

    # Recovery formula weights
    RECOVERY_WEIGHTS = {
        "sleep": 0.40,
        "hrv": 0.30,
        "rhr": 0.20,
        "strain": -0.10,
        "spo2": 0.05
    }

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "health_engine.log"

    # Database (future use)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///health_engine.db")

    # Environment
    ENV = os.getenv("ENV", "development")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        dirs = [
            cls.DATA_RAW,
            cls.DATA_PROCESSED,
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
            cls.LOGS_DIR
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    ENV = "development"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    ENV = "production"


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    ENV = "testing"
    DATABASE_URL = "sqlite:///:memory:"


def get_config():
    """Get configuration based on environment."""
    env = os.getenv("ENV", "development").lower()

    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Create global config instance
config = get_config()

# Ensure directories exist on import
config.ensure_directories()
