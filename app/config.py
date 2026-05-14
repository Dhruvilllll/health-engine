"""
===================================================
Health Engine - API Configuration
===================================================

Configuration specific to the API layer.
"""

from config.config import config


class APIConfig:
    """API configuration."""

    # Basic settings
    HOST = config.API_HOST
    PORT = config.API_PORT
    TITLE = config.API_TITLE
    DESCRIPTION = config.API_DESCRIPTION
    VERSION = config.API_VERSION

    # CORS
    CORS_ORIGINS = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:7860",
    ]

    # Request limits
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_BATCH_SIZE = 1000

    # Timeouts
    REQUEST_TIMEOUT = 30
    PREDICTION_TIMEOUT = 10

    # Recovery thresholds
    RECOVERY_THRESHOLDS = {
        "poor": 40,
        "moderate": 60,
        "good": 80,
        "excellent": 100
    }

    # Model parameters
    MODEL_PATH = config.MODEL_PATH
    FEATURES = config.PRIMARY_FEATURES + config.ENGINEERED_FEATURES

    # Recovery weights for documentation
    RECOVERY_WEIGHTS = config.RECOVERY_WEIGHTS


# Create API config instance
api_config = APIConfig()
