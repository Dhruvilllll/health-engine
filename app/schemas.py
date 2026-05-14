"""
===================================================
Health Engine - API Request/Response Schemas
===================================================

Pydantic models for request validation and response formatting.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


class RecoveryStatusEnum(str, Enum):
    """Recovery status values."""
    POOR = "Poor Recovery"
    MODERATE = "Moderate Recovery"
    GOOD = "Good Recovery"
    EXCELLENT = "Excellent Recovery"


class ConfidenceEnum(str, Enum):
    """Confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ===================================================
# Request Models
# ===================================================

class PhysiologicalInput(BaseModel):
    """Input data for physiological metrics."""

    # Primary metrics - updated to match training data
    heart_rate_variability_ms: Optional[float] = Field(None, description="Heart Rate Variability (ms)")
    resting_heart_rate_bpm: Optional[float] = Field(None, description="Resting HR (bpm)")
    day_strain: Optional[float] = Field(None, description="Daily strain score")
    blood_oxygen_: Optional[float] = Field(None, description="Blood oxygen saturation (%)")
    sleep_performance_: Optional[float] = Field(None, description="Sleep quality percentage")
    average_hr_bpm: Optional[float] = Field(None, description="Average heart rate (bpm)")

    # Sleep metrics
    asleep_duration_min: Optional[float] = Field(None, description="Total sleep duration (min)")
    deep_sws_duration_min: Optional[float] = Field(None, description="Deep sleep (min)")
    rem_duration_min: Optional[float] = Field(None, description="REM sleep (min)")
    light_sleep_duration_min: Optional[float] = Field(None, description="Light sleep (min)")

    # Additional metrics
    respiratory_rate_rpm: Optional[float] = Field(None, description="Respiratory rate (rpm)")
    skin_temp_celsius: Optional[float] = Field(None, description="Skin temperature (°C)")
    max_hr_bpm: Optional[float] = Field(None, description="Max heart rate (bpm)")

    # Engineered features
    stress_ratio: Optional[float] = Field(None, description="RHR / HRV ratio")
    hrv_strain_balance: Optional[float] = Field(None, description="HRV / Strain balance")
    hrv_7d_avg: Optional[float] = Field(None, description="7-day average HRV")
    strain_7d_avg: Optional[float] = Field(None, description="7-day average strain")

    class Config:
        json_schema_extra = {
            "example": {
                "heart_rate_variability_ms": 60.0,
                "resting_heart_rate_bpm": 62.0,
                "day_strain": 5.2,
                "blood_oxygen_": 97.0,
                "sleep_performance_": 87.0,
                "average_hr_bpm": 75.0
            }
        }


class PredictionOutput(BaseModel):
    """Output of prediction."""

    predicted_score: float = Field(..., description="Raw prediction score")
    recovery_score: float = Field(..., description="Recovery score (0-100)")
    recovery_status: RecoveryStatusEnum = Field(..., description="Recovery classification")
    recommendation: str = Field(..., description="Workout recommendation")
    confidence: ConfidenceEnum = Field(..., description="Confidence level")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_score": 74.25,
                "recovery_score": 74.25,
                "recovery_status": "Good Recovery",
                "recommendation": "Moderate intensity workout recommended",
                "confidence": "high"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    records: List[PhysiologicalInput] = Field(..., description="List of input records")
    save_output: bool = Field(False, description="Save results to file")

    class Config:
        json_schema_extra = {
            "example": {
                "records": [
                    {
                        "heart_rate_variability_ms": 60.0,
                        "resting_heart_rate_bpm": 62.0,
                        "day_strain": 5.2,
                        "blood_oxygen_": 97.0,
                        "sleep_performance_": 87.0,
                        "average_hr_bpm": 75.0
                    },
                    {
                        "heart_rate_variability_ms": 55.0,
                        "resting_heart_rate_bpm": 65.0,
                        "day_strain": 7.5,
                        "blood_oxygen_": 96.5,
                        "sleep_performance_": 82.0,
                        "average_hr_bpm": 78.0
                    }
                ],
                "save_output": False
            }
        }


# ===================================================
# Response Models
# ===================================================

class BatchPredictionRecord(BaseModel):
    """Single record in batch prediction response."""

    original_data: Dict
    predictions: PredictionOutput


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction."""

    total_records: int
    successful: int
    failed: int
    success_rate: float
    results: List[Dict]
    errors: Optional[List[Dict]] = None


class ModelInfo(BaseModel):
    """Information about the model."""

    model_type: str = "Random Forest Regressor"
    n_estimators: int = 300
    max_depth: int = 12
    training_features: List[str]
    recovery_weights: Dict[str, float]
    thresholds: Dict[str, float]


class FeatureInfo(BaseModel):
    """Information about features."""

    primary_features: List[str]
    engineered_features: List[str]
    total_features: int


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    model_loaded: bool
    message: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    status_code: int
