"""
===================================================
Health Engine - FastAPI Application
===================================================

Production-grade API for recovery predictions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io

from config.config import config
from app.config import api_config
from app.schemas import (
    PhysiologicalInput,
    PredictionOutput,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheck,
    ModelInfo,
    FeatureInfo,
    ErrorResponse
)
from app.middleware.error_handler import (
    health_engine_exception_handler,
    general_exception_handler,
    HealthEngineException,
    ValidationException,
    ModelException
)
from src.services.prediction_service import PredictionService
from src.services.batch_service import BatchService
from src.common.logger import setup_logger


# Initialize logger
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=api_config.TITLE,
    description=api_config.DESCRIPTION,
    version=api_config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(
    HealthEngineException,
    health_engine_exception_handler
)
app.add_exception_handler(
    Exception,
    general_exception_handler
)

# Initialize services
try:
    prediction_service = PredictionService()
    batch_service = BatchService(prediction_service)
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    prediction_service = None
    batch_service = None


# ===================================================
# HEALTH CHECK ENDPOINTS
# ===================================================

@app.get("/health", response_model=HealthCheck)
def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        version=api_config.VERSION,
        model_loaded=prediction_service is not None,
        message="API is running and ready for predictions"
    )


@app.get("/ready")
def readiness_check():
    """Readiness probe for Kubernetes."""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services not ready"
        )
    return {"status": "ready"}


# ===================================================
# PREDICTION ENDPOINTS
# ===================================================

@app.post("/api/v1/predict", response_model=PredictionOutput)
def predict(input_data: PhysiologicalInput):
    """
    Make recovery prediction for single record.

    Returns recovery score, status, and recommendations.
    """
    if prediction_service is None:
        raise ModelException("Model service not initialized")

    try:
        # Convert Pydantic model to dict, removing None values
        data_dict = input_data.dict(exclude_none=True)

        if not data_dict:
            raise ValidationException("No input data provided")

        # Make prediction
        result = prediction_service.predict(data_dict)

        return PredictionOutput(**result)

    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise ModelException(f"Prediction failed: {str(e)}")


@app.post("/api/v1/batch-predict")
def batch_predict(request: BatchPredictionRequest):
    """
    Make predictions for multiple records.

    Accepts list of physiological metrics and returns predictions.
    Handles errors gracefully - processes valid records even if some fail.
    """
    if batch_service is None:
        raise ModelException("Batch service not initialized")

    try:
        # Validate input
        if not request.records:
            raise ValidationException("No records provided")

        if len(request.records) > api_config.MAX_BATCH_SIZE:
            raise ValidationException(
                f"Too many records. Maximum {api_config.MAX_BATCH_SIZE} allowed"
            )

        # Convert Pydantic models to DataFrame
        data_list = [r.dict(exclude_none=True) for r in request.records]
        df = pd.DataFrame(data_list)

        # Process batch
        results_df, stats = batch_service._process_dataframe(df)

        return {
            "total_records": stats["total_records"],
            "successful": stats["successful"],
            "failed": stats["failed"],
            "success_rate": stats["success_rate"],
            "results": results_df.to_dict(orient="records"),
            "errors": stats["errors"] if stats["errors"] else None
        }

    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise ModelException(f"Batch prediction failed: {str(e)}")


@app.post("/api/v1/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload CSV file and get predictions.

    CSV should have columns matching physiological metrics.
    Returns predictions with original data.
    """
    if batch_service is None:
        raise ModelException("Batch service not initialized")

    try:
        # Validate file
        if not file.filename.endswith(".csv"):
            raise ValidationException("File must be CSV format")

        if file.size > api_config.MAX_UPLOAD_SIZE:
            raise ValidationException("File too large")

        # Read file
        contents = await file.read()
        results_df, stats = batch_service.process_csv_bytes(contents)

        return {
            "total_records": stats["total_records"],
            "successful": stats["successful"],
            "failed": stats["failed"],
            "success_rate": stats["success_rate"],
            "results": results_df.to_dict(orient="records"),
            "errors": stats["errors"] if stats["errors"] else None
        }

    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise ModelException(f"CSV processing failed: {str(e)}")


# ===================================================
# MODEL INFO ENDPOINTS
# ===================================================

@app.get("/api/v1/model-info", response_model=ModelInfo)
def get_model_info():
    """Get information about the trained model."""
    if prediction_service is None:
        raise ModelException("Model not loaded")

    return ModelInfo(
        model_type="Random Forest Regressor",
        n_estimators=config.MODEL_N_ESTIMATORS,
        max_depth=config.MODEL_MAX_DEPTH,
        training_features=prediction_service.feature_names or [],
        recovery_weights=config.RECOVERY_WEIGHTS,
        thresholds=config.RECOVERY_THRESHOLDS
    )


@app.get("/api/v1/features", response_model=FeatureInfo)
def get_features():
    """Get information about features."""
    primary = config.PRIMARY_FEATURES
    engineered = config.ENGINEERED_FEATURES

    return FeatureInfo(
        primary_features=primary,
        engineered_features=engineered,
        total_features=len(primary) + len(engineered)
    )


# ===================================================
# EXPLANABILITY ENDPOINTS
# ===================================================

@app.post("/api/v1/explain")
def explain_prediction(input_data: PhysiologicalInput):
    """
    Get SHAP explanation for prediction.

    Returns feature contribution to recovery score.
    """
    if prediction_service is None:
        raise ModelException("Model service not initialized")

    try:
        data_dict = input_data.dict(exclude_none=True)
        result = prediction_service.predict(data_dict)

        # TODO: Integrate SHAP for feature importance
        return {
            "prediction": result,
            "feature_explanations": "SHAP integration coming soon",
            "note": "Feature importance available via /docs"
        }

    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise ModelException(f"Explanation failed: {str(e)}")


# ===================================================
# ROOT ENDPOINT
# ===================================================

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Health Engine API",
        "version": api_config.VERSION,
        "description": "Recovery & Readiness Prediction System",
        "docs": "/docs",
        "health": "/health"
    }


# ===================================================
# STARTUP/SHUTDOWN
# ===================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Health Engine API starting up")
    logger.info(f"Model path: {config.MODEL_PATH}")
    logger.info(f"Features: {len(config.PRIMARY_FEATURES + config.ENGINEERED_FEATURES)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Health Engine API shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=api_config.HOST,
        port=api_config.PORT,
        log_level="info"
    )
