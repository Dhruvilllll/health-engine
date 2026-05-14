"""
Health Engine - Full Pipeline Orchestration

Execute complete data pipeline:
Ingestion -> Cleaning -> Feature Engineering -> Recovery -> Model Training
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import config
from src.common.logger import setup_logger


logger = setup_logger(__name__)


def run_pipeline():
    """Run complete health engine pipeline."""
    print("\n" + "=" * 60)
    print(" HEALTH ENGINE - FULL PIPELINE")
    print("=" * 60)

    config.ensure_directories()
    logger.info("Project directories verified")

    print("\n[1/5] Data Ingestion...")
    try:
        from src.ingestion.load_physiological import main as ingest_phys
        from src.ingestion.load_sleep import main as ingest_sleep
        from src.ingestion.load_workouts import main as ingest_workouts

        logger.info("Loading physiological data...")
        ingest_phys()

        logger.info("Loading sleep data...")
        ingest_sleep()

        logger.info("Loading workout data...")
        ingest_workouts()

        logger.info("Data ingestion complete")
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        print(f"[ERROR] Ingestion failed: {str(e)}")
        return False

    print("\n[2/5] Data Cleaning & Preprocessing...")
    try:
        from src.preprocessing.clean_physiological import main as clean_phys

        logger.info("Cleaning physiological data...")
        clean_phys()
        logger.info("Data cleaning complete")

    except Exception as e:
        logger.error(f"Data cleaning failed: {str(e)}")
        print(f"[ERROR] Cleaning failed: {str(e)}")
        return False

    print("\n[3/5] Recovery Engine & Feature Engineering...")
    try:
        from src.recovery_engine.recovery_score import main as recovery_engine

        logger.info("Running recovery engine...")
        recovery_engine()
        logger.info("Recovery engine complete")

    except Exception as e:
        logger.error(f"Recovery engine failed: {str(e)}")
        print(f"[ERROR] Recovery engine failed: {str(e)}")
        return False

    print("\n[4/5] Model Training...")
    try:
        from src.models.train_recovery_model import main as train_model

        logger.info("Training Random Forest model...")
        train_model()
        logger.info("Model training complete")

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        print(f"[ERROR] Model training failed: {str(e)}")
        return False

    print("\n[5/5] Model Explainability...")
    try:
        from src.explainaibility.explain_model import main as explain

        logger.info("Generating model explanations...")
        explain()
        logger.info("Explainability analysis complete")

    except Exception as e:
        logger.error(f"Explainability failed: {str(e)}")
        logger.warning("Continuing without explainability (non-critical)")
        print(f"[WARN] Explainability skipped: {str(e)}")

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print("\nModel saved to: ", config.MODEL_PATH)
    print("You can now:")
    print("  - Run API: python -m uvicorn app.main:app --reload")
    print("  - Run Gradio UI: python app.py")
    print("=" * 60 + "\n")

    return True


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
