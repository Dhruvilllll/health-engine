"""
===================================================
Health Engine - Single Prediction CLI
===================================================

Make a single recovery prediction from command line.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.prediction_service import PredictionService
from src.common.logger import setup_logger
import argparse

logger = setup_logger(__name__)


def main():
    """Run single prediction."""
    parser = argparse.ArgumentParser(
        description="Health Engine - Single Prediction"
    )

    parser.add_argument("--hrv", type=float, help="Heart Rate Variability (ms)")
    parser.add_argument("--rhr", type=float, help="Resting Heart Rate (bpm)")
    parser.add_argument("--strain", type=float, help="Daily Strain")
    parser.add_argument("--spo2", type=float, help="Blood Oxygen (%)")
    parser.add_argument("--sleep-perf", type=float, help="Sleep Performance (%)")
    parser.add_argument("--avg-hr", type=float, help="Average HR (bpm)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Build input data
    input_data = {}

    if args.hrv is not None:
        input_data["hrv"] = args.hrv
    if args.rhr is not None:
        input_data["resting_heart_rate"] = args.rhr
    if args.strain is not None:
        input_data["day_strain"] = args.strain
    if args.spo2 is not None:
        input_data["spo2"] = args.spo2
    if args.sleep_perf is not None:
        input_data["sleep_performance_percentage"] = args.sleep_perf
    if args.avg_hr is not None:
        input_data["average_hr_bpm"] = args.avg_hr

    if not input_data:
        print("Error: No input provided. Use --help for usage.")
        return False

    try:
        # Initialize service
        service = PredictionService()

        # Make prediction
        result = service.predict(input_data)

        # Output results
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "="*50)
            print(" PREDICTION RESULT")
            print("="*50)
            print(f"Recovery Score: {result['recovery_score']:.2f}")
            print(f"Status: {result['recovery_status']}")
            print(f"Recommendation: {result['recommendation']}")
            print(f"Confidence: {result['confidence']}")
            print("="*50 + "\n")

        return True

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
