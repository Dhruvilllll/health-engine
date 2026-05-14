"""
===================================================
Health Engine - Batch Prediction CLI
===================================================

Process CSV file with batch predictions.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.batch_service import BatchService
from src.services.prediction_service import PredictionService
from src.common.logger import setup_logger
import argparse

logger = setup_logger(__name__)


def main():
    """Run batch prediction."""
    parser = argparse.ArgumentParser(
        description="Health Engine - Batch Prediction"
    )

    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path (default: input_file_predictions.csv)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    # Determine output path
    output_path = Path(args.output) if args.output else \
        input_path.parent / f"{input_path.stem}_predictions.csv"

    try:
        print(f"\nProcessing: {input_path}")

        # Initialize service
        service = PredictionService()
        batch_service = BatchService(service)

        # Process CSV
        results_df, stats = batch_service.process_csv_file(input_path)

        # Save results
        batch_service.save_results(results_df, output_path)

        # Display statistics
        print("\n" + "="*50)
        print(" BATCH PROCESSING RESULTS")
        print("="*50)
        print(f"Total Records: {stats['total_records']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Output: {output_path}")
        print("="*50 + "\n")

        if stats["errors"] and args.verbose:
            print("Errors:")
            for error in stats["errors"]:
                print(f"  Row {error['row_index']}: {error['error']}")

        return True

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
