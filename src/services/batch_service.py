"""
===================================================
Health Engine - Batch Processing Service
===================================================

Service for batch processing CSV files.
"""

import pandas as pd
import io
from pathlib import Path
from typing import Optional, Tuple

from src.services.prediction_service import PredictionService
from src.common.logger import setup_logger


logger = setup_logger(__name__)


class BatchService:
    """Service for batch prediction processing."""

    def __init__(self, prediction_service: Optional[PredictionService] = None):
        """
        Initialize batch service.

        Args:
            prediction_service: Optional PredictionService instance
        """
        self.service = prediction_service or PredictionService()

    def process_csv_file(
        self,
        file_path: Path
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Process CSV file and return predictions.

        Args:
            file_path: Path to input CSV

        Returns:
            Tuple of (results DataFrame, processing stats)
        """
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV: {file_path} with {len(df)} records")

            # Process
            results, stats = self._process_dataframe(df)

            return results, stats

        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise ValueError(f"CSV processing failed: {str(e)}")

    def process_csv_bytes(
        self,
        file_bytes: bytes
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Process CSV from bytes.

        Args:
            file_bytes: CSV file content as bytes

        Returns:
            Tuple of (results DataFrame, processing stats)
        """
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            logger.info(f"Loaded CSV from bytes with {len(df)} records")

            results, stats = self._process_dataframe(df)
            return results, stats

        except Exception as e:
            logger.error(f"Error processing CSV bytes: {str(e)}")
            raise ValueError(f"CSV processing failed: {str(e)}")

    def _process_dataframe(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Process DataFrame with error handling per row.

        Returns:
            Tuple of (results with predictions, stats dict)
        """
        results = []
        errors = []
        successful = 0

        for idx, row in df.iterrows():
            try:
                record = row.to_dict()
                prediction = self.service.predict(record)

                # Add original data + prediction
                result_row = {**record, **prediction}
                results.append(result_row)
                successful += 1

            except Exception as e:
                error_entry = {
                    "row_index": idx,
                    "error": str(e)
                }
                errors.append(error_entry)
                logger.warning(f"Row {idx} prediction failed: {str(e)}")

        results_df = pd.DataFrame(results) if results else pd.DataFrame()

        stats = {
            "total_records": len(df),
            "successful": successful,
            "failed": len(errors),
            "success_rate": round(100 * successful / len(df), 2) if df.shape[0] > 0 else 0,
            "errors": errors
        }

        logger.info(f"Batch processing complete: {successful}/{len(df)} successful")

        return results_df, stats

    def save_results(
        self,
        results_df: pd.DataFrame,
        output_path: Path
    ):
        """Save prediction results to CSV."""
        try:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise ValueError(f"Failed to save results: {str(e)}")
