"""
===========================================================
Health Engine — Physiological Data Ingestion Pipeline
===========================================================

Purpose:
--------
Loads, validates, standardizes, and merges physiological
wearable datasets for recovery prediction.

Author:
-------
Dhruvil Malvania

===========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import pandas as pd
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

RAW_DATA_PATH = "data/raw/physiological"

FILES = [
    "physiological_cycles.csv",
    "physiological_cycles(1).csv",
    "physiological_cycles(2).csv"
]

OUTPUT_PATH = "data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# =========================================================
# LOAD DATASET FUNCTION
# =========================================================

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV safely.

    Args:
        file_path (str): CSV file path

    Returns:
        pd.DataFrame
    """

    try:
        df = pd.read_csv(file_path)

        print(f"\n Loaded: {file_path}")
        print(f" Shape: {df.shape}")

        return df

    except Exception as e:
        print(f"\n Error loading {file_path}")
        print(e)

        return pd.DataFrame()


# =========================================================
# STANDARDIZE COLUMN NAMES
# =========================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names.

    - lowercase
    - remove spaces
    - replace special chars
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


# =========================================================
# MISSING VALUE REPORT
# =========================================================

def missing_value_report(df: pd.DataFrame):
    """
    Print missing value analysis.
    """

    print("\n Missing Value Report")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print(" No missing values found.")

    else:
        print(
            missing.sort_values(ascending=False)
        )


# =========================================================
# DATASET SUMMARY
# =========================================================

def dataset_summary(df: pd.DataFrame):
    """
    Print dataset summary.
    """

    print("\n Dataset Info")
    print(df.info())

    print("\n Statistical Summary")
    print(df.describe(include="all"))


# =========================================================
# MAIN INGESTION PIPELINE
# =========================================================

def main():

    print("\n================================================")
    print(" HEALTH ENGINE — PHYSIOLOGICAL INGESTION")
    print("================================================")

    dataframes = []

    # -----------------------------------------------------
    # LOAD FILES
    # -----------------------------------------------------

    for file in FILES:

        full_path = os.path.join(
            RAW_DATA_PATH,
            file
        )

        df = load_csv(full_path)

        if df.empty:
            continue

        # -------------------------------------------------
        # STANDARDIZE COLUMNS
        # -------------------------------------------------

        df = standardize_columns(df)

        print("\n Columns:")
        print(df.columns.tolist())

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        dataset_summary(df)

        # -------------------------------------------------
        # MISSING VALUES
        # -------------------------------------------------

        missing_value_report(df)

        dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(
            f"No physiological CSV files were loaded from {RAW_DATA_PATH}"
        )

    # -----------------------------------------------------
    # VALIDATE COLUMN CONSISTENCY
    # -----------------------------------------------------

    print("\n Checking schema consistency...")

    base_columns = set(dataframes[0].columns)

    for idx, df in enumerate(dataframes[1:], start=1):

        current_columns = set(df.columns)

        missing_cols = base_columns - current_columns
        extra_cols = current_columns - base_columns

        print(f"\n Dataset {idx}")

        print(f" Missing Columns: {missing_cols}")
        print(f" Extra Columns: {extra_cols}")

    # -----------------------------------------------------
    # MERGE DATASETS
    # -----------------------------------------------------

    print("\n Merging physiological datasets...")

    merged_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(f"\n Final Shape: {merged_df.shape}")

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    before = merged_df.shape[0]

    merged_df.drop_duplicates(inplace=True)

    after = merged_df.shape[0]

    print(f"\n Removed {before - after} duplicate rows.")

    # -----------------------------------------------------
    # SAVE PROCESSED DATA
    # -----------------------------------------------------

    output_file = os.path.join(
        OUTPUT_PATH,
        "physiological_merged.csv"
    )

    merged_df.to_csv(
        output_file,
        index=False
    )

    print("\n Processed file saved:")
    print(output_file)

    print("\n INGESTION PIPELINE COMPLETE")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
