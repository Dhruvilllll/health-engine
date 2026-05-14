"""
===========================================================
Health Engine — Workout Data Ingestion
===========================================================
"""

import os
import pandas as pd

RAW_DATA_PATH = "data/raw/workouts"

FILES = [
    "workouts.csv",
    "workouts(1).csv",
    "workouts(2).csv"
]

OUTPUT_PATH = "data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def standardize_columns(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w\s]", "", regex=True)
    )

    return df


def load_file(path):

    df = pd.read_csv(path)

    df = standardize_columns(df)

    return df


def main():

    print("\n Loading workout datasets...")

    dfs = []

    for file in FILES:

        full_path = os.path.join(
            RAW_DATA_PATH,
            file
        )

        df = load_file(full_path)

        print(f"\n Loaded: {file}")
        print(df.shape)

        dfs.append(df)

    merged = pd.concat(
        dfs,
        ignore_index=True
    )

    merged.drop_duplicates(inplace=True)

    print("\n Final Workout Shape:")
    print(merged.shape)

    output_file = os.path.join(
        OUTPUT_PATH,
        "workout_merged.csv"
    )

    merged.to_csv(
        output_file,
        index=False
    )

    print("\n Workout dataset saved.")


if __name__ == "__main__":
    main()