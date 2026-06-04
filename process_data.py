"""process_data.py - Clean BeamNG road/electrics driving data.

Author(s): Matt Gallenberger, Tanuj Reddy Thummala
Class: CS450-01
Date: 04/29/26

Input:  raw_data.csv from collect_data.py
Output: processed_data.csv with the same 9 columns, ready for DrivingDataset

This script intentionally does not min-max normalize Steering, Throttle, Brake,
or Speed.  The live BeamNG agents currently read raw speed and send raw controls,
so training should use the same scale unless a shared normalization/inverse-
normalization system is added later.

Usage:
    python process_data.py
    python process_data.py --input raw_data.csv --output processed_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from state_schema import REQUIRED_COLUMNS, TARGET_CLIP_RANGES, validate_required_columns


def parse_args() -> argparse.Namespace:
    """Read command-line options for input/output CSV paths.

    Returns:
        argparse.Namespace with input and output path strings.
    """
    parser = argparse.ArgumentParser(description="Clean BeamNG driving CSV for training.")
    parser.add_argument("--input", default="raw_data.csv", help="Raw CSV from collect_data.py.")
    parser.add_argument("--output", default="processed_data.csv", help="Cleaned CSV to write.")
    return parser.parse_args()


def process_csv(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Clean a raw BeamNG driving CSV and write the processed CSV.

    Args:
        input_path: Path to raw_data.csv.
        output_path: Path where processed_data.csv should be written.

    Returns:
        Cleaned DataFrame.  Returning the DataFrame makes this function easy to
        test from a short Python command without re-reading the output file.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If required driving columns are missing.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Validate the schema before selecting columns.  A clear error here is much
    # easier to debug than a KeyError inside the training notebook later.
    missing = validate_required_columns(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing columns: {missing}\n"
            "Make sure you are using raw_data.csv from the current collect_data.py."
        )

    before = len(df)

    # Keep only the exact columns used by DrivingDataset, in canonical order.
    # This prevents unrelated debug columns from accidentally leaking into
    # training code.
    df = df[REQUIRED_COLUMNS].copy()
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    # Controls are labels for the supervised model.  Clip impossible values but
    # preserve their meaning.  In particular, do not map Steering to [0, 1].
    for column, (low, high) in TARGET_CLIP_RANGES.items():
        df[column] = df[column].astype(float).clip(lower=low, upper=high)

    df.to_csv(output_path, index=False)

    dropped = before - len(df)
    print(f"Read    {before} rows from {input_path}")
    print(f"Dropped {dropped} rows with missing values")
    print(f"Wrote   {len(df)} rows to {output_path}")
    print(f"Columns: {', '.join(REQUIRED_COLUMNS)}")

    return df


if __name__ == "__main__":
    args = parse_args()
    process_csv(args.input, args.output)
