"""
Module 1 (Lego Block: Data Ingestion & Schema Validation)
Responsible for loading raw comment exports (CSV, Parquet, or JSONL),
validating the schema contract, and standardizing Reddit ID prefixes.
"""

import os
from typing import Tuple, List
import pandas as pd


REQUIRED_COLUMNS = [
    "comment_id",
    "parent_id",
    "thread_id",
    "subreddit",
    "body",
    "created_utc"
]

OPTIONAL_COLUMNS = ["author", "score"]


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that the loaded DataFrame conforms to the required contract.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if not errors:
        # Check for non-empty DataFrame
        if len(df) == 0:
            errors.append("Dataset is empty.")
        
        # Check for null values in critical columns
        for col in ["comment_id", "parent_id", "thread_id", "body", "created_utc"]:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} null/NaN values.")
                
    return (len(errors) == 0, errors)


def clean_reddit_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips Reddit type prefixes (e.g., 't1_' for comments, 't3_' for submissions)
    from identifier columns to ensure consistent graph node linkage.
    """
    df = df.copy()
    for col in ["comment_id", "parent_id", "thread_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"^t[13]_", "", regex=True)
            
    # Normalize subreddit to lowercase
    if "subreddit" in df.columns:
        df["subreddit"] = df["subreddit"].astype(str).str.lower()
        
    # Remove text artifact rows like [deleted] or [removed] if any slipped through
    df = df[~df["body"].isin(["[deleted]", "[removed]"])].reset_index(drop=True)
    return df


def load_raw_comments(file_path: str) -> pd.DataFrame:
    """
    Loads raw comment data from a local CSV, Parquet, or JSON file,
    validates the schema, and cleans prefixes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(file_path)
    elif ext in [".json", ".jsonl"]:
        df = pd.read_json(file_path, lines=ext == ".jsonl")
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .csv, .parquet, or .jsonl")

    is_valid, errors = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Schema validation failed:\n" + "\n".join(f"- {e}" for e in errors))

    cleaned_df = clean_reddit_prefixes(df)
    print(f"Successfully loaded and validated {len(cleaned_df):,} comments across {cleaned_df['thread_id'].nunique()} threads.")
    return cleaned_df


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw/reddit_comments.csv"
    print(f"Checking data file: {target}")
    if os.path.exists(target):
        loaded = load_raw_comments(target)
        print(loaded.head())
    else:
        print(f"File '{target}' not yet present. Please drop your export into data/raw/.")
