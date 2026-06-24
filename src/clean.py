import pandas as pd
import numpy as np
import sqlite3

def load_raw_from_db(db_path: str) -> pd.DataFrame:
    """
    Load the raw Superstore dataset from the SQLite database.

    This function reads the table `raw_superstore` created during ingestion
    and returns it as a pandas DataFrame.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        DataFrame containing the raw Superstore data.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM raw_superstore", conn)
    conn.close()
    return df

def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric columns in the dataset.

    Operations:
    - Convert Sales, Profit, Quantity, and Discount to numeric types.
    - Replace invalid numeric values with NaN.
    - Clip Discount values to the valid range [0, 1].

    Notes:
    - Missing values and outliers are retained for dashboard analysis.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned numeric columns.
    """
    df = df.copy()
    numeric_cols = ["Sales", "Profit", "Quantity", "Discount"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Discount" in df.columns:
        df["Discount"] = df["Discount"].clip(lower=0, upper=1)

    return df

def _clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize categorical columns.

    Operations:
    - Remove leading and trailing whitespace.
    - Standardize text formatting using title case.
    - Convert columns to the pandas category dtype.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned categorical columns.
    """
    df = df.copy()
    categorical_cols = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .str.strip()
                .str.title()
                .astype("category")
            )  
    return df

def _clean_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean date columns in the dataset.

    Operations:
    - Convert Order Date and Ship Date to datetime format.
    - Replace invalid dates with NaT.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned date columns.
    """
    df = df.copy()

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors = "coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors = "coerce")

    df.loc[df["Ship Date"] < df["Order Date"], "Ship Date"] = pd.NaT

    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all data cleaning steps to the dataset.

    Cleaning steps:
    1. Clean numeric columns.
    2. Clean categorical columns.
    3. Clean date columns.
    4. Remove duplicate rows.

    Args:
        df: Raw input DataFrame.

    Returns:
        Cleaned DataFrame ready for analysis and dashboarding.
    """
    return (
        df.copy()
        .pipe(_clean_numeric)
        .pipe(_clean_categorical)
        .pipe(_clean_date)
        .drop_duplicates()
    )

def save_cleaned_to_db(df: pd.DataFrame, db_path: str):
    """
    Save the cleaned dataset to the SQLite database.

    The cleaned data is stored in a table named `cleaned_superstore`.

    Args:
        df: Cleaned DataFrame.
        db_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    df.to_sql("cleaned_superstore", conn, if_exists="replace", index=False)
    conn.close()


def save_cleaned_to_csv(df: pd.DataFrame, path: str = "data/processed/cleaned_superstore.csv"):
    """
    Save the cleaned dataset as a CSV file in the processed data folder.

    Args:
        df: Cleaned DataFrame.
        path: Output CSV file path.
    """
    df.to_csv(path, index=False)

def run_cleaning(db_path: str):
    """
    Execute the full cleaning pipeline:

    Steps:
    1. Load raw data from SQLite.
    2. Apply all cleaning functions.
    3. Save cleaned data to SQLite.
    4. Save cleaned data to CSV.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Cleaned DataFrame.
    """
    df_raw = load_raw_from_db(db_path)
    df_clean = clean(df_raw)

    save_cleaned_to_db(df_clean, db_path)
    save_cleaned_to_csv(df_clean)

    return df_clean


