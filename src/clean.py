import pandas as pd
import numpy as np

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

    #Convert to numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            #Clip Discount to known valid range
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
                #Remove leading/trailing whitespace
                .str.strip()
                #Normalize text formatting
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
