import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime


def load_json_data(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded JSON data
    """
    logger = logging.getLogger(__name__)
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        raise


def load_csv_data(file_path, **kwargs):
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pandas.read_csv

    Returns:
        DataFrame containing the CSV data
    """
    logger = logging.getLogger(__name__)
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Successfully loaded data from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        raise


def save_dataframe(df, file_path, **kwargs):
    """
    Save a DataFrame to a file.

    Args:
        df: DataFrame to save
        file_path: Path to save the file
        **kwargs: Additional arguments to pass to DataFrame.to_csv or DataFrame.to_excel
    """
    logger = logging.getLogger(__name__)
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Determine file format and save accordingly
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".csv":
            df.to_csv(file_path, index=False, **kwargs)
        elif file_ext in [".xlsx", ".xls"]:
            df.to_excel(file_path, index=False, **kwargs)
        elif file_ext == ".json":
            df.to_json(file_path, orient="records", **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")

        logger.info(f"Successfully saved DataFrame to {file_path}, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {e}", exc_info=True)
        raise


def clean_column_names(df):
    """
    Clean column names in a DataFrame.

    Args:
        df: DataFrame with column names to clean

    Returns:
        DataFrame with cleaned column names
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("[^a-zA-Z0-9_]", "", regex=True)
    )

    return df


def merge_dataframes(df1, df2, on, how="inner", **kwargs):
    """
    Merge two DataFrames with error handling.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        on: Column(s) to join on
        how: Type of merge to perform
        **kwargs: Additional arguments to pass to pandas.merge

    Returns:
        Merged DataFrame
    """
    logger = logging.getLogger(__name__)
    try:
        merged_df = pd.merge(df1, df2, on=on, how=how, **kwargs)
        logger.info(
            f"Successfully merged DataFrames with shapes {df1.shape} and {df2.shape} -> {merged_df.shape}"
        )
        return merged_df
    except Exception as e:
        logger.error(f"Error merging DataFrames: {e}", exc_info=True)
        raise


def fill_missing_values(df, columns=None, strategy="mean"):
    """
    Fill missing values in a DataFrame.

    Args:
        df: DataFrame with missing values
        columns: Columns to fill (if None, fill all columns)
        strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'zero', 'forward', 'backward')

    Returns:
        DataFrame with filled missing values
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # If columns not specified, use all columns
    if columns is None:
        columns = df.columns

    # Apply filling strategy
    for col in columns:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else np.nan
                )
            elif strategy == "zero":
                df[col] = df[col].fillna(0)
            elif strategy == "forward":
                df[col] = df[col].fillna(method="ffill")
            elif strategy == "backward":
                df[col] = df[col].fillna(method="bfill")
        else:
            if strategy == "mode":
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else ""
                )
            elif strategy == "zero" or strategy == "empty":
                df[col] = df[col].fillna("")
            elif strategy == "forward":
                df[col] = df[col].fillna(method="ffill")
            elif strategy == "backward":
                df[col] = df[col].fillna(method="bfill")

    return df


def parse_date(date_str, formats=None):
    """
    Parse a date string with various formats.

    Args:
        date_str: Date string to parse
        formats: List of date formats to try

    Returns:
        Parsed datetime object or None if parsing fails
    """
    if formats is None:
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y%m%d",
            "%d.%m.%Y",
            "%Y.%m.%d",
            "%b %d, %Y",
            "%B %d, %Y",
            "%d %b %Y",
            "%d %B %Y",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]

    if not date_str or pd.isna(date_str):
        return None

    if isinstance(date_str, (datetime, pd.Timestamp)):
        return date_str

    # Convert to string if needed
    date_str = str(date_str).strip()

    # Try each format
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue

    return None


def remove_duplicate_rows(df, subset=None, keep="first"):
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df: DataFrame to process
        subset: Columns to consider for duplicates (if None, use all columns)
        keep: Which duplicates to keep ('first', 'last', False)

    Returns:
        DataFrame with duplicates removed
    """
    logger = logging.getLogger(__name__)

    initial_rows = len(df)
    df_deduped = df.drop_duplicates(subset=subset, keep=keep)
    removed_rows = initial_rows - len(df_deduped)

    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} duplicate rows")

    return df_deduped


def validate_dataframe(df, required_columns=None, data_types=None):
    """
    Validate a DataFrame against expected schema.

    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        data_types: Dictionary mapping column names to expected data types

    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []

    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

    # Check data types
    if data_types:
        for col, expected_type in data_types.items():
            if col in df.columns:
                # Check if column values match expected type
                if expected_type == "int":
                    if not pd.api.types.is_integer_dtype(df[col]) and not all(
                        pd.isna(val) or float(val).is_integer() for val in df[col]
                    ):
                        errors.append(f"Column '{col}' does not contain integer values")
                elif expected_type == "float":
                    if not pd.api.types.is_float_dtype(
                        df[col]
                    ) and not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"Column '{col}' is not numeric")
                elif expected_type == "date":
                    try:
                        pd.to_datetime(df[col])
                    except:
                        errors.append(f"Column '{col}' cannot be converted to datetime")
                elif expected_type == "bool":
                    if not pd.api.types.is_bool_dtype(df[col]) and not all(
                        pd.isna(val)
                        or val
                        in (
                            0,
                            1,
                            True,
                            False,
                            "True",
                            "False",
                            "true",
                            "false",
                            "yes",
                            "no",
                            "y",
                            "n",
                        )
                        for val in df[col]
                    ):
                        errors.append(f"Column '{col}' does not contain boolean values")
                elif expected_type == "category":
                    if (
                        not pd.api.types.is_categorical_dtype(df[col])
                        and len(df[col].unique()) / len(df) < 0.5
                    ):
                        # It's not a categorical column but has fewer unique values than 50% of total rows
                        # Not an error, but could be optimized
                        pass

    is_valid = len(errors) == 0
    return (is_valid, errors)
