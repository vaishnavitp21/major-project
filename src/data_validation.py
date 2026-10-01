"""Reusable, non-mutating dataset and modeling-input validation."""

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .config import LEAKAGE_COLUMNS


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[Iterable[str]] = None) -> dict:
    """Validate a dataframe's basic structure and return a compact schema summary."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame contains no rows.")
    missing = [column for column in (required_columns or []) if column not in df.columns]
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "duplicate_column_names": df.columns[df.columns.duplicated()].tolist(),
        "memory_bytes": int(df.memory_usage(index=True, deep=True).sum()),
    }


def inspect_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtype, non-null, missing, missing-percent, and unique-value counts."""
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).to_numpy(),
        "Non-Null Count": df.notna().sum().to_numpy(),
        "Missing Count": df.isna().sum().to_numpy(),
        "Missing Percentage": (df.isna().mean().to_numpy() * 100).round(3),
        "Unique Values": df.nunique(dropna=True).to_numpy(),
    })


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize column missingness, sorted from most to least missing."""
    out = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing Percentage": (df.isna().mean() * 100).round(3),
        "Non-Missing Count": df.notna().sum(),
        "Unique Non-Missing Values": df.nunique(dropna=True),
    })
    out.index.name = "Column"
    return out.sort_values(["Missing Percentage", "Missing Count"], ascending=False)


def check_duplicates(df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> dict:
    """Return exact/subset duplicate counts and percentages without dropping rows."""
    duplicated = df.duplicated(subset=list(subset) if subset is not None else None)
    count = int(duplicated.sum())
    return {
        "duplicate_count": count,
        "duplicate_percentage": (count / len(df) * 100) if len(df) else 0.0,
        "subset": list(subset) if subset is not None else None,
    }


def check_numeric_validity(
    df: pd.DataFrame,
    columns: Iterable[str],
    valid_ranges: Optional[Mapping[str, tuple]] = None,
) -> pd.DataFrame:
    """Report missing, nonnumeric/nonfinite, range-invalid, and observed numeric bounds."""
    valid_ranges = valid_ranges or {}
    records = []
    for column in columns:
        if column not in df:
            records.append({"Column": column, "Missing Count": np.nan, "Invalid Count": np.nan,
                            "Minimum": np.nan, "Maximum": np.nan, "Present": False})
            continue
        source = df[column]
        values = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & values.isna()
        finite = pd.Series(np.isfinite(values.fillna(0)), index=values.index)
        invalid |= values.notna() & ~finite
        if column in valid_ranges:
            low, high = valid_ranges[column]
            invalid |= values.notna() & ~values.between(low, high)
        clean = values.where(finite)
        records.append({"Column": column, "Missing Count": int(source.isna().sum()),
                        "Invalid Count": int(invalid.sum()), "Minimum": clean.min(),
                        "Maximum": clean.max(), "Present": True})
    return pd.DataFrame(records)


def validate_target(
    df: pd.DataFrame,
    target_column: str,
    expected_classes: Optional[int] = None,
    allow_missing: bool = False,
) -> dict:
    """Validate target presence/completeness and return observed class information."""
    if target_column not in df:
        raise KeyError(f"Target column {target_column!r} is absent.")
    missing = int(df[target_column].isna().sum())
    if missing and not allow_missing:
        raise ValueError(f"Target {target_column!r} contains {missing} missing values.")
    classes = df[target_column].dropna().unique().tolist()
    if expected_classes is not None and len(classes) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes in {target_column!r}; found {len(classes)}.")
    return {"target": target_column, "class_count": len(classes), "classes": classes, "missing_count": missing}


def get_class_distribution(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Return class counts and percentages, sorted by frequency descending."""
    if target_column not in df:
        raise KeyError(f"Target column {target_column!r} is absent.")
    counts = df[target_column].value_counts(dropna=False)
    return pd.DataFrame({"Class": counts.index, "Count": counts.values,
                         "Percentage": (counts.values / len(df) * 100).round(4)})


def check_target_leakage(
    feature_columns: Iterable[str],
    target_column: str,
    forbidden_columns: Optional[Iterable[str]] = None,
) -> dict:
    """Reject target/known-leakage fields in a proposed predictor list."""
    features = list(feature_columns)
    forbidden = set(forbidden_columns if forbidden_columns is not None else LEAKAGE_COLUMNS)
    forbidden.add(target_column)
    violations = sorted(set(features) & forbidden)
    if violations:
        raise ValueError(f"Target/leakage columns cannot be predictors: {violations}")
    return {"valid": True, "feature_columns": features, "target_column": target_column,
            "excluded_columns_checked": sorted(forbidden)}
