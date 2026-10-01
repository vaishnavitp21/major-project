"""
Feature Engineering Module
Philadelphia Crime Prediction Project

Creates three model-ready datasets:

1. crime_classification.csv
   -> XGBoost
   -> CatBoost
   -> Random Forest

2. daily_crime_counts.csv
   -> LSTM
   -> BiLSTM

3. spatial_temporal_grid.csv
   -> CNN-LSTM
"""

import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD COMBINED DATASET
# ============================================================

def find_combined_dataset():
    """
    Find the original combined Philadelphia crime dataset.
    """

    target_file = os.path.join(
        PROCESSED_DIR,
        "Philadelphia_Crime_Incidents_2018_2022.csv"
    )

    if not os.path.exists(target_file):
        raise FileNotFoundError(
            f"Philadelphia crime dataset not found: {target_file}"
        )

    return target_file  


def load_data():
    """
    Load the combined Philadelphia crime dataset.
    """

    file_path = find_combined_dataset()

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    print(f"File: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# BASIC CLEANING
# ============================================================

def basic_cleaning(df):
    """
    Basic cleaning before feature engineering.
    """

    df = df.copy()

    print("\n" + "=" * 70)
    print("BASIC CLEANING")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Duplicate rows removed: {before - after:,}")

    # --------------------------------------------------------
    # Convert datetime
    # --------------------------------------------------------

    df["dispatch_date_time"] = pd.to_datetime(
        df["dispatch_date_time"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df["dispatch_date"] = pd.to_datetime(
        df["dispatch_date"],
        errors="coerce"
    )

    # If dispatch_date is missing, obtain it from datetime
    missing_date = df["dispatch_date"].isna()

    df.loc[missing_date, "dispatch_date"] = (
        df.loc[missing_date, "dispatch_date_time"].dt.normalize()
    )

    # --------------------------------------------------------
    # Remove rows without target crime type
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=["text_general_code"]
    )

    after = len(df)

    print(
        f"Rows removed due to missing crime type: "
        f"{before - after:,}"
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "dc_dist",
        "psa",
        "hour",
        "lat",
        "lng"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def create_legacy_temporal_features(df):
    """
    Create temporal features from dispatch_date_time.
    """

    df = df.copy()

    print("\n" + "=" * 70)
    print("CREATING TEMPORAL FEATURES")
    print("=" * 70)

    dt = df["dispatch_date_time"]

    # --------------------------------------------------------
    # Basic date features
    # --------------------------------------------------------

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day

    df["day_of_week"] = dt.dt.dayofweek

    df["week_of_year"] = dt.dt.isocalendar().week.astype(
        "Int64"
    )

    df["quarter"] = dt.dt.quarter

    # --------------------------------------------------------
    # Weekend indicator
    # --------------------------------------------------------

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Time of day
    # --------------------------------------------------------

    df["hour"] = dt.dt.hour

    # If hour was missing, use the existing hour column
    if "hour" in df.columns:

        original_hour = pd.to_numeric(
            df["hour"],
            errors="coerce"
        )

        df["hour"] = original_hour.fillna(
            dt.dt.hour
        )

    # --------------------------------------------------------
    # Cyclical hour encoding
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # --------------------------------------------------------
    # Cyclical month encoding
    # --------------------------------------------------------

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    print("Temporal features created:")
    print(
        [
            "year",
            "month",
            "day",
            "day_of_week",
            "week_of_year",
            "quarter",
            "is_weekend",
            "hour",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos"
        ]
    )

    return df


# ============================================================
# DATASET 1: CRIME CLASSIFICATION
# ============================================================

def create_classification_dataset(df):
    """
    Create dataset for:

    XGBoost
    CatBoost
    Random Forest

    Target:
        text_general_code
    """

    print("\n" + "=" * 70)
    print("CREATING CRIME CLASSIFICATION DATASET")
    print("=" * 70)

    features = [
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "quarter",
        "is_weekend",
        "hour",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dc_dist",
        "psa",
        "lat",
        "lng",
        "text_general_code"
    ]

    # Keep only columns that exist
    features = [
        column
        for column in features
        if column in df.columns
    ]

    classification_df = df[features].copy()

    # --------------------------------------------------------
    # Remove rows with missing essential spatial features
    # --------------------------------------------------------

    classification_df = classification_df.dropna(
        subset=["dc_dist", "psa"]
    )

    # --------------------------------------------------------
    # Fill coordinate missing values with median
    # --------------------------------------------------------

    for column in ["lat", "lng"]:

        if column in classification_df.columns:

            classification_df[column] = (
                classification_df[column]
                .fillna(classification_df[column].median())
            )

    # --------------------------------------------------------
    # Convert categorical spatial fields to numeric
    # --------------------------------------------------------

    for column in ["dc_dist", "psa"]:

        if column in classification_df.columns:

            classification_df[column] = pd.to_numeric(
                classification_df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Remove remaining invalid rows
    # --------------------------------------------------------

    classification_df = classification_df.dropna()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        "crime_classification.csv"
    )

    classification_df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")
    print(
        f"Shape: {classification_df.shape[0]:,} "
        f"rows × {classification_df.shape[1]} columns"
    )

    print("\nTarget distribution:")
    print(
        classification_df["text_general_code"]
        .value_counts()
        .head(15)
    )

    return classification_df


# ============================================================
# DATASET 2: DAILY CRIME COUNTS
# ============================================================

def create_daily_crime_dataset(df):
    """
    Create daily crime-count dataset for:

    LSTM
    BiLSTM

    Output contains daily total crime counts and
    temporal features.
    """

    print("\n" + "=" * 70)
    print("CREATING DAILY CRIME COUNT DATASET")
    print("=" * 70)

    temp = df.copy()

    # --------------------------------------------------------
    # Create clean date
    # --------------------------------------------------------

    temp["date"] = temp["dispatch_date_time"].dt.date

    temp["date"] = pd.to_datetime(
        temp["date"]
    )

    # --------------------------------------------------------
    # Daily total crime count
    # --------------------------------------------------------

    daily = (
        temp.groupby("date")
        .size()
        .reset_index(name="crime_count")
    )

    # --------------------------------------------------------
    # Create complete date range
    # --------------------------------------------------------

    full_dates = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D"
    )

    daily = (
        daily.set_index("date")
        .reindex(full_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["week_of_year"] = (
        daily["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    daily["quarter"] = daily["date"].dt.quarter

    daily["is_weekend"] = (
        daily["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical features
    # --------------------------------------------------------

    daily["month_sin"] = np.sin(
        2 * np.pi * daily["month"] / 12
    )

    daily["month_cos"] = np.cos(
        2 * np.pi * daily["month"] / 12
    )

    daily["day_of_week_sin"] = np.sin(
        2 * np.pi * daily["day_of_week"] / 7
    )

    daily["day_of_week_cos"] = np.cos(
        2 * np.pi * daily["day_of_week"] / 7
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    daily["lag_1"] = (
        daily["crime_count"].shift(1)
    )

    daily["lag_7"] = (
        daily["crime_count"].shift(7)
    )

    daily["lag_14"] = (
        daily["crime_count"].shift(14)
    )

    daily["lag_30"] = (
        daily["crime_count"].shift(30)
    )

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    daily["rolling_mean_7"] = (
        daily["crime_count"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    daily["rolling_mean_14"] = (
        daily["crime_count"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    daily["rolling_mean_30"] = (
        daily["crime_count"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    # --------------------------------------------------------
    # Drop initial rows created by lag features
    # --------------------------------------------------------

    daily = daily.dropna().reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        "daily_crime_counts.csv"
    )

    daily.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")
    print(
        f"Shape: {daily.shape[0]:,} "
        f"rows × {daily.shape[1]} columns"
    )

    print("\nDaily crime statistics:")
    print(
        daily["crime_count"].describe()
    )

    return daily


# ============================================================
# DATASET 3: DISTRICT-BASED SPATIO-TEMPORAL DATA
# ============================================================

def create_spatial_temporal_dataset(df):
    """
    Create district-based spatial-temporal dataset.

    This dataset is intended as the intermediate representation
    for CNN-LSTM.

    Spatial unit:
        dc_dist

    Temporal unit:
        date
    """

    print("\n" + "=" * 70)
    print("CREATING SPATIO-TEMPORAL DATASET")
    print("=" * 70)

    temp = df.copy()

    # --------------------------------------------------------
    # Create date
    # --------------------------------------------------------

    temp["date"] = (
        temp["dispatch_date_time"]
        .dt.normalize()
    )

    # --------------------------------------------------------
    # Remove missing spatial units
    # --------------------------------------------------------

    temp = temp.dropna(
        subset=["dc_dist"]
    )

    # --------------------------------------------------------
    # Daily crime count by district
    # --------------------------------------------------------

    spatial = (
        temp.groupby(
            ["date", "dc_dist"]
        )
        .size()
        .reset_index(
            name="crime_count"
        )
    )

    # --------------------------------------------------------
    # Complete date × district combinations
    # --------------------------------------------------------

    dates = pd.date_range(
        start=spatial["date"].min(),
        end=spatial["date"].max(),
        freq="D"
    )

    districts = sorted(
        spatial["dc_dist"].dropna().unique()
    )

    full_index = pd.MultiIndex.from_product(
        [dates, districts],
        names=["date", "dc_dist"]
    )

    spatial = (
        spatial
        .set_index(["date", "dc_dist"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    spatial["year"] = spatial["date"].dt.year
    spatial["month"] = spatial["date"].dt.month
    spatial["day"] = spatial["date"].dt.day

    spatial["day_of_week"] = (
        spatial["date"].dt.dayofweek
    )

    spatial["is_weekend"] = (
        spatial["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical temporal features
    # --------------------------------------------------------

    spatial["month_sin"] = np.sin(
        2 * np.pi * spatial["month"] / 12
    )

    spatial["month_cos"] = np.cos(
        2 * np.pi * spatial["month"] / 12
    )

    spatial["day_of_week_sin"] = np.sin(
        2 * np.pi * spatial["day_of_week"] / 7
    )

    spatial["day_of_week_cos"] = np.cos(
        2 * np.pi * spatial["day_of_week"] / 7
    )

    # --------------------------------------------------------
    # District lag features
    # --------------------------------------------------------

    spatial = spatial.sort_values(
        ["dc_dist", "date"]
    )

    spatial["lag_1"] = (
        spatial
        .groupby("dc_dist")["crime_count"]
        .shift(1)
    )

    spatial["lag_7"] = (
        spatial
        .groupby("dc_dist")["crime_count"]
        .shift(7)
    )

    spatial["rolling_mean_7"] = (
        spatial
        .groupby("dc_dist")["crime_count"]
        .transform(
            lambda x:
            x.shift(1).rolling(7).mean()
        )
    )

    # --------------------------------------------------------
    # Fill lag-related missing values
    # --------------------------------------------------------

    spatial["lag_1"] = (
        spatial["lag_1"].fillna(0)
    )

    spatial["lag_7"] = (
        spatial["lag_7"].fillna(0)
    )

    spatial["rolling_mean_7"] = (
        spatial["rolling_mean_7"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        "spatial_temporal_grid.csv"
    )

    spatial.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")

    print(
        f"Shape: {spatial.shape[0]:,} "
        f"rows × {spatial.shape[1]} columns"
    )

    print(
        f"Number of districts: "
        f"{spatial['dc_dist'].nunique()}"
    )

    print(
        f"Date range: "
        f"{spatial['date'].min().date()} "
        f"to "
        f"{spatial['date'].max().date()}"
    )

    return spatial


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_feature_engineering():
    """
    Run the complete feature engineering pipeline.
    """

    print("\n")
    print("#" * 70)
    print("# PHILADELPHIA CRIME FEATURE ENGINEERING")
    print("#" * 70)

    # 1. Load
    df = load_data()

    # 2. Basic cleaning
    df = basic_cleaning(df)

    # 3. Temporal features
    df = create_legacy_temporal_features(df)

    # 4. Classification dataset
    classification_df = (
        create_classification_dataset(df)
    )

    # 5. Daily time-series dataset
    daily_df = (
        create_daily_crime_dataset(df)
    )

    # 6. Spatial-temporal dataset
    spatial_df = (
        create_spatial_temporal_dataset(df)
    )

    print("\n")
    print("#" * 70)
    print("# FEATURE ENGINEERING COMPLETED")
    print("#" * 70)

    print("\nGenerated files:")

    print(
        "1. dataset/processed/"
        "crime_classification.csv"
    )

    print(
        "2. dataset/processed/"
        "daily_crime_counts.csv"
    )

    print(
        "3. dataset/processed/"
        "spatial_temporal_grid.csv"
    )

    return (
        classification_df,
        daily_df,
        spatial_df
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_feature_engineering()


# Shared temporal helper used by the grouped base-model experiment.
def create_temporal_features(
    df,
    date_column="dispatch_date",
    datetime_column="dispatch_date_time",
    time_column="dispatch_time",
    hour_column="hour",
):
    """Return a copy with calendar/time fields matching base-model definitions."""
    out = df.copy()
    date_source = date_column if date_column in out else (datetime_column if datetime_column in out else None)
    parsed = None
    if date_source:
        parsed = pd.to_datetime(out[date_source], errors="coerce", utc=(date_source == datetime_column))
        out["year"] = parsed.dt.year
        out["month"] = parsed.dt.month
        out["day_of_week"] = parsed.dt.dayofweek
        out["day_of_month"] = parsed.dt.day
        out["day_of_year"] = parsed.dt.dayofyear
        out["week_of_year"] = parsed.dt.isocalendar().week.astype("float64")
    if hour_column in out:
        out["hour"] = pd.to_numeric(out[hour_column], errors="coerce")
    elif time_column in out:
        out["hour"] = pd.to_datetime(out[time_column], format="%H:%M:%S", errors="coerce").dt.hour
    elif parsed is not None:
        out["hour"] = parsed.dt.hour
    if time_column in out:
        out["minute"] = pd.to_datetime(out[time_column], format="%H:%M:%S", errors="coerce").dt.minute
    elif datetime_column in out:
        out["minute"] = pd.to_datetime(out[datetime_column], errors="coerce", utc=True).dt.minute
    return out
