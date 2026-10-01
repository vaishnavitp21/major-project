"""Shared project paths and modeling constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

COMBINED_DATASET = PROCESSED_DATA_DIR / "Philadelphia_Crime_Incidents_2018_2022_Combined.csv"
GROUPED_DATASET = PROCESSED_DATA_DIR / "Philadelphia_Crime_Incidents_2018_2022_Grouped.csv"
GROUP_MAPPING = PROCESSED_DATA_DIR / "crime_class_group_mapping.csv"

TARGET_ORIGINAL = "text_general_code"
TARGET_GROUPED = "crime_group"
RANDOM_STATE = 42
TEST_SIZE = 0.30

IDENTIFIER_COLUMNS = ("objectid", "dc_key", "cartodb_id")
LEAKAGE_COLUMNS = ("ucr_general", TARGET_ORIGINAL, TARGET_GROUPED)
CATEGORICAL_FEATURES = ("dc_dist", "psa")
NUMERICAL_FEATURES = (
    "year", "month", "day_of_week", "day_of_month", "day_of_year", "week_of_year",
    "hour", "minute", "lat", "lng", "point_x", "point_y",
)
