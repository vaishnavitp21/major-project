"""Read-only CSV loading helpers for project datasets."""

from pathlib import Path
from typing import Union

import pandas as pd

from .config import COMBINED_DATASET, GROUPED_DATASET, GROUP_MAPPING, PROJECT_ROOT

PathLike = Union[str, Path]


def load_csv_dataset(path: PathLike, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV from an absolute or project-relative path without modifying it."""
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path, **read_csv_kwargs)


def load_combined_philadelphia_data(**read_csv_kwargs) -> pd.DataFrame:
    """Load the canonical combined Philadelphia incident dataset."""
    return load_csv_dataset(COMBINED_DATASET, **read_csv_kwargs)


def load_grouped_philadelphia_data(**read_csv_kwargs) -> pd.DataFrame:
    """Load the grouped Philadelphia dataset used by the base-model experiment."""
    return load_csv_dataset(GROUPED_DATASET, **read_csv_kwargs)


def load_class_group_mapping(**read_csv_kwargs) -> pd.DataFrame:
    """Load the documented original-to-grouped crime-class mapping."""
    return load_csv_dataset(GROUP_MAPPING, **read_csv_kwargs)
