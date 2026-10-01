"""Reusable multiclass evaluation and comparison helpers."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_multiclass_model(y_true, y_pred) -> dict:
    """Return accuracy and weighted/macro precision, recall, and F1."""
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted Precision": weighted[0],
        "Weighted Recall": weighted[1],
        "Weighted F1": weighted[2],
        "Macro Precision": macro[0],
        "Macro Recall": macro[1],
        "Macro F1": macro[2],
    }


def generate_classification_report(y_true, y_pred, labels=None, target_names=None) -> pd.DataFrame:
    """Return the class-wise sklearn classification report as a DataFrame."""
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names,
                                   output_dict=True, zero_division=0)
    return pd.DataFrame(report).T


def create_confusion_matrix_data(y_true, y_pred, labels=None, display_labels=None) -> pd.DataFrame:
    """Return a labeled confusion matrix as a DataFrame."""
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    names = list(display_labels) if display_labels is not None else labels
    if len(labels) != len(names):
        raise ValueError("labels and display_labels must have the same length.")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=pd.Index(names, name="Actual"),
                        columns=pd.Index(names, name="Predicted"))


def add_model_result(model_name: str, metrics: dict, timings: dict | None = None) -> dict:
    """Combine model name, metric values, and optional timing fields into a result row."""
    return {"Model": model_name, **metrics, **(timings or {})}


def create_model_comparison_table(results, sort_by: str = "Weighted F1", ascending: bool = False) -> pd.DataFrame:
    """Create a model comparison table, optionally sorted by an available metric."""
    table = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=ascending)
    return table.reset_index(drop=True)
