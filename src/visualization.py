"""Reusable matplotlib/seaborn helpers; figures are returned for notebook display."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_class_distribution(data, label_column=None, count_column=None, title="Class Distribution", figsize=(10, 6)):
    """Plot a horizontal class-count bar chart from a Series or count table."""
    if hasattr(data, "columns"):
        if label_column is None or count_column is None:
            raise ValueError("label_column and count_column are required for tabular input.")
        labels, counts = data[label_column], data[count_column]
    else:
        labels, counts = data.index, data.values
    order = counts.argsort()
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh([labels.iloc[i] if hasattr(labels, "iloc") else labels[i] for i in order], counts.iloc[order] if hasattr(counts, "iloc") else counts[order])
    ax.set(title=title, xlabel="Count", ylabel="Class")
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(matrix, title="Confusion Matrix", figsize=(12, 10), cmap="Blues"):
    """Plot a labeled confusion-matrix DataFrame and return its figure/axis."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, cmap=cmap, ax=ax, cbar_kws={"label": "Record count"})
    ax.set(title=title, xlabel="Predicted class", ylabel="Actual class")
    ax.tick_params(axis="x", labelrotation=55)
    fig.tight_layout()
    return fig, ax


def plot_missing_values(summary, column="Missing Percentage", title="Missing Values by Column", figsize=(10, 6)):
    """Plot the selected missingness summary column as a horizontal bar chart."""
    values = summary[column].dropna()
    values = values[values > 0].sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    if values.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.barh(values.index.astype(str), values.values, color="steelblue")
    ax.set(title=title, xlabel="Missing values (%)", ylabel="Column")
    fig.tight_layout()
    return fig, ax


def plot_feature_importance(importance, feature_column="Feature", importance_column="Importance", top_n=20, figsize=(10, 7), title="Feature Importance"):
    """Plot a feature-importance table in descending order."""
    view = importance.sort_values(importance_column, ascending=False).head(top_n).sort_values(importance_column)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(view[feature_column].astype(str), view[importance_column], color="darkcyan")
    ax.set(title=title, xlabel="Importance", ylabel="Feature")
    fig.tight_layout()
    return fig, ax


def plot_model_comparison(results, metric, model_column="Model", figsize=(9, 5), xlim=None):
    """Plot one metric by model with a consistent readable horizontal layout."""
    view = results.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(view[model_column].astype(str), view[metric], color="steelblue")
    ax.set(title=f"{metric} by Model", xlabel=metric, ylabel="Model")
    if xlim is not None:
        ax.set_xlim(*xlim)
    pad = (xlim[1] - xlim[0]) * 0.015 if xlim is not None else max(view[metric].max() * 0.015, 0.005)
    for bar, value in zip(bars, view[metric]):
        ax.text(value + pad, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_ordered_counts(
    counts,
    title,
    xlabel="Incident count",
    ylabel="Category",
    kind="bar",
    figsize=(11, 5),
    color="steelblue",
):
    """Plot a count Series while preserving its caller-specified category order."""
    if kind not in {"bar", "barh", "line"}:
        raise ValueError("kind must be 'bar', 'barh', or 'line'.")
    fig, ax = plt.subplots(figsize=figsize)
    if kind == "line":
        ax.plot(counts.index, counts.values, marker="o", color=color)
    elif kind == "barh":
        ax.barh([str(value) for value in counts.index], counts.values, color=color)
    else:
        ax.bar([str(value) for value in counts.index], counts.values, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    return fig, ax


def plot_temporal_heatmap(values, title="Temporal Count Heatmap", xlabel="Hour of day", ylabel="Day of week", figsize=(14, 6), cmap="YlGnBu"):
    """Plot a pre-ordered temporal count table as an annotated count heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(values, cmap=cmap, ax=ax, cbar_kws={"label": "Incident count"})
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    return fig, ax


def plot_geographic_hexbin(data, latitude="lat", longitude="lng", gridsize=100, figsize=(9, 8), cmap="viridis"):
    """Aggregate all finite in-range coordinate rows into a 2D hexbin density plot.

    Invalid or missing coordinate pairs are excluded from the plot and returned
    as a count. No coordinates are imputed or fabricated.
    """
    lat = pd.to_numeric(data[latitude], errors="coerce")
    lon = pd.to_numeric(data[longitude], errors="coerce")
    valid = (lat.between(-90, 90) & lon.between(-180, 180) & np.isfinite(lat) & np.isfinite(lon))
    fig, ax = plt.subplots(figsize=figsize)
    artist = ax.hexbin(lon[valid], lat[valid], gridsize=gridsize, mincnt=1, cmap=cmap, bins="log")
    fig.colorbar(artist, ax=ax, label="Incident records per hexagon (log scale)")
    ax.set(title="Geographic Distribution of Recorded Incidents", xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
    fig.tight_layout()
    return fig, ax, int(valid.sum()), int((~valid).sum())


def plot_correlation_heatmap(data, title="Correlation Among Numerical Variables", figsize=(8, 6), cmap="vlag"):
    """Plot correlations for the numeric columns supplied by the analysis."""
    matrix = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax)
    ax.set(title=title, xlabel="Variable", ylabel="Variable")
    fig.tight_layout()
    return fig, ax


def plot_weighted_metrics(results, metrics=("Weighted Precision", "Weighted Recall", "Weighted F1"), figsize=(14, 5)):
    """Plot weighted precision, recall, and F1 in aligned panels for each model."""
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        view = results.sort_values(metric, ascending=True)
        bars = ax.barh(view["Model"].astype(str), view[metric], color="steelblue")
        ax.set(title=metric, xlabel="Score", xlim=(0, 1))
        for bar, value in zip(bars, view[metric]):
            ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("Model")
    fig.suptitle("Base Model Weighted Precision, Recall, and F1")
    fig.tight_layout()
    return fig, axes


def plot_model_metric_summary(results, metrics=None, model_column="Model", figsize=(13, 5)):
    """Show model-by-metric values without combining them into a ranking score."""
    metrics = list(metrics or [
        "Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1",
        "Macro Precision", "Macro Recall", "Macro F1",
    ])
    matrix = results.set_index(model_column)[metrics]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Score"})
    ax.set(title="Base Model Evaluation Metrics", xlabel="Evaluation metric", ylabel="Model")
    fig.tight_layout()
    return fig, ax


def plot_feature_count_validation(history, count_column="Features After Count", score_column="Validation Score", figsize=(10, 5)):
    """Plot training-only validation scores along a feature-elimination path."""
    view = history.sort_values(count_column, ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(view[count_column], view[score_column], marker="o", color="darkcyan")
    ax.set(title="Feature Count vs Internal Validation Weighted F1",
           xlabel="Encoded model features retained", ylabel="Internal validation weighted F1")
    ax.invert_xaxis()
    fig.tight_layout()
    return fig, ax


def plot_reduction_validation(results, metric, count_column="Feature Count", strategy_column="Strategy", figsize=(10, 5)):
    """Plot a validation metric against retained features for reduction strategies."""
    score_column = f"Validation {metric}"
    fig, ax = plt.subplots(figsize=figsize)
    for strategy, group in results.groupby(strategy_column, sort=False):
        view = group.sort_values(count_column, ascending=False)
        ax.plot(view[count_column], view[score_column], marker="o", label=strategy)
    ax.set(title=f"Retained Features vs Internal Validation {metric}",
           xlabel="Encoded model features retained", ylabel=f"Internal validation {metric}")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_baseline_reduced_comparison(
    results,
    metric,
    model_column="Model",
    baseline_prefix="Baseline",
    reduced_prefix="Reduced",
    figsize=(10, 6),
):
    """Compare baseline and reduced metrics without combining them into a score."""
    baseline_column = f"{baseline_prefix} {metric}"
    reduced_column = f"{reduced_prefix} {metric}"
    view = results.sort_values(baseline_column, ascending=True).reset_index(drop=True)
    y = np.arange(len(view))
    height = 0.36
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y - height / 2, view[baseline_column], height, label="39-feature baseline", color="slateblue")
    ax.barh(y + height / 2, view[reduced_column], height, label="Selected subset", color="darkorange")
    ax.set_yticks(y, view[model_column].astype(str))
    ax.set(title=f"Baseline vs Reduced {metric}", xlabel=metric, ylabel="Model", xlim=(0, 1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax
