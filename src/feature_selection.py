"""Small reusable feature-analysis helpers; no selection is run automatically."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import get_scorer
from sklearn.metrics import accuracy_score, f1_score


def extract_feature_importance(estimator, feature_names) -> pd.DataFrame:
    """Return estimator importances or absolute linear coefficients by feature."""
    names = list(feature_names)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_)
        values = np.abs(values).mean(axis=0) if values.ndim > 1 else np.abs(values)
    else:
        raise ValueError("Estimator exposes neither feature_importances_ nor coef_.")
    if len(names) != len(values):
        raise ValueError(f"Feature-name count ({len(names)}) does not match importance count ({len(values)}).")
    return pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False, kind="stable").reset_index(drop=True)


def rank_feature_importances(estimator, feature_names) -> pd.DataFrame:
    """Return descending Feature/Importance/Rank columns with stable tie ordering."""
    ranked = extract_feature_importance(estimator, feature_names)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    return ranked[["Feature", "Importance", "Rank"]]


def compare_feature_importance(random_forest, xgboost) -> pd.DataFrame:
    """Join model-specific importances and ranks without creating a composite score."""
    rf = random_forest.rename(columns={"Importance": "Random Forest Importance", "Rank": "Random Forest Rank"})
    xgb = xgboost.rename(columns={"Importance": "XGBoost Importance", "Rank": "XGBoost Rank"})
    return rf.merge(xgb, on="Feature", how="outer").sort_values("Random Forest Rank").reset_index(drop=True)


def aggregate_importance_by_source(importance, feature_mapping, source_column="Source Feature"):
    """Sum encoded-column importance within its source variable.

    Summation preserves each estimator's total importance across one-hot
    children and avoids treating each category dummy as an independent source.
    Importances from different estimators remain separate and are not combined.
    """
    required = {"Feature", "Importance"}
    if not required.issubset(importance.columns):
        raise ValueError(f"importance must contain {sorted(required)}")
    if not {"Encoded Feature", source_column}.issubset(feature_mapping.columns):
        raise ValueError(f"feature_mapping must contain Encoded Feature and {source_column}")
    merged = importance.merge(
        feature_mapping[["Encoded Feature", source_column]],
        left_on="Feature", right_on="Encoded Feature", how="left", validate="one_to_one",
    )
    if merged[source_column].isna().any():
        missing = merged.loc[merged[source_column].isna(), "Feature"].tolist()
        raise ValueError(f"Missing source mapping for encoded features: {missing}")
    result = (merged.groupby(source_column, sort=False)
              .agg(Importance=("Importance", "sum"),
                   **{"Encoded Features": ("Feature", lambda values: list(values))})
              .reset_index().rename(columns={source_column: "Source Feature"})
              .sort_values("Importance", ascending=False, kind="stable").reset_index(drop=True))
    result["Rank"] = np.arange(1, len(result) + 1)
    return result[["Source Feature", "Importance", "Rank", "Encoded Features"]]


def build_encoded_reduction_candidates(importance, feature_counts, feature_order=None):
    """Construct top-ranked encoded subsets in a stable model-column order."""
    ranked = importance.sort_values("Importance", ascending=False, kind="stable")
    ranked_names = ranked["Feature"].tolist()
    names = list(feature_order) if feature_order is not None else ranked_names
    if len(names) != len(ranked_names) or set(names) != set(ranked_names):
        raise ValueError("feature_order must contain every importance feature exactly once.")
    rows = []
    for count in feature_counts:
        if not 1 <= count <= len(names):
            raise ValueError(f"Invalid requested feature count: {count}")
        selected = set(ranked_names[:count])
        retained = [name for name in names if name in selected]
        removed = [name for name in names if name not in selected]
        rows.append({
            "Strategy": "Encoded-level",
            "Requested Feature Count": count,
            "Feature Count": len(retained),
            "Retained Features": retained,
            "Removed Features": removed,
            "Rationale": "Retain highest-ranked individual encoded features by training-only Random Forest importance.",
        })
    return pd.DataFrame(rows)


def build_source_reduction_candidates(source_importance, feature_mapping, target_counts):
    """Remove whole source variables by summed training-only importance.

    For each target, return the largest whole-source subset not exceeding that
    encoded-column count. Source groups can have unequal widths, so exact target
    counts are not always possible without breaking a source variable apart.
    """
    low_to_high = source_importance.sort_values("Importance", ascending=True, kind="stable")["Source Feature"].tolist()
    source_for_encoded = feature_mapping.set_index("Encoded Feature")["Source Feature"].to_dict()
    all_features = feature_mapping["Encoded Feature"].tolist()
    rows = []
    for target in target_counts:
        removed_sources = []
        retained = all_features.copy()
        for source in low_to_high:
            if len(retained) <= target:
                break
            removed_sources.append(source)
            retained = [feature for feature in retained if source_for_encoded[feature] != source]
        removed = [feature for feature in all_features if feature not in set(retained)]
        rows.append({
            "Strategy": "Source-level",
            "Requested Feature Count": target,
            "Feature Count": len(retained),
            "Retained Features": retained,
            "Removed Features": removed,
            "Removed Source Features": removed_sources,
            "Rationale": "Remove lowest-summed-importance source variables as complete groups; actual count reflects group widths.",
        })
    return pd.DataFrame(rows)


def evaluate_reduction_candidates(estimator, X_train, y_train, X_valid, y_valid, candidates):
    """Fit candidate matrices on internal train and return validation metrics."""
    rows = []
    for candidate in candidates:
        features = candidate["Retained Features"]
        fitted = clone(estimator).fit(X_train.loc[:, features], y_train)
        prediction = fitted.predict(X_valid.loc[:, features])
        rows.append({
            "Strategy": candidate["Strategy"],
            "Requested Feature Count": candidate["Requested Feature Count"],
            "Feature Count": candidate["Feature Count"],
            "Validation Accuracy": float(accuracy_score(y_valid, prediction)),
            "Validation Weighted F1": float(f1_score(y_valid, prediction, average="weighted", zero_division=0)),
            "Validation Macro F1": float(f1_score(y_valid, prediction, average="macro", zero_division=0)),
        })
    return pd.DataFrame(rows)


def correlation_matrix(features: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Calculate a numeric feature correlation matrix without selecting features."""
    numeric = features.select_dtypes(include=[np.number])
    return numeric.corr(method=method)


def find_correlated_feature_pairs(
    features: pd.DataFrame,
    threshold: float = 0.95,
    method: str = "pearson",
) -> pd.DataFrame:
    """List highly correlated numeric feature pairs for later review only."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    matrix = correlation_matrix(features, method=method).abs()
    upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    rows = [{"Feature 1": col, "Feature 2": row, "Absolute Correlation": upper.loc[row, col]}
            for col in upper.columns for row in upper.index if pd.notna(upper.loc[row, col]) and upper.loc[row, col] >= threshold]
    return pd.DataFrame(rows, columns=["Feature 1", "Feature 2", "Absolute Correlation"])


def build_rfe_selector(estimator, n_features_to_select="auto", step=1, importance_getter="auto") -> RFE:
    """Construct an RFE selector; fitting and selection remain the caller's choice."""
    return RFE(estimator=estimator, n_features_to_select=n_features_to_select,
               step=step, importance_getter=importance_getter)


def build_sequential_selector(
    estimator, n_features_to_select="auto", direction="backward", scoring=None, cv=5,
    n_jobs=None,
) -> SequentialFeatureSelector:
    """Construct a forward/backward selector for fitting on training data only."""
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'.")
    return SequentialFeatureSelector(
        estimator, n_features_to_select=n_features_to_select, direction=direction,
        scoring=scoring, cv=cv, n_jobs=n_jobs,
    )


def evaluate_feature_subset(estimator, X_train, y_train, X_valid, y_valid, scoring="f1_weighted") -> float:
    """Fit one proposed feature subset on train data and score it on held-out data."""
    fitted = estimator.fit(X_train, y_train)
    return float(get_scorer(scoring)(fitted, X_valid, y_valid))


def backward_elimination(
    estimator,
    X_train,
    y_train,
    X_valid,
    y_valid,
    min_features_to_select=1,
    scoring="f1_weighted",
):
    """Iteratively remove the least important feature and score each subset.

    Callers must pass an internal validation split taken only from training
    data. The function has no final-test parameter. A fresh cloned estimator
    is fit at every feature count, and removal importance is recalculated at
    each step.
    """
    names = list(X_train.columns) if hasattr(X_train, "columns") else list(range(X_train.shape[1]))
    if hasattr(X_valid, "columns"):
        same_columns = list(X_valid.columns) == names
    else:
        same_columns = X_valid.shape[1] == len(names)
    if not same_columns:
        raise ValueError("Training and validation matrices must have identical ordered feature columns.")
    if not 1 <= min_features_to_select <= len(names):
        raise ValueError("min_features_to_select must be between 1 and the initial feature count.")

    scorer = get_scorer(scoring)
    current = names.copy()
    rows = []
    pending_removal = None
    def subset(matrix, features):
        if hasattr(matrix, "loc"):
            return matrix.loc[:, features]
        positions = [names.index(name) for name in features]
        return matrix[:, positions]

    while True:
        X_current = subset(X_train, current)
        X_valid_current = subset(X_valid, current)
        fitted = clone(estimator).fit(X_current, y_train)
        score = float(scorer(fitted, X_valid_current, y_valid))
        if pending_removal is None:
            rows.append({
                "Step": 0,
                "Features Before": len(current),
                "Removed Feature": "",
                "Features After Count": len(current),
                "Features After": current.copy(),
                "Validation Metric": scoring,
                "Validation Score": score,
            })
        else:
            before_count, removed_feature = pending_removal
            rows.append({
                "Step": len(rows),
                "Features Before": before_count,
                "Removed Feature": removed_feature,
                "Features After Count": len(current),
                "Features After": current.copy(),
                "Validation Metric": scoring,
                "Validation Score": score,
            })
            pending_removal = None

        if len(current) <= min_features_to_select:
            break
        importance = extract_feature_importance(fitted, current)
        removed_feature = importance.iloc[-1]["Feature"]
        pending_removal = (len(current), removed_feature)
        current.remove(removed_feature)

    return pd.DataFrame(rows), current
