"""Clean 32-Class Baseline and Grouped+Balanced Classification Pipeline.

This script executes:
PART 1: Clean 32-Class Baseline (All 32 original crime categories preserved without dropping any class).
PART 2: Grouped + Balanced Classification (8 domain-meaningful categories mapped from all 32 original classes, ROS balancing on training set only).
PART 3: Direct comparative evaluation (Accuracy, Weighted F1, Macro F1, training times, devices).
"""
from __future__ import annotations

import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "processed" / "Philadelphia_Crime_Incidents_2018_2022.csv"
GROUPED_DATA_PATH = ROOT / "dataset" / "Philadelphia_Crime_Incidents_2018_2022_Grouped.csv"
MODELS_DIR = ROOT / "models"
BASE32_OUT = ROOT / "results" / "classification_32_baseline"
BASE32_FIGS = BASE32_OUT / "figures"
GRP32_OUT = ROOT / "results" / "classification_grouped_32"
GRP32_FIGS = GRP32_OUT / "figures"

RAW_TARGET = "text_general_code"
GROUPED_TARGET = "crime_group"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CATEGORICAL = ["dc_dist", "psa"]
NUMERIC = ["year", "month", "day", "day_of_week", "hour", "weekend", "lat", "lng"]
FEATURES = CATEGORICAL + NUMERIC

SVM_TRAIN_LIMIT = 5000
KNN_TRAIN_LIMIT = 5000
MLP_TRAIN_LIMIT = 100000
LOGISTIC_TRAIN_LIMIT = 100000

CRIME_GROUP_MAPPING = {
    "Thefts": "Theft & Property Offenses",
    "Theft from Vehicle": "Theft & Property Offenses",
    "Motor Vehicle Theft": "Theft & Property Offenses",
    "Burglary Residential": "Theft & Property Offenses",
    "Burglary Non-Residential": "Theft & Property Offenses",
    "Receiving Stolen Property": "Theft & Property Offenses",

    "Other Assaults": "Violent Crimes",
    "Aggravated Assault No Firearm": "Violent Crimes",
    "Aggravated Assault Firearm": "Violent Crimes",
    "Robbery No Firearm": "Violent Crimes",
    "Robbery Firearm": "Violent Crimes",
    "Homicide - Criminal": "Violent Crimes",
    "Homicide - Justifiable": "Violent Crimes",
    "Homicide - Gross Negligence": "Violent Crimes",

    "Fraud": "Financial & Fraud Offenses",
    "Embezzlement": "Financial & Fraud Offenses",
    "Forgery and Counterfeiting": "Financial & Fraud Offenses",

    "Narcotic / Drug Law Violations": "Substance & Alcohol Offenses",
    "DRIVING UNDER THE INFLUENCE": "Substance & Alcohol Offenses",
    "Liquor Law Violations": "Substance & Alcohol Offenses",
    "Public Drunkenness": "Substance & Alcohol Offenses",

    "Vandalism/Criminal Mischief": "Vandalism & Arson",
    "Arson": "Vandalism & Arson",

    "Rape": "Sexual Offenses",
    "Other Sex Offenses (Not Commercialized)": "Sexual Offenses",

    "Weapon Violations": "Public Order & Weapon Offenses",
    "Disorderly Conduct": "Public Order & Weapon Offenses",
    "Prostitution and Commercialized Vice": "Public Order & Weapon Offenses",
    "Offenses Against Family and Children": "Public Order & Weapon Offenses",
    "Gambling Violations": "Public Order & Weapon Offenses",
    "Vagrancy/Loitering": "Public Order & Weapon Offenses",

    "All Other Offenses": "All Other Offenses",
}


def setup_dirs() -> None:
    for d in (MODELS_DIR, BASE32_OUT, BASE32_FIGS, GRP32_OUT, GRP32_FIGS):
        d.mkdir(parents=True, exist_ok=True)


def nvidia_smi_info() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError as err:
        return f"Unavailable: {err}"


def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    csv_file = GROUPED_DATA_PATH if GROUPED_DATA_PATH.exists() else DATA_PATH
    df = pd.read_csv(
        csv_file,
        usecols=[RAW_TARGET, "dc_dist", "psa", "dispatch_date_time", "dispatch_date", "hour", "lat", "lng"],
        low_memory=False,
    )
    keep = df[RAW_TARGET].notna()
    df = df.loc[keep].copy()

    dt = pd.to_datetime(df["dispatch_date_time"], errors="coerce", format="mixed")
    fallback = pd.to_datetime(df["dispatch_date"], errors="coerce", format="mixed")
    dt = dt.fillna(fallback)

    work = pd.DataFrame(index=df.index)
    work["dc_dist"] = df["dc_dist"]
    work["psa"] = df["psa"]
    work["year"] = dt.dt.year
    work["month"] = dt.dt.month
    work["day"] = dt.dt.day
    work["day_of_week"] = dt.dt.dayofweek
    work["weekend"] = (dt.dt.dayofweek >= 5).astype("float")
    work["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(dt.dt.hour)
    work["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    work["lng"] = pd.to_numeric(df["lng"], errors="coerce")

    raw_target = df[RAW_TARGET].astype(str)
    grouped_target = raw_target.map(CRIME_GROUP_MAPPING)

    return work, raw_target, grouped_target


def get_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


def stratified_subset(X: np.ndarray, y: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= maximum:
        return X, y
    positions = np.arange(len(y))
    counts = np.bincount(y)
    rare = np.isin(y, np.flatnonzero(counts < 2))
    fixed = positions[rare]
    remaining = positions[~rare]
    chosen, _ = train_test_split(
        remaining,
        train_size=maximum - len(fixed),
        random_state=RANDOM_STATE,
        stratify=y[~rare],
    )
    chosen = np.concatenate([fixed, chosen])
    return X[chosen], y[chosen]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion_matrix(name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], normalized: bool, fig_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize="true" if normalized else None)
    fig_size = (18, 14) if len(labels) > 10 else (10, 8)
    plt.figure(figsize=fig_size)
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=len(labels) <= 12,
        fmt=".2f" if normalized else "d",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"{name} {'Normalized ' if normalized else ''}Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(rotation=90 if len(labels) > 10 else 45, ha="right", fontsize=8 if len(labels) > 10 else 9)
    plt.yticks(fontsize=8 if len(labels) > 10 else 9)
    plt.tight_layout()
    kind = "normalized_confusion_matrix" if normalized else "confusion_matrix"
    filename = f"{kind}_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_dir / filename, dpi=200)
    plt.close()


def run_full_pipeline() -> None:
    setup_dirs()
    print("=" * 80)
    print("STARTING 32-CLASS BASELINE & GROUPED+BALANCED EXPERIMENTS")
    print("=" * 80)

    X, raw_target, grouped_target = load_dataset()
    print(f"Dataset loaded: {len(X):,} rows.")

    # Check CUDA GPU for XGBoost
    xgb_base_args = {
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    try:
        probe = XGBClassifier(tree_method="hist", device="cuda", n_estimators=2, random_state=RANDOM_STATE)
        probe.fit(np.random.randn(50, 5), np.random.randint(0, 3, 50), verbose=False)
        del probe
        gpu_available = True
        xgb_device_str = "GPU (CUDA)"
    except Exception as exc:
        xgb_base_args["device"] = "cpu"
        gpu_available = False
        xgb_device_str = f"CPU (CUDA failed: {exc!r})"

    # =========================================================================
    # PART 1: CLEAN 32-CLASS BASELINE EXPERIMENT
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: RUNNING CLEAN 32-CLASS BASELINE (NO CLASSES DROPPED)")
    print("=" * 80)

    encoder_32 = LabelEncoder()
    y_32 = encoder_32.fit_transform(raw_target)
    n_classes_32 = len(encoder_32.classes_)
    print(f"32-Class Baseline Target: {n_classes_32} unique classes.")

    # Stratify split preserving singletons
    counts_32 = np.bincount(y_32)
    singletons = np.flatnonzero(counts_32 < 2)
    is_singleton = np.isin(y_32, singletons)
    idx_singletons = np.flatnonzero(is_singleton)
    idx_regular = np.flatnonzero(~is_singleton)

    train_reg, test_reg = train_test_split(
        idx_regular, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_32[idx_regular]
    )
    train_idx_32 = np.concatenate([idx_singletons, train_reg])
    test_idx_32 = test_reg

    X_train_32, y_train_32 = X.iloc[train_idx_32], y_32[train_idx_32]
    X_test_32, y_test_32 = X.iloc[test_idx_32], y_32[test_idx_32]

    print(f"32-Class Split: {len(y_train_32):,} Train / {len(y_test_32):,} Test.")

    tree_prep_32 = get_preprocessor(scale=False)
    scaled_prep_32 = get_preprocessor(scale=True)

    tree_tr_32 = tree_prep_32.fit_transform(X_train_32)
    tree_te_32 = tree_prep_32.transform(X_test_32)
    scaled_tr_32 = scaled_prep_32.fit_transform(X_train_32)
    scaled_te_32 = scaled_prep_32.transform(X_test_32)

    dump(tree_prep_32, MODELS_DIR / "baseline32_classification_tree_preprocessor.joblib")
    dump(scaled_prep_32, MODELS_DIR / "baseline32_classification_preprocessor.joblib")
    dump(encoder_32, MODELS_DIR / "baseline32_classification_label_encoder.joblib")

    base32_results = []
    base32_reports = []

    def train_and_eval_32(name: str, build_fn, tr_X, tr_y, te_X, te_y, model_file: Path, dev_str: str, sub_lim: int | None = None):
        t0 = time.perf_counter()
        fit_X, fit_y = stratified_subset(tr_X, tr_y, sub_lim) if sub_lim else (tr_X, tr_y)
        print(f"Training Baseline 32-Class {name} on {len(fit_y):,} rows ({dev_str})...", flush=True)
        clf = build_fn()
        clf.fit(fit_X, fit_y)
        pred = clf.predict(te_X)
        res = compute_metrics(te_y, pred)
        elapsed = time.perf_counter() - t0

        plot_confusion_matrix(name, te_y, pred, encoder_32.classes_, False, BASE32_FIGS)
        plot_confusion_matrix(name, te_y, pred, encoder_32.classes_, True, BASE32_FIGS)

        dump(clf, model_file)
        if name == "XGBoost":
            clf.save_model(str(model_file.with_suffix(".json")))

        base32_reports.append(
            f"\n{'='*80}\n32-CLASS BASELINE: {name}\n"
            + classification_report(te_y, pred, labels=np.arange(len(encoder_32.classes_)), target_names=encoder_32.classes_, zero_division=0)
        )
        base32_results.append({
            "Model": name,
            "Experiment": "32-Class Baseline",
            "Number of Classes": n_classes_32,
            **res,
            "Training Time (s)": elapsed,
            "Device": dev_str,
            "Training Sample Count": len(fit_y),
        })

    # XGBoost 32
    xgb_32_kwargs = {**xgb_base_args, "objective": "multi:softprob", "num_class": n_classes_32}
    train_and_eval_32(
        "XGBoost",
        lambda: XGBClassifier(**xgb_32_kwargs),
        tree_tr_32,
        y_train_32,
        tree_te_32,
        y_test_32,
        MODELS_DIR / "baseline32_xgboost_model.joblib",
        xgb_device_str,
    )

    # Random Forest 32
    train_and_eval_32(
        "Random Forest",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        tree_tr_32,
        y_train_32,
        tree_te_32,
        y_test_32,
        MODELS_DIR / "baseline32_random_forest_model.joblib",
        "CPU",
    )

    # SVM 32
    train_and_eval_32(
        "SVM",
        lambda: SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1024, random_state=RANDOM_STATE),
        scaled_tr_32,
        y_train_32,
        scaled_te_32,
        y_test_32,
        MODELS_DIR / "baseline32_svm_model.joblib",
        "CPU (Subset 5k)",
        SVM_TRAIN_LIMIT,
    )

    # MLP 32
    train_and_eval_32(
        "MLP",
        lambda: MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=0.1, max_iter=30, random_state=RANDOM_STATE, batch_size=512),
        scaled_tr_32,
        y_train_32,
        scaled_te_32,
        y_test_32,
        MODELS_DIR / "baseline32_mlp_model.joblib",
        "CPU (Subset 100k)",
        MLP_TRAIN_LIMIT,
    )

    # KNN 32
    train_and_eval_32(
        "KNN",
        lambda: KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute", n_jobs=-1),
        scaled_tr_32,
        y_train_32,
        scaled_te_32,
        y_test_32,
        MODELS_DIR / "baseline32_knn_model.joblib",
        "CPU (Subset 5k)",
        KNN_TRAIN_LIMIT,
    )

    # Logistic Regression 32
    train_and_eval_32(
        "Logistic Regression",
        lambda: LogisticRegression(solver="saga", multi_class="multinomial", max_iter=30, tol=1e-3, random_state=RANDOM_STATE, n_jobs=-1),
        scaled_tr_32,
        y_train_32,
        scaled_te_32,
        y_test_32,
        MODELS_DIR / "baseline32_logistic_regression_model.joblib",
        "CPU (Subset 100k)",
        LOGISTIC_TRAIN_LIMIT,
    )

    base32_df = pd.DataFrame(base32_results)
    base32_df.to_csv(BASE32_OUT / "baseline32_model_comparison.csv", index=False)
    (BASE32_OUT / "baseline32_classification_reports.txt").write_text("\n".join(base32_reports), encoding="utf-8")
    print("\nCompleted 32-Class Baseline Training.")

    # =========================================================================
    # PART 2: GROUPED + BALANCED EXPERIMENT (8 CLASSES)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: RUNNING GROUPED + BALANCED EXPERIMENT (8 CLASSES)")
    print("=" * 80)

    encoder_grp = LabelEncoder()
    y_grp = encoder_grp.fit_transform(grouped_target)
    n_classes_grp = len(encoder_grp.classes_)

    # 80/20 Stratified Split on crime_group FIRST
    X_train_grp, X_test_grp, y_train_grp, y_test_grp = train_test_split(
        X, y_grp, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_grp
    )

    # Save train distributions before balance and test distribution
    train_before_counts = np.bincount(y_train_grp, minlength=n_classes_grp)
    test_counts = np.bincount(y_test_grp, minlength=n_classes_grp)

    pd.DataFrame({
        "crime_group": encoder_grp.classes_,
        "train_count": train_before_counts,
        "percentage": (train_before_counts / len(y_train_grp)) * 100,
    }).to_csv(GRP32_OUT / "train_distribution_before_balance.csv", index=False)

    pd.DataFrame({
        "crime_group": encoder_grp.classes_,
        "test_count": test_counts,
        "percentage": (test_counts / len(y_test_grp)) * 100,
    }).to_csv(GRP32_OUT / "test_distribution.csv", index=False)

    # Fit preprocessor strictly on X_train_grp
    tree_prep_grp = get_preprocessor(scale=False)
    scaled_prep_grp = get_preprocessor(scale=True)

    tree_tr_grp = tree_prep_grp.fit_transform(X_train_grp)
    tree_te_grp = tree_prep_grp.transform(X_test_grp)

    scaled_tr_grp = scaled_prep_grp.fit_transform(X_train_grp)
    scaled_te_grp = scaled_prep_grp.transform(X_test_grp)

    dump(tree_prep_grp, MODELS_DIR / "grouped32_classification_tree_preprocessor.joblib")
    dump(scaled_prep_grp, MODELS_DIR / "grouped32_classification_preprocessor.joblib")
    dump(encoder_grp, MODELS_DIR / "grouped32_classification_label_encoder.joblib")

    # Balance TRAINING set ONLY using RandomOverSampler
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    tree_tr_res, y_tr_res = ros.fit_resample(tree_tr_grp, y_train_grp)
    scaled_tr_res, _ = ros.fit_resample(scaled_tr_grp, y_train_grp)

    train_after_counts = np.bincount(y_tr_res, minlength=n_classes_grp)
    pd.DataFrame({
        "crime_group": encoder_grp.classes_,
        "balanced_train_count": train_after_counts,
        "percentage": (train_after_counts / len(y_tr_res)) * 100,
    }).to_csv(GRP32_OUT / "train_distribution_after_balance.csv", index=False)

    print(f"Grouped Split: {len(y_train_grp):,} -> {len(y_tr_res):,} rows after RandomOverSampler.")
    print(f"Test Set (UNTOUCHED): {len(y_test_grp):,} rows.")

    grp32_results = []
    grp32_reports = []

    def train_and_eval_grp(name: str, build_fn, tr_X, tr_y, te_X, te_y, model_file: Path, dev_str: str, sub_lim: int | None = None):
        t0 = time.perf_counter()
        fit_X, fit_y = stratified_subset(tr_X, tr_y, sub_lim) if sub_lim else (tr_X, tr_y)
        print(f"Training Grouped {name} on {len(fit_y):,} rows ({dev_str})...", flush=True)
        clf = build_fn()
        clf.fit(fit_X, fit_y)
        pred = clf.predict(te_X)
        res = compute_metrics(te_y, pred)
        elapsed = time.perf_counter() - t0

        plot_confusion_matrix(name, te_y, pred, encoder_grp.classes_, False, GRP32_FIGS)
        plot_confusion_matrix(name, te_y, pred, encoder_grp.classes_, True, GRP32_FIGS)

        dump(clf, model_file)
        if name == "XGBoost":
            clf.save_model(str(model_file.with_suffix(".json")))
            imp = pd.DataFrame({
                "feature": tree_prep_grp.get_feature_names_out(),
                "importance": clf.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp.to_csv(GRP32_FIGS / "feature_importance_xgboost.csv", index=False)
        elif name == "Random Forest":
            imp = pd.DataFrame({
                "feature": tree_prep_grp.get_feature_names_out(),
                "importance": clf.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp.to_csv(GRP32_FIGS / "feature_importance_random_forest.csv", index=False)

        grp32_reports.append(
            f"\n{'='*80}\nGROUPED + BALANCED: {name}\n"
            + classification_report(te_y, pred, labels=np.arange(len(encoder_grp.classes_)), target_names=encoder_grp.classes_, zero_division=0)
        )
        grp32_results.append({
            "Model": name,
            "Experiment": "Grouped + Balanced (8-Class)",
            "Number of Classes": n_classes_grp,
            **res,
            "Training Time (s)": elapsed,
            "Device": dev_str,
            "Training Sample Count": len(fit_y),
        })

    # XGBoost Grouped
    xgb_grp_kwargs = {**xgb_base_args, "objective": "multi:softprob", "num_class": n_classes_grp}
    train_and_eval_grp(
        "XGBoost",
        lambda: XGBClassifier(**xgb_grp_kwargs),
        tree_tr_res,
        y_tr_res,
        tree_te_grp,
        y_test_grp,
        MODELS_DIR / "grouped32_xgboost_model.joblib",
        xgb_device_str,
    )

    # Random Forest Grouped
    train_and_eval_grp(
        "Random Forest",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        tree_tr_res,
        y_tr_res,
        tree_te_grp,
        y_test_grp,
        MODELS_DIR / "grouped32_random_forest_model.joblib",
        "CPU",
    )

    # SVM Grouped
    train_and_eval_grp(
        "SVM",
        lambda: SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1024, random_state=RANDOM_STATE),
        scaled_tr_res,
        y_tr_res,
        scaled_te_grp,
        y_test_grp,
        MODELS_DIR / "grouped32_svm_model.joblib",
        "CPU (Subset 5k)",
        SVM_TRAIN_LIMIT,
    )

    # MLP Grouped
    train_and_eval_grp(
        "MLP",
        lambda: MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=0.1, max_iter=30, random_state=RANDOM_STATE, batch_size=512),
        scaled_tr_res,
        y_tr_res,
        scaled_te_grp,
        y_test_grp,
        MODELS_DIR / "grouped32_mlp_model.joblib",
        "CPU (Subset 100k)",
        MLP_TRAIN_LIMIT,
    )

    # KNN Grouped
    train_and_eval_grp(
        "KNN",
        lambda: KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute", n_jobs=-1),
        scaled_tr_res,
        y_tr_res,
        scaled_te_grp,
        y_test_grp,
        MODELS_DIR / "grouped32_knn_model.joblib",
        "CPU (Subset 5k)",
        KNN_TRAIN_LIMIT,
    )

    # Logistic Regression Grouped
    train_and_eval_grp(
        "Logistic Regression",
        lambda: LogisticRegression(solver="saga", multi_class="multinomial", max_iter=30, tol=1e-3, random_state=RANDOM_STATE, n_jobs=-1),
        scaled_tr_res,
        y_tr_res,
        scaled_te_grp,
        y_test_grp,
        MODELS_DIR / "grouped32_logistic_regression_model.joblib",
        "CPU (Subset 100k)",
        LOGISTIC_TRAIN_LIMIT,
    )

    grp32_df = pd.DataFrame(grp32_results)
    (GRP32_OUT / "grouped_classification_reports.txt").write_text("\n".join(grp32_reports), encoding="utf-8")

    # =========================================================================
    # PART 3: DIRECT COMPARISON TABLE & SUMMARY
    # =========================================================================
    comparison_rows = []
    for model_name in ["XGBoost", "Random Forest", "SVM", "MLP", "KNN", "Logistic Regression"]:
        b_r = base32_df[base32_df["Model"] == model_name].iloc[0]
        g_r = grp32_df[grp32_df["Model"] == model_name].iloc[0]
        comparison_rows.append(b_r.to_dict())
        comparison_rows.append(g_r.to_dict())

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(GRP32_OUT / "grouped32_model_comparison.csv", index=False)

    pd.DataFrame({
        "feature": FEATURES,
        "role": ["categorical"] * len(CATEGORICAL) + ["numeric"] * len(NUMERIC),
        "missing_values_before_train_imputation": X[FEATURES].isna().sum().values,
    }).to_csv(GRP32_OUT / "feature_list.csv", index=False)

    env_info = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "dataset_rows": len(X),
        "original_classes_count": n_classes_32,
        "grouped_classes_count": n_classes_grp,
        "features": FEATURES,
        "train_size_unbalanced": len(y_train_grp),
        "train_size_balanced": len(y_tr_res),
        "test_size": len(y_test_grp),
        "random_state": RANDOM_STATE,
        "resampling_method": "RandomOverSampler (Training set only)",
        "nvidia_smi": nvidia_smi_info(),
    }
    (GRP32_OUT / "environment.txt").write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    summary_text = [
        "================================================================================",
        "32-CLASS BASELINE VS GROUPED + BALANCED CLASSIFICATION SUMMARY",
        "================================================================================",
        f"Original Classes: {n_classes_32} | Grouped Categories: {n_classes_grp}",
        f"Dataset Rows: {len(X):,} | Train/Test Split: 80/20 Stratified",
        f"Training set: {len(y_train_grp):,} -> {len(y_tr_res):,} rows (RandomOverSampler)",
        f"Test set (UNTOUCHED): {len(y_test_grp):,}",
        f"XGBoost Device: {xgb_device_str}",
        "",
        "COMPARATIVE RESULTS TABLE:",
        comparison_df[["Model", "Experiment", "Number of Classes", "Accuracy", "Weighted F1", "Macro F1", "Training Time (s)", "Device"]].to_string(index=False),
        "",
        "METHODOLOGICAL INTEGRITY CHECK:",
        "1. Start point: Original 32-class Philadelphia dataset (0 classes dropped).",
        "2. Train/Test Split: Stratified BEFORE balancing or scaling.",
        "3. Test set: UNTOUCHED (0 test rows were balanced, oversampled, or fitted).",
        "4. Preprocessing: Fitted strictly on X_train.",
        "5. Comparison: Validated against clean 32-class baseline.",
        "================================================================================",
    ]
    (GRP32_OUT / "summary.txt").write_text("\n".join(summary_text), encoding="utf-8")

    print("\n" + "=" * 80)
    print("32-CLASS BASELINE VS GROUPED + BALANCED RESULTS")
    print("=" * 80)
    print(comparison_df[["Model", "Experiment", "Number of Classes", "Accuracy", "Weighted F1", "Macro F1", "Training Time (s)", "Device"]].to_string(index=False))
    print(f"\nPipeline complete! Results saved in:\n- {BASE32_OUT}\n- {GRP32_OUT}")


if __name__ == "__main__":
    run_full_pipeline()
