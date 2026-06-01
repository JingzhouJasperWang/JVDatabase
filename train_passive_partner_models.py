#!/usr/bin/env python3
"""Train passive-partner classifiers from the two prepared model workbooks.

The script intentionally uses a strict feature set: IDs, source fields,
derived labels/outcome evidence, and near-label text such as role evidence are
excluded by default. Outputs are written under results/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

try:
    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        make_scorer,
        precision_score,
        precision_recall_curve,
        recall_score,
        roc_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    missing = exc.name or "a required Python package"
    raise SystemExit(
        f"Missing dependency: {missing}. Install pandas, numpy, openpyxl, "
        "scikit-learn, matplotlib, and joblib, then rerun this script."
    ) from exc


EVENT_FILE = "passive_partner_event_level_model_data.xlsx"
PARTNER_FILE = "passive_partner_partner_level_model_data.xlsx"
TRAINING_SHEET = "Training_Eligible"


NUMERIC_FEATURES = {
    "announcement_year",
    "cross_border_binary",
    "partner_count",
    "deal_or_investment_value",
    "deal_value_reported_flag",
    "ownership_equal_50_50_flag",
    "ownership_pct_numeric",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    filename: str
    target: str
    id_columns: tuple[str, ...]
    features: tuple[str, ...]


DATASETS = (
    DatasetConfig(
        name="event_level",
        filename=EVENT_FILE,
        target="label_event_has_passive_financial_partner",
        id_columns=("event_id", "jv_name"),
        features=(
            "announcement_year",
            "date_precision",
            "sector",
            "subsector",
            "geography",
            "cross_border",
            "cross_border_binary",
            "partner_count",
            "deal_or_investment_value",
            "currency",
            "deal_value_reported_flag",
            "ownership_equal_50_50_flag",
            "equity_JV_flag",
        ),
    ),
    DatasetConfig(
        name="partner_level",
        filename=PARTNER_FILE,
        target="label_passive_financial",
        id_columns=("row_id", "event_id", "jv_name", "partner_name"),
        features=(
            "partner_country_or_region",
            "announcement_year",
            "date_precision",
            "sector",
            "subsector",
            "geography",
            "cross_border",
            "cross_border_binary",
            "partner_count",
            "deal_or_investment_value",
            "currency",
            "deal_value_reported_flag",
            "equity_JV_flag",
            "ownership_equal_50_50_flag",
            "ownership_pct_numeric",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML models for passive-financial-partner prediction."
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        type=Path,
        help="Directory containing the two input .xlsx files.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=Path,
        help="Directory where reports, predictions, and models are saved.",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed for splits, models, and permutation importance.",
    )
    parser.add_argument(
        "--test-size",
        default=0.25,
        type=float,
        help="Stratified holdout fraction.",
    )
    parser.add_argument(
        "--cv-folds",
        default=5,
        type=int,
        help="Maximum number of stratified cross-validation folds.",
    )
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs = {
        "handle_unknown": "ignore",
        "min_frequency": 2,
    }
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def prepare_xy(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    filtered = df.copy()
    if "training_eligible" in filtered.columns:
        eligible = filtered["training_eligible"].astype(str).str.strip().str.lower().eq("yes")
        filtered = filtered.loc[eligible].copy()

    filtered = filtered.loc[filtered[config.target].notna()].copy()
    filtered[config.target] = pd.to_numeric(filtered[config.target], errors="coerce")
    filtered = filtered.loc[filtered[config.target].notna()].copy()

    feature_cols = [col for col in config.features if col in filtered.columns]
    missing = sorted(set(config.features) - set(feature_cols))
    if missing:
        warnings.warn(f"{config.name}: missing configured features: {missing}", stacklevel=2)

    X = filtered[feature_cols].copy()
    for col in feature_cols:
        if col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].astype(object).where(pd.notna(X[col]), np.nan)

    y = filtered[config.target].astype(int)
    return X, y, feature_cols


def split_column_types(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric_cols = [col for col in feature_cols if col in NUMERIC_FEATURES]
    categorical_cols = [col for col in feature_cols if col not in NUMERIC_FEATURES]
    return numeric_cols, categorical_cols


def make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_column_types(feature_cols)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def make_models(random_state: int) -> OrderedDict[str, Any]:
    return OrderedDict(
        [
            ("dummy_prior", DummyClassifier(strategy="prior")),
            (
                "logistic_regression_balanced",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
            (
                "random_forest_balanced",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
            (
                "extra_trees_balanced",
                ExtraTreesClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
            (
                "hist_gradient_boosting_balanced",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    learning_rate=0.05,
                    l2_regularization=0.01,
                    max_iter=200,
                    min_samples_leaf=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_pipeline(feature_cols: list[str], model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(feature_cols)),
            ("model", model),
        ]
    )


def make_scorers() -> dict[str, Any]:
    return {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }


def get_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    model = estimator.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = estimator.decision_function(X)
        raw_min = np.min(raw)
        raw_max = np.max(raw)
        if raw_max == raw_min:
            return np.full_like(raw, 0.5, dtype=float)
        return (raw - raw_min) / (raw_max - raw_min)
    return estimator.predict(X).astype(float)


def choose_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), scores]))
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        key = (
            f1_score(y_true, preds, zero_division=0),
            balanced_accuracy_score(y_true, preds),
            recall_score(y_true, preds, zero_division=0),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def evaluate_predictions(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=labels).ravel()
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "brier_loss": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan
    return result


def evaluate_model_cv(
    name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    cv: StratifiedKFold,
) -> dict[str, Any]:
    estimator = make_pipeline(feature_cols, model)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=make_scorers(),
        n_jobs=None,
        error_score="raise",
        return_train_score=False,
    )
    result: dict[str, Any] = {"model": name}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric = key.removeprefix("test_")
        result[f"cv_{metric}_mean"] = float(np.mean(values))
        result[f"cv_{metric}_std"] = float(np.std(values, ddof=0))
    return result


def evaluate_model_holdout(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: list[str],
) -> tuple[dict[str, Any], Pipeline, pd.DataFrame]:
    estimator = make_pipeline(feature_cols, clone(model))
    estimator.fit(X_train, y_train)
    train_scores = get_scores(estimator, X_train)
    threshold = choose_threshold(y_train, train_scores)
    test_scores = get_scores(estimator, X_test)
    metrics = evaluate_predictions(y_test, test_scores, threshold)
    metrics["model"] = name
    pred_df = pd.DataFrame(
        {
            "model": name,
            "y_true": y_test.to_numpy(),
            "score_passive_financial": test_scores,
            "predicted_label": (test_scores >= threshold).astype(int),
            "threshold": threshold,
        },
        index=X_test.index,
    )
    return metrics, estimator, pred_df


def select_best_model(cv_results: pd.DataFrame) -> str:
    candidates = cv_results.loc[cv_results["model"] != "dummy_prior"].copy()
    if candidates.empty:
        candidates = cv_results.copy()
    candidates = candidates.sort_values(
        by=["cv_average_precision_mean", "cv_f1_mean", "cv_roc_auc_mean"],
        ascending=[False, False, False],
    )
    return str(candidates.iloc[0]["model"])


def save_permutation_importance(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    random_state: int,
) -> None:
    if len(np.unique(y_test)) < 2:
        pd.DataFrame(columns=["feature", "importance_mean", "importance_std"]).to_csv(
            output_path, index=False
        )
        return

    importance = permutation_importance(
        estimator,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=30,
        random_state=random_state,
        n_jobs=1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(output_path, index=False)


def display_name(name: str) -> str:
    return name.replace("_", " ").replace("balanced", "bal.").title()


def style_axis(ax: Any, y_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)


def save_figure(fig: Any, output_path: Path) -> str:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_class_balance(
    y: pd.Series,
    config: DatasetConfig,
    graph_dir: Path,
) -> str:
    output_path = graph_dir / f"{config.name}_class_balance.png"
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    labels = ["Negative", "Passive/Financial"]
    colors = ["#6C757D", "#D1495B"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, counts.to_numpy(), color=colors)
    ax.set_title(f"{display_name(config.name)} Class Balance")
    ax.set_ylabel("Rows")
    ax.set_ylim(0, max(counts.max() * 1.18, 1))
    style_axis(ax)

    total = counts.sum()
    for bar, value in zip(bars, counts.to_numpy()):
        pct = value / total if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value)}\n{pct:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    return save_figure(fig, output_path)


def plot_cv_metrics(cv_df: pd.DataFrame, config: DatasetConfig, graph_dir: Path) -> str:
    output_path = graph_dir / f"{config.name}_cv_model_comparison.png"
    metrics = [
        ("cv_average_precision_mean", "Avg Precision"),
        ("cv_roc_auc_mean", "ROC AUC"),
        ("cv_f1_mean", "F1"),
    ]
    plot_df = cv_df.sort_values("cv_average_precision_mean", ascending=False).copy()
    model_labels = [display_name(name) for name in plot_df["model"]]
    x = np.arange(len(plot_df))
    width = 0.24
    colors = ["#1F77B4", "#2CA02C", "#FF7F0E"]

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for idx, (column, label) in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, plot_df[column], width, label=label, color=colors[idx])
    ax.set_title(f"{display_name(config.name)} Cross-Validation Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=25, ha="right")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    style_axis(ax)
    return save_figure(fig, output_path)


def plot_holdout_metrics(holdout_df: pd.DataFrame, config: DatasetConfig, graph_dir: Path) -> str:
    output_path = graph_dir / f"{config.name}_holdout_model_comparison.png"
    metrics = [
        ("average_precision", "Avg Precision"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("balanced_accuracy", "Balanced Acc."),
    ]
    plot_df = holdout_df.sort_values("average_precision", ascending=False).copy()
    model_labels = [display_name(name) for name in plot_df["model"]]
    x = np.arange(len(plot_df))
    width = 0.16
    colors = ["#1F77B4", "#9467BD", "#2CA02C", "#FF7F0E", "#8C564B"]

    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    offsets = np.linspace(-2 * width, 2 * width, len(metrics))
    for offset, (column, label), color in zip(offsets, metrics, colors):
        ax.bar(x + offset, plot_df[column], width, label=label, color=color)
    ax.set_title(f"{display_name(config.name)} Holdout Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=25, ha="right")
    ax.legend(ncol=5, frameon=False, loc="upper right")
    style_axis(ax)
    return save_figure(fig, output_path)


def plot_precision_recall_curves(
    predictions_df: pd.DataFrame,
    config: DatasetConfig,
    graph_dir: Path,
) -> str:
    output_path = graph_dir / f"{config.name}_precision_recall_curve.png"
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    y_true = predictions_df["y_true"].astype(int)
    baseline = float(y_true.mean())

    for model_name, model_preds in predictions_df.groupby("model", sort=False):
        precision, recall, _ = precision_recall_curve(
            model_preds["y_true"].astype(int),
            model_preds["score_passive_financial"].astype(float),
        )
        ap = average_precision_score(
            model_preds["y_true"].astype(int),
            model_preds["score_passive_financial"].astype(float),
        )
        ax.plot(recall, precision, linewidth=2, label=f"{display_name(model_name)} AP={ap:.2f}")

    ax.axhline(baseline, color="#6C757D", linestyle="--", linewidth=1.5, label=f"Baseline={baseline:.2f}")
    ax.set_title(f"{display_name(config.name)} Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="best")
    style_axis(ax)
    return save_figure(fig, output_path)


def plot_roc_curves(predictions_df: pd.DataFrame, config: DatasetConfig, graph_dir: Path) -> str:
    output_path = graph_dir / f"{config.name}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    for model_name, model_preds in predictions_df.groupby("model", sort=False):
        y_true = model_preds["y_true"].astype(int)
        scores = model_preds["score_passive_financial"].astype(float)
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{display_name(model_name)} AUC={auc:.2f}")

    ax.plot([0, 1], [0, 1], color="#6C757D", linestyle="--", linewidth=1.2, label="Random")
    ax.set_title(f"{display_name(config.name)} ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(ax)
    return save_figure(fig, output_path)


def plot_confusion_matrix(
    predictions_df: pd.DataFrame,
    best_model_name: str,
    config: DatasetConfig,
    graph_dir: Path,
) -> str:
    output_path = graph_dir / f"{config.name}_best_model_confusion_matrix.png"
    best_preds = predictions_df.loc[predictions_df["model"] == best_model_name].copy()
    cm = confusion_matrix(
        best_preds["y_true"].astype(int),
        best_preds["predicted_label"].astype(int),
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{display_name(config.name)} Best Model Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Passive"])
    ax.set_yticklabels(["Negative", "Passive"])

    max_value = cm.max() if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > max_value / 2 else "#1F2933"
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color=color, fontsize=13)
    return save_figure(fig, output_path)


def plot_feature_importance(
    importance_path: Path,
    config: DatasetConfig,
    graph_dir: Path,
    top_n: int = 12,
) -> str:
    output_path = graph_dir / f"{config.name}_feature_importance.png"
    imp_df = pd.read_csv(importance_path)
    if imp_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        ax.axis("off")
        return save_figure(fig, output_path)

    plot_df = imp_df.head(top_n).iloc[::-1]
    colors = ["#D1495B" if value > 0 else "#6C757D" for value in plot_df["importance_mean"]]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        color=colors,
        alpha=0.9,
    )
    ax.axvline(0, color="#343A40", linewidth=1)
    ax.set_title(f"{display_name(config.name)} Permutation Importance")
    ax.set_xlabel("Average precision decrease when permuted")
    style_axis(ax, y_grid=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    return save_figure(fig, output_path)


def make_dataset_graphs(
    config: DatasetConfig,
    y: pd.Series,
    cv_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    importance_path: Path,
    best_model_name: str,
    graph_dir: Path,
) -> list[str]:
    graph_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_class_balance(y, config, graph_dir),
        plot_cv_metrics(cv_df, config, graph_dir),
        plot_holdout_metrics(holdout_df, config, graph_dir),
        plot_precision_recall_curves(predictions_df, config, graph_dir),
        plot_roc_curves(predictions_df, config, graph_dir),
        plot_confusion_matrix(predictions_df, best_model_name, config, graph_dir),
        plot_feature_importance(importance_path, config, graph_dir),
    ]


def plot_overall_summary(summaries: list[dict[str, Any]], graph_dir: Path) -> str:
    graph_dir.mkdir(parents=True, exist_ok=True)
    output_path = graph_dir / "overall_best_model_summary.png"
    summary_df = pd.DataFrame(summaries)
    dataset_labels = [display_name(name) for name in summary_df["dataset"]]
    x = np.arange(len(summary_df))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    axes[0].bar(dataset_labels, summary_df["positive_rate"], color=["#D1495B", "#1F77B4"])
    axes[0].set_title("Positive Class Rate")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, max(summary_df["positive_rate"].max() * 1.3, 0.1))
    style_axis(axes[0])
    for idx, value in enumerate(summary_df["positive_rate"]):
        axes[0].text(idx, value, f"{value:.1%}", ha="center", va="bottom")

    metrics = [
        ("best_cv_average_precision_mean", "CV AP"),
        ("best_holdout_average_precision", "Holdout AP"),
        ("best_holdout_f1", "Holdout F1"),
    ]
    width = 0.22
    colors = ["#1F77B4", "#2CA02C", "#FF7F0E"]
    for offset, (column, label), color in zip([-width, 0, width], metrics, colors):
        axes[1].bar(x + offset, summary_df[column], width, label=label, color=color)
    axes[1].set_title("Best Model Metrics")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dataset_labels)
    axes[1].legend(frameon=False, loc="upper right")
    style_axis(axes[1])
    return save_figure(fig, output_path)


def dataset_label(config: DatasetConfig) -> str:
    return config.name.replace("_", " ")


def train_dataset(
    config: DatasetConfig,
    data_dir: Path,
    results_dir: Path,
    random_state: int,
    test_size: float,
    max_cv_folds: int,
) -> dict[str, Any]:
    input_path = data_dir / config.filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_excel(input_path, sheet_name=TRAINING_SHEET)
    X, y, feature_cols = prepare_xy(raw_df, config)

    if len(np.unique(y)) != 2:
        raise ValueError(f"{config.name}: target must contain both classes.")

    min_class_count = int(y.value_counts().min())
    cv_folds = max(2, min(max_cv_folds, min_class_count))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    models = make_models(random_state)
    cv_rows = []
    holdout_rows = []
    prediction_rows = []
    holdout_estimators: dict[str, Pipeline] = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    id_cols = [col for col in config.id_columns if col in raw_df.columns]
    id_frame = raw_df.loc[X_test.index, id_cols].copy() if id_cols else pd.DataFrame(index=X_test.index)

    for model_name, model in models.items():
        cv_rows.append(evaluate_model_cv(model_name, clone(model), X, y, feature_cols, cv))
        holdout_metrics, holdout_estimator, pred_df = evaluate_model_holdout(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            feature_cols,
        )
        holdout_rows.append(holdout_metrics)
        pred_with_ids = pd.concat([id_frame.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        prediction_rows.append(pred_with_ids)
        holdout_estimators[model_name] = holdout_estimator

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_average_precision_mean", ascending=False)
    holdout_df = pd.DataFrame(holdout_rows).sort_values("average_precision", ascending=False)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    best_model_name = select_best_model(cv_df)

    full_estimator = make_pipeline(feature_cols, clone(models[best_model_name]))
    full_estimator.fit(X, y)
    final_model_path = results_dir / f"{config.name}_best_model.joblib"
    joblib.dump(
        {
            "dataset": config.name,
            "target": config.target,
            "features": feature_cols,
            "model_name": best_model_name,
            "estimator": full_estimator,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        final_model_path,
    )

    cv_path = results_dir / f"{config.name}_cv_metrics.csv"
    holdout_path = results_dir / f"{config.name}_holdout_metrics.csv"
    pred_path = results_dir / f"{config.name}_holdout_predictions.csv"
    importance_path = results_dir / f"{config.name}_feature_importance.csv"

    cv_df.to_csv(cv_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    predictions_df.to_csv(pred_path, index=False)
    save_permutation_importance(
        holdout_estimators[best_model_name],
        X_test,
        y_test,
        importance_path,
        random_state,
    )
    graph_files = make_dataset_graphs(
        config=config,
        y=y,
        cv_df=cv_df,
        holdout_df=holdout_df,
        predictions_df=predictions_df,
        importance_path=importance_path,
        best_model_name=best_model_name,
        graph_dir=results_dir / "graphs",
    )

    best_cv = cv_df.loc[cv_df["model"] == best_model_name].iloc[0].to_dict()
    best_holdout = holdout_df.loc[holdout_df["model"] == best_model_name].iloc[0].to_dict()

    return {
        "dataset": config.name,
        "input_file": str(input_path),
        "rows": int(len(X)),
        "features": feature_cols,
        "target": config.target,
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "cv_folds": int(cv_folds),
        "best_model": best_model_name,
        "best_cv_average_precision_mean": float(best_cv["cv_average_precision_mean"]),
        "best_cv_f1_mean": float(best_cv["cv_f1_mean"]),
        "best_cv_roc_auc_mean": float(best_cv["cv_roc_auc_mean"]),
        "best_holdout_average_precision": float(best_holdout["average_precision"]),
        "best_holdout_f1": float(best_holdout["f1"]),
        "best_holdout_precision": float(best_holdout["precision"]),
        "best_holdout_recall": float(best_holdout["recall"]),
        "best_holdout_balanced_accuracy": float(best_holdout["balanced_accuracy"]),
        "best_holdout_threshold": float(best_holdout["threshold"]),
        "cv_metrics_file": str(cv_path),
        "holdout_metrics_file": str(holdout_path),
        "holdout_predictions_file": str(pred_path),
        "feature_importance_file": str(importance_path),
        "best_model_file": str(final_model_path),
        "graph_files": graph_files,
    }


def write_readme(results_dir: Path, summaries: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    graph_files: list[str] = []
    if metadata.get("summary_graph_file"):
        graph_files.append(str(metadata["summary_graph_file"]))
    for item in summaries:
        graph_files.extend(item.get("graph_files", []))

    lines = [
        "# Passive Partner ML Results",
        "",
        f"Generated UTC: {metadata['run_started_utc']}",
        "",
        "## Modeling choices",
        "",
        "- Data source: Training_Eligible sheet from each passive partner model workbook.",
        "- Target: event-level passive-financial-partner flag and partner-level passive-financial flag.",
        "- Feature policy: strict tabular features only; IDs, URLs, source fields, derived outcome evidence, and near-label text fields are excluded.",
        "- Models compared: Dummy prior baseline, balanced logistic regression, balanced random forest, balanced extra trees, and balanced histogram gradient boosting.",
        "- Primary selection metric: mean cross-validated average precision, suitable for rare positive labels.",
        "",
        "## Best model summary",
        "",
        "| Dataset | Rows | Positives | Positive rate | Best model | CV avg precision | Holdout avg precision | Holdout F1 | Holdout recall |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {dataset} | {rows} | {positive_count} | {positive_rate:.3f} | {best_model} | "
            "{best_cv_average_precision_mean:.3f} | {best_holdout_average_precision:.3f} | "
            "{best_holdout_f1:.3f} | {best_holdout_recall:.3f} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## Graphs",
            "",
            "PNG charts are saved under `graphs/` and are regenerated on every run.",
            "",
        ]
    )
    for path in graph_files:
        relative_path = Path(path).relative_to(results_dir)
        lines.append(f"- {relative_path.as_posix()}")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- model_summary.csv: compact comparison and chosen model for each dataset.",
            "- *_cv_metrics.csv: stratified cross-validation metrics by model.",
            "- *_holdout_metrics.csv: stratified holdout metrics by model.",
            "- *_holdout_predictions.csv: holdout predictions and scores by model.",
            "- *_feature_importance.csv: permutation importance for the selected model on the holdout split.",
            "- *_best_model.joblib: final selected pipeline retrained on all eligible rows.",
            "- graphs/*.png: visual summaries, model comparisons, curves, confusion matrices, and feature importance charts.",
            "- run_metadata.json: Python, package, feature, and runtime details.",
            "",
            "## Caution",
            "",
            "The positive class is rare in both datasets. Treat holdout results as directional, and prefer average precision, recall, and precision over plain accuracy.",
            "",
        ]
    )
    (results_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(results_dir: Path, summaries: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    metadata["summary_graph_file"] = plot_overall_summary(summaries, results_dir / "graphs")
    summary_records = []
    for item in summaries:
        record = {key: value for key, value in item.items() if key != "graph_files"}
        record["graph_count"] = len(item.get("graph_files", []))
        summary_records.append(record)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(results_dir / "model_summary.csv", index=False)
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(results_dir, summaries, metadata)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "max_cv_folds": args.cv_folds,
        "dataset_configs": [
            {
                "name": config.name,
                "filename": config.filename,
                "target": config.target,
                "features": list(config.features),
            }
            for config in DATASETS
        ],
    }

    summaries = []
    for config in DATASETS:
        print(f"Training {dataset_label(config)} model set...")
        summaries.append(
            train_dataset(
                config=config,
                data_dir=data_dir,
                results_dir=results_dir,
                random_state=args.random_state,
                test_size=args.test_size,
                max_cv_folds=args.cv_folds,
            )
        )

    metadata["run_finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_outputs(results_dir, summaries, metadata)

    print("\nDone. Key results:")
    for item in summaries:
        print(
            "- {dataset}: best={best_model}, CV AP={best_cv_average_precision_mean:.3f}, "
            "holdout AP={best_holdout_average_precision:.3f}, holdout F1={best_holdout_f1:.3f}".format(
                **item
            )
        )
    print(f"\nResults written to: {results_dir}")


if __name__ == "__main__":
    main()
