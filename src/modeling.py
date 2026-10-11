"""Training, evaluation and prediction (scikit-learn Pipelines).

Target = Approach A (rule-based, current-threshold). The models are therefore
learning to reproduce a known rule from raw lecture counts; a rule baseline and a
majority-class baseline are reported alongside to make that explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .data_utils import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    LABEL_COLUMN,
    attendance_pct,
    ensure_feature_columns,
    validate_dataframe,
)
from .evaluation import class_report_df, compute_metrics, confusion, comparison_table
from .labeling import (
    AT_RISK,
    DEFAULT_THRESHOLD,
    add_risk_labels,
    decode_labels,
    encode_labels,
    label_series,
    validate_threshold,
)

RANDOM_STATE = 42
TEST_SIZE = 0.25
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
BASELINE_RULE = "Rule baseline (threshold)"
BASELINE_MAJORITY = "Majority-class baseline"


class ModelingError(ValueError):
    """Raised for problems the user can fix (too little data, one class only...)."""


def _imputer() -> SimpleImputer:
    # fitted on training data only (it is a Pipeline step); keep_empty_features
    # prevents a column from vanishing if an optional column is entirely blank.
    return SimpleImputer(strategy="median", keep_empty_features=True)


def build_models(random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Three pipelines: impute (+ scale for LR) -> classifier."""
    return {
        "Logistic Regression": Pipeline([
            ("impute", _imputer()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Decision Tree": Pipeline([
            ("impute", _imputer()),
            ("clf", DecisionTreeClassifier(max_depth=6, random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("impute", _imputer()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=8,
                                           random_state=random_state, n_jobs=1)),
        ]),
    }


def prepare_training_data(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD):
    """Validate, label and split into (X, y, labeled_df, n_rejected).

    ``X`` contains FEATURE_COLUMNS only - the student ID and label never enter it.
    """
    threshold = validate_threshold(threshold)
    result = validate_dataframe(df)
    if not result.ok:
        raise ModelingError(" ".join(result.schema_errors))
    labeled = add_risk_labels(result.valid_df, threshold)
    X = labeled[FEATURE_COLUMNS].astype(float)
    y = pd.Series(encode_labels(labeled[LABEL_COLUMN]), index=labeled.index, name="y")
    return X, y, labeled, len(result.invalid_df)


def split_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Reproducible stratified split with friendly errors for tiny / one-class data."""
    counts = y.value_counts()
    if len(counts) < 2:
        raise ModelingError(
            "Only one risk class is present at this threshold. Choose a threshold "
            "that produces both 'At Risk' and 'Not at Risk' students."
        )
    n_test = int(np.ceil(len(y) * test_size))
    if counts.min() < 2 or len(y) < 10 or n_test < 2 or n_test < len(counts):
        raise ModelingError(
            "Not enough data for a stratified train/test split "
            "(need at least 10 rows and 2 students in each class)."
        )
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


@dataclass
class Experiment:
    """Everything the Model Evaluation tab needs, computed from held-out data."""

    threshold: float
    models: Dict[str, Pipeline]
    metrics: pd.DataFrame
    confusion_matrices: Dict[str, np.ndarray]
    class_reports: Dict[str, pd.DataFrame]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    n_rejected_rows: int = 0
    extra: dict = field(default_factory=dict)


def run_experiment(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
                   random_state: int = RANDOM_STATE) -> Experiment:
    """Fit the three models on the training split and score them on the test split."""
    X, y, _, n_rejected = prepare_training_data(df, threshold)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)

    models = build_models(random_state)
    predictions: Dict[str, np.ndarray] = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        predictions[name] = pipe.predict(X_test)

    # Baselines
    pct_test = attendance_pct(X_test["lectures_attended"], X_test["lectures_conducted"])
    predictions[BASELINE_RULE] = encode_labels(label_series(pct_test, threshold))
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    predictions[BASELINE_MAJORITY] = dummy.predict(X_test)

    metrics = {n: compute_metrics(y_test, p) for n, p in predictions.items()}
    return Experiment(
        threshold=float(threshold),
        models=models,
        metrics=comparison_table(metrics),
        confusion_matrices={n: confusion(y_test, p) for n, p in predictions.items()},
        class_reports={n: class_report_df(y_test, p) for n, p in predictions.items()},
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        n_rejected_rows=n_rejected,
    )


def predict_dataframe(model: Pipeline, df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Predict for already-validated rows. Returns an output table (CSV-ready)."""
    threshold = validate_threshold(threshold)
    data = ensure_feature_columns(df).reset_index(drop=True)
    X = data[FEATURE_COLUMNS].astype(float)
    pct = attendance_pct(X["lectures_attended"], X["lectures_conducted"])
    codes = model.predict(X)
    classes = list(model.classes_)
    prob_risk = model.predict_proba(X)[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

    out = pd.DataFrame()
    if ID_COLUMN in data.columns:
        out[ID_COLUMN] = data[ID_COLUMN]
    out["lectures_conducted"] = X["lectures_conducted"].astype(int)
    out["lectures_attended"] = X["lectures_attended"].astype(int)
    out["attendance_pct"] = np.round(pct, 2)
    out["threshold_pct"] = threshold
    out["rule_based_label"] = label_series(pct, threshold).values
    out["predicted_label"] = decode_labels(codes)
    out["prob_at_risk"] = np.round(prob_risk, 4)
    return out


def feature_importance(model: Pipeline, feature_names=FEATURE_COLUMNS) -> Optional[pd.DataFrame]:
    """Importance for models that expose it; ``None`` otherwise.

    Trees/forests: impurity-based ``feature_importances_``.
    Logistic regression: absolute standardized coefficient (+ signed coefficient).
    These describe association with the model's output, not causation.
    """
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": list(feature_names), "importance": imp}).sort_values(
            "importance", ascending=False, ignore_index=True)
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_).ravel()
        return pd.DataFrame({
            "feature": list(feature_names), "importance": np.abs(coef), "signed_coefficient": coef,
        }).sort_values("importance", ascending=False, ignore_index=True)
    return None
