import io

import numpy as np
import pandas as pd
import pytest

from src.data_utils import FEATURE_COLUMNS, ID_COLUMN, generate_synthetic_data, validate_dataframe
from src.labeling import AT_RISK, NOT_AT_RISK
from src.modeling import (
    BASELINE_MAJORITY, BASELINE_RULE, MODEL_NAMES, ModelingError, build_models, feature_importance,
    predict_dataframe, prepare_training_data, run_experiment, split_data,
)


@pytest.fixture(scope="module")
def df():
    return generate_synthetic_data(300, seed=3)


@pytest.fixture(scope="module")
def exp(df):
    return run_experiment(df, 75)


def test_ids_and_label_excluded_from_features(df):
    X, y, labeled, _ = prepare_training_data(df, 75)
    assert list(X.columns) == FEATURE_COLUMNS
    assert ID_COLUMN not in X.columns and "risk_label" not in X.columns and "attendance_pct" not in X.columns


def test_split_is_stratified_reproducible_and_disjoint(df):
    X, y, _, _ = prepare_training_data(df, 75)
    a = split_data(X, y)
    b = split_data(X, y)
    assert a[0].index.equals(b[0].index)
    assert set(a[0].index).isdisjoint(a[1].index)
    assert len(a[0]) + len(a[1]) == len(X)
    assert abs(a[2].mean() - a[3].mean()) < 0.05      # class ratio preserved


def test_pipelines_have_imputer_and_fit_on_train_only(exp):
    for name in MODEL_NAMES:
        pipe = exp.models[name]
        assert "impute" in pipe.named_steps
        imputer = pipe.named_steps["impute"]
        train_median = exp.X_train["recent_absences"].median()
        assert imputer.statistics_[FEATURE_COLUMNS.index("recent_absences")] == pytest.approx(train_median)


def test_experiment_contains_models_and_baselines(exp):
    assert set(exp.metrics.index) == set(MODEL_NAMES) | {BASELINE_RULE, BASELINE_MAJORITY}
    assert exp.metrics.loc[BASELINE_RULE, "accuracy"] == 1.0     # rule reproduces the label by definition
    for name in exp.metrics.index:
        cm = exp.confusion_matrices[name]
        assert cm.sum() == len(exp.y_test)


def test_metrics_come_from_actual_predictions(exp):
    for name in MODEL_NAMES:
        pred = exp.models[name].predict(exp.X_test)
        acc = float((pred == exp.y_test.to_numpy()).mean())
        assert exp.metrics.loc[name, "accuracy"] == pytest.approx(acc)


def test_experiment_is_reproducible(df):
    a, b = run_experiment(df, 75), run_experiment(df, 75)
    pd.testing.assert_frame_equal(a.metrics, b.metrics)


def test_prediction_output_schema_and_values(exp, df):
    valid = validate_dataframe(df).valid_df.head(20)
    out = predict_dataframe(exp.models["Random Forest"], valid, 75)
    assert len(out) == 20
    assert set(out["predicted_label"]) <= {AT_RISK, NOT_AT_RISK}
    assert out["prob_at_risk"].between(0, 1).all()
    assert {"attendance_pct", "rule_based_label", "predicted_label", "prob_at_risk"} <= set(out.columns)
    assert out[ID_COLUMN].tolist() == valid[ID_COLUMN].tolist()


def test_prediction_works_without_optional_columns(exp):
    row = pd.DataFrame({"lectures_conducted": [80], "lectures_attended": [40]})
    out = predict_dataframe(exp.models["Logistic Regression"], row, 75)
    assert out.loc[0, "predicted_label"] == AT_RISK
    assert out.loc[0, "rule_based_label"] == AT_RISK


def test_csv_export_roundtrip(exp, df):
    valid = validate_dataframe(df).valid_df.head(10)
    out = predict_dataframe(exp.models["Decision Tree"], valid, 75)
    buf = io.StringIO(out.to_csv(index=False))
    back = pd.read_csv(buf)
    assert list(back.columns) == list(out.columns) and len(back) == 10
    assert back["predicted_label"].tolist() == out["predicted_label"].tolist()


def test_feature_importance_availability(exp):
    for name in MODEL_NAMES:
        fi = feature_importance(exp.models[name])
        assert fi is not None and set(fi["feature"]) == set(FEATURE_COLUMNS)
        assert (fi["importance"] >= 0).all()
    assert feature_importance(object()) is None      # unsupported model -> None


def test_changing_threshold_changes_labels(df):
    low, high = run_experiment(df, 50), run_experiment(df, 90)
    assert low.y_train.mean() < high.y_train.mean()


def test_single_class_and_tiny_data_raise_friendly_errors(df):
    with pytest.raises(ModelingError):
        run_experiment(df, 1)       # nobody below 1% -> one class only
    with pytest.raises(ModelingError):
        run_experiment(df.head(6), 75)
    with pytest.raises(ModelingError):
        run_experiment(pd.DataFrame({"lectures_conducted": [10]}), 75)


def test_build_models_returns_three_pipelines():
    assert list(build_models()) == MODEL_NAMES
