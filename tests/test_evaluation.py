import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation import (
    class_report_df, compute_metrics, comparison_table, confusion, confusion_df, plot_confusion_matrix,
)


def test_confusion_matrix_known_values():
    y_true = [1, 1, 1, 0, 0, 0, 0, 1]
    y_pred = [1, 1, 0, 0, 0, 1, 0, 1]
    cm = confusion(y_true, y_pred)
    assert cm.tolist() == [[3, 1], [1, 3]]       # rows actual [0,1], cols predicted [0,1]
    assert confusion_df(cm).shape == (2, 2)


def test_metrics_match_hand_calculation():
    y_true = [1, 1, 1, 0, 0, 0, 0, 1]
    y_pred = [1, 1, 0, 0, 0, 1, 0, 1]
    m = compute_metrics(y_true, y_pred)
    # TP=3, FP=1, FN=1, TN=3
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))


def test_zero_division_handled():
    m = compute_metrics([1, 1, 0, 0], [0, 0, 0, 0])   # no predicted positives
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["accuracy"] == 0.5


def test_single_class_in_truth_confusion_is_still_2x2():
    assert confusion([0, 0, 0], [0, 0, 1]).shape == (2, 2)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [1])


def test_class_report_and_comparison_table():
    rep = class_report_df([1, 0, 1, 0], [1, 0, 0, 0])
    assert rep.loc["At Risk", "support"] == 2
    table = comparison_table({"A": compute_metrics([1, 0], [1, 0]), "B": compute_metrics([1, 0], [0, 0])})
    assert list(table.index) == ["A", "B"] and table.loc["A", "f1"] == 1.0


def test_plot_returns_figure():
    fig = plot_confusion_matrix(np.array([[3, 1], [1, 3]]), "demo")
    assert fig.axes
