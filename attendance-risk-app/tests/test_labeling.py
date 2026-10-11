import numpy as np
import pandas as pd
import pytest

from src.labeling import (
    AT_RISK, NOT_AT_RISK, add_risk_labels, classify_attendance, decode_labels,
    encode_labels, label_series, validate_threshold,
)


def test_threshold_boundary_cases():
    assert classify_attendance(74.99, 75) == AT_RISK
    assert classify_attendance(75.0, 75) == NOT_AT_RISK   # exactly at threshold
    assert classify_attendance(75.01, 75) == NOT_AT_RISK
    assert classify_attendance(0, 75) == AT_RISK
    assert classify_attendance(100, 75) == NOT_AT_RISK


def test_custom_threshold():
    assert classify_attendance(70, 65) == NOT_AT_RISK
    assert classify_attendance(70, 80) == AT_RISK


def test_float_noise_does_not_flip_boundary():
    pct = 29 / 100 * 100            # 28.999999999999996 in floating point
    assert classify_attendance(pct, 29) == NOT_AT_RISK


@pytest.mark.parametrize("bad", [0, -5, 101, "abc", None, float("nan")])
def test_invalid_threshold_rejected(bad):
    with pytest.raises(ValueError):
        validate_threshold(bad)


def test_undefined_percentage_raises():
    with pytest.raises(ValueError):
        classify_attendance(float("nan"))
    with pytest.raises(ValueError):
        label_series(pd.Series([80.0, np.nan]))


def test_add_risk_labels_columns_and_values():
    df = pd.DataFrame({"lectures_conducted": [100, 100, 80], "lectures_attended": [74, 75, 60]})
    out = add_risk_labels(df, 75)
    assert out["risk_label"].tolist() == [AT_RISK, NOT_AT_RISK, NOT_AT_RISK]
    assert out["attendance_pct"].tolist() == pytest.approx([74.0, 75.0, 75.0])
    assert "risk_label" not in df.columns  # input untouched


def test_label_encoding_roundtrip():
    labels = [AT_RISK, NOT_AT_RISK, AT_RISK]
    codes = encode_labels(labels)
    assert codes.tolist() == [1, 0, 1]
    assert decode_labels(codes).tolist() == labels
