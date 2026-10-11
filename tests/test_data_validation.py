import io
import numpy as np
import pandas as pd
import pytest

from src.data_utils import (
    FEATURE_COLUMNS, ID_COLUMN, LABEL_COLUMN, REQUIRED_COLUMNS, attendance_pct,
    generate_synthetic_data, template_csv, validate_dataframe, validate_single,
)


def test_attendance_percentage_basic():
    assert attendance_pct(60, 80) == pytest.approx(75.0)
    assert attendance_pct(0, 50) == 0.0
    assert attendance_pct(50, 50) == 100.0


def test_attendance_percentage_vectorised_and_zero_conducted():
    out = attendance_pct([10, 5, 0], [20, 0, 0])
    assert out[0] == pytest.approx(50.0)
    assert np.isnan(out[1]) and np.isnan(out[2])  # safe, no ZeroDivisionError


@pytest.mark.parametrize("conducted,attended,fragment", [
    (-5, 0, "negative"),
    (50, -1, "negative"),
    (50, 51, "exceed"),
    (0, 0, "greater than zero"),
    (50.5, 10, "whole number"),
    (None, 10, "must be a number"),
])
def test_validate_single_rejects_bad_counts(conducted, attended, fragment):
    errors = validate_single(conducted, attended)
    assert any(fragment in e for e in errors), errors


def test_validate_single_accepts_boundaries_and_checks_optional_ranges():
    assert validate_single(50, 50) == []
    assert validate_single(50, 0) == []
    assert validate_single(50, 40, previous_pct=101)
    assert validate_single(50, 40, previous_pct=-1)
    assert validate_single(50, 40, previous_pct=100, recent_absences=0) == []
    assert validate_single(50, 40, recent_absences=-2)


def test_dataframe_validation_splits_valid_and_invalid_rows():
    df = pd.DataFrame({
        "lectures_conducted": [50, 50, 0, -3, 40, 60],
        "lectures_attended": [40, 60, 0, 1, 40, "abc"],
        "previous_attendance_pct": [70, 70, 70, 70, 120, 70],
    })
    res = validate_dataframe(df)
    assert list(res.valid_df.index) == [0]
    assert len(res.invalid_df) == 5
    assert "validation_error" in res.invalid_df.columns
    reasons = " ".join(res.invalid_df["validation_error"])
    for fragment in ["exceeds", "zero", "negative", "outside 0-100", "not numeric"]:
        assert fragment in reasons


def test_missing_optional_values_are_not_rejected():
    df = generate_synthetic_data(100, seed=1)
    assert df["recent_absences"].isna().any()
    res = validate_dataframe(df)
    assert len(res.invalid_df) == 0 and len(res.valid_df) == 100


def test_schema_validation_missing_required_column():
    res = validate_dataframe(pd.DataFrame({"lectures_conducted": [10]}))
    assert not res.ok
    assert "lectures_attended" in res.schema_errors[0]


def test_schema_validation_empty_and_columnless_files():
    assert not validate_dataframe(pd.DataFrame(columns=REQUIRED_COLUMNS)).ok
    assert not validate_dataframe(pd.DataFrame()).ok


def test_column_names_are_normalised():
    res = validate_dataframe(pd.DataFrame({" Lectures_Conducted ": [10], "LECTURES_ATTENDED": [7]}))
    assert res.ok and len(res.valid_df) == 1


def test_id_and_label_are_not_features():
    assert ID_COLUMN not in FEATURE_COLUMNS
    assert LABEL_COLUMN not in FEATURE_COLUMNS
    assert "attendance_pct" not in FEATURE_COLUMNS


def test_template_csv_is_valid():
    df = pd.read_csv(io.StringIO(template_csv()))
    assert validate_dataframe(df).ok


def test_synthetic_data_is_reproducible_and_fictional():
    a, b = generate_synthetic_data(50, seed=7), generate_synthetic_data(50, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert a[ID_COLUMN].str.match(r"^STU\d{4}$").all()
    assert (a["lectures_attended"] <= a["lectures_conducted"]).all()
