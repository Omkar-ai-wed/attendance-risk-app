"""Target definition - Approach A (current-threshold, rule-based classification).

``At Risk``      : attendance % is strictly BELOW the threshold.
``Not at Risk``  : attendance % is at or above the threshold.

This is a rule, not a forecast. The default threshold of 75% is an EXAMPLE only;
check your institution's actual rule and change it in the sidebar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_utils import LABEL_COLUMN, attendance_pct

AT_RISK = "At Risk"
NOT_AT_RISK = "Not at Risk"
POSITIVE_LABEL = AT_RISK          # positive class for precision / recall / F1
DEFAULT_THRESHOLD = 75.0          # example only - verify with the institution
LABEL_TO_INT = {NOT_AT_RISK: 0, AT_RISK: 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def validate_threshold(threshold) -> float:
    try:
        t = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("Threshold must be a number.") from None
    if np.isnan(t) or not (0 < t <= 100):
        raise ValueError("Threshold must be greater than 0 and at most 100.")
    return t


def classify_attendance(pct: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Label a single attendance percentage. Boundary: exactly at threshold = Not at Risk."""
    t = validate_threshold(threshold)
    if pct is None or np.isnan(pct):
        raise ValueError("Attendance percentage is undefined (check lectures conducted).")
    # round() removes float noise such as 28.999999999999996 vs 29.0
    return AT_RISK if round(float(pct), 9) < t else NOT_AT_RISK


def label_series(pct, threshold: float = DEFAULT_THRESHOLD) -> pd.Series:
    """Vectorised labelling. NaN percentages raise (validate the data first)."""
    t = validate_threshold(threshold)
    arr = np.asarray(pct, dtype=float)
    if np.isnan(arr).any():
        raise ValueError("Cannot label rows with undefined attendance percentage.")
    labels = np.where(np.round(arr, 9) < t, AT_RISK, NOT_AT_RISK)
    return pd.Series(labels, index=getattr(pct, "index", None), name=LABEL_COLUMN)


def add_risk_labels(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Return a copy with ``attendance_pct`` and ``risk_label`` columns."""
    out = df.copy()
    out["attendance_pct"] = attendance_pct(out["lectures_attended"], out["lectures_conducted"])
    out[LABEL_COLUMN] = label_series(out["attendance_pct"], threshold).values
    return out


def encode_labels(labels) -> np.ndarray:
    """Map text labels to 0/1 (1 = At Risk)."""
    return pd.Series(labels).map(LABEL_TO_INT).to_numpy(dtype=int)


def decode_labels(codes) -> np.ndarray:
    return pd.Series(codes).map(INT_TO_LABEL).to_numpy()
