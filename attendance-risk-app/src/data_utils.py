"""Data utilities: schema, attendance calculation, validation, synthetic data.

All data used by this project is FICTIONAL. Student IDs are synthetic
(e.g. ``STU0001``) and are never used as model features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
ID_COLUMN = "student_id"          # identifier only - NEVER a feature
LABEL_COLUMN = "risk_label"       # target (derived by src/labeling.py)
REQUIRED_COLUMNS = ["lectures_conducted", "lectures_attended"]
OPTIONAL_COLUMNS = ["previous_attendance_pct", "recent_absences"]
FEATURE_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Path is relative to this file, so it works locally and on Streamlit Cloud.
DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_attendance.csv"

DATA_DICTIONARY = pd.DataFrame(
    [
        (ID_COLUMN, "text", "No (identifier)", "Fictional ID such as STU0001. Never used as a feature."),
        ("lectures_conducted", "integer > 0", "Yes (required)", "Total lectures held so far in the term."),
        ("lectures_attended", "integer, 0..conducted", "Yes (required)", "Lectures the student attended."),
        ("previous_attendance_pct", "float, 0-100", "Yes (optional)", "Attendance % in the previous term. Missing values are median-imputed."),
        ("recent_absences", "integer >= 0", "Yes (optional)", "Lectures missed in the most recent two weeks. Missing values are median-imputed."),
        (LABEL_COLUMN, "At Risk / Not at Risk", "Target", "Derived from current attendance % and the chosen threshold (Approach A)."),
    ],
    columns=["column", "type / range", "used as model input?", "description"],
)


# --------------------------------------------------------------------------
# Attendance calculation
# --------------------------------------------------------------------------
def attendance_pct(attended, conducted):
    """Return ``attended / conducted * 100``.

    Works for scalars and array-likes. Where ``conducted`` is zero, negative or
    missing, the result is ``NaN`` (never raises ZeroDivisionError).
    """
    a = np.asarray(attended, dtype=float)
    c = np.asarray(conducted, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(c > 0, a / c * 100.0, np.nan)
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_single(conducted, attended, previous_pct=None, recent_absences=None) -> List[str]:
    """Validate one manual entry. Returns a list of human-readable errors."""
    errors: List[str] = []

    def _num(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(v) else v

    c, a = _num(conducted), _num(attended)
    if c is None:
        errors.append("Lectures conducted must be a number.")
    if a is None:
        errors.append("Lectures attended must be a number.")
    if c is not None:
        if c < 0:
            errors.append("Lectures conducted cannot be negative.")
        elif c == 0:
            errors.append("Lectures conducted must be greater than zero (attendance % is undefined otherwise).")
        elif c != int(c):
            errors.append("Lectures conducted must be a whole number.")
    if a is not None:
        if a < 0:
            errors.append("Lectures attended cannot be negative.")
        elif a != int(a):
            errors.append("Lectures attended must be a whole number.")
    if c is not None and a is not None and c > 0 and a > c:
        errors.append("Lectures attended cannot exceed lectures conducted.")

    p = _num(previous_pct)
    if previous_pct is not None and p is not None and not (0 <= p <= 100):
        errors.append("Previous attendance % must be between 0 and 100.")
    r = _num(recent_absences)
    if recent_absences is not None and r is not None and (r < 0 or r != int(r)):
        errors.append("Recent absences must be a whole number of zero or more.")
    return errors


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_dataframe`."""

    valid_df: pd.DataFrame
    invalid_df: pd.DataFrame
    schema_errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.schema_errors


def _empty_result(columns, schema_errors: List[str]) -> ValidationResult:
    return ValidationResult(pd.DataFrame(columns=columns), pd.DataFrame(columns=columns), schema_errors)


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing optional feature columns as NaN (handled by the imputer)."""
    out = df.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate an uploaded / loaded table.

    * Missing required columns or an empty file -> ``schema_errors`` (nothing usable).
    * Row-level problems -> the row goes to ``invalid_df`` with a reason; good rows
      are returned in ``valid_df`` (with numeric dtypes and all feature columns).
    """
    if df is None or len(df.columns) == 0:
        return _empty_result([], ["The file has no columns. Is it a valid CSV?"])

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return _empty_result(df.columns, [
            f"Missing required column(s): {', '.join(missing)}. "
            f"Required: {', '.join(REQUIRED_COLUMNS)}."
        ])
    if len(df) == 0:
        return _empty_result(df.columns, ["The file contains a header but no data rows."])

    raw = df.copy()
    num_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    errs = pd.Series("", index=df.index, dtype=object)

    def flag(mask, message):
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        errs.loc[mask] = errs.loc[mask] + message + "; "

    conducted, attended = df["lectures_conducted"], df["lectures_attended"]
    flag(conducted.isna(), "lectures_conducted missing or not numeric")
    flag(attended.isna(), "lectures_attended missing or not numeric")
    flag(conducted < 0, "lectures_conducted is negative")
    flag(attended < 0, "lectures_attended is negative")
    flag(conducted == 0, "lectures_conducted is zero (attendance % undefined)")
    flag(conducted.notna() & (conducted % 1 != 0), "lectures_conducted is not a whole number")
    flag(attended.notna() & (attended % 1 != 0), "lectures_attended is not a whole number")
    flag((conducted > 0) & (attended > conducted), "lectures_attended exceeds lectures_conducted")

    if "previous_attendance_pct" in df.columns:
        p = df["previous_attendance_pct"]
        flag(raw["previous_attendance_pct"].notna() & p.isna(), "previous_attendance_pct is not numeric")
        flag((p < 0) | (p > 100), "previous_attendance_pct outside 0-100")
    if "recent_absences" in df.columns:
        r = df["recent_absences"]
        flag(raw["recent_absences"].notna() & r.isna(), "recent_absences is not numeric")
        flag(r.notna() & ((r < 0) | (r % 1 != 0)), "recent_absences must be a whole number >= 0")

    bad = errs != ""
    invalid = raw.loc[bad].copy()
    invalid.insert(0, "csv_row", invalid.index + 2)  # +2: header row + 1-based
    invalid["validation_error"] = errs.loc[bad].str.rstrip("; ")
    valid = ensure_feature_columns(df.loc[~bad]).reset_index(drop=True)

    schema_errors: List[str] = []
    if len(valid) == 0:
        schema_errors.append("No valid rows remain after validation. See the rejected rows for reasons.")
    return ValidationResult(valid, invalid.reset_index(drop=True), schema_errors)


# --------------------------------------------------------------------------
# Loading and synthetic data
# --------------------------------------------------------------------------
def load_csv(path) -> pd.DataFrame:
    """Read a CSV with a clear error if the file is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.name}. Run `python -m src.data_utils` to regenerate it.")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # malformed CSV
        raise ValueError(f"Could not read {path.name} as CSV: {exc}") from exc


def generate_synthetic_data(n_students: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible, FICTIONAL attendance dataset (no real records).

    Each student has a latent attendance propensity; attended lectures are binomial
    draws from it. Optional features are noisy functions of the same propensity, and
    ~5% of optional values are blanked to exercise the imputer. No labels are drawn
    at random: the label is added separately by ``src.labeling`` from the counts.
    """
    rng = np.random.default_rng(seed)
    propensity = rng.beta(7.0, 2.4, size=n_students)
    conducted = rng.integers(40, 121, size=n_students)
    attended = rng.binomial(conducted, propensity)
    previous = np.clip(propensity * 100 + rng.normal(0, 8, n_students), 0, 100).round(1)
    recent = rng.poisson((1 - propensity) * 6).astype(float)

    previous = previous.astype(float)
    previous[rng.random(n_students) < 0.05] = np.nan
    recent[rng.random(n_students) < 0.05] = np.nan

    return pd.DataFrame({
        ID_COLUMN: [f"STU{i + 1:04d}" for i in range(n_students)],
        "lectures_conducted": conducted,
        "lectures_attended": attended,
        "previous_attendance_pct": previous,
        "recent_absences": recent,
    })


def template_csv() -> str:
    """A tiny CSV template users can fill in for batch prediction."""
    return (
        "student_id,lectures_conducted,lectures_attended,previous_attendance_pct,recent_absences\n"
        "STU9001,80,64,78.5,1\n"
        "STU9002,80,52,70.0,4\n"
    )


if __name__ == "__main__":
    # Regenerate the bundled sample file:  python -m src.data_utils
    from .labeling import DEFAULT_THRESHOLD, add_risk_labels

    df = add_risk_labels(generate_synthetic_data(), DEFAULT_THRESHOLD).drop(columns=["attendance_pct"])
    DEFAULT_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_SAMPLE_PATH, index=False)
    print(f"Wrote {len(df)} synthetic rows to {DEFAULT_SAMPLE_PATH}")
    print(df[LABEL_COLUMN].value_counts().to_string())
