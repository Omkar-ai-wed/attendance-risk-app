"""Metrics, confusion matrices and plotting. Everything is computed from
held-out predictions - nothing is hardcoded."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .labeling import AT_RISK, NOT_AT_RISK

CLASS_NAMES = [NOT_AT_RISK, AT_RISK]   # index 0 / 1 ; positive class = At Risk (1)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Binary metrics with At Risk (1) as the positive class.

    ``zero_division=0`` means a metric with an empty denominator (e.g. no
    predicted positives) is reported as 0.0 instead of raising.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and the same length.")
    kw = dict(zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, **kw)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, **kw)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, **kw)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", **kw)),
    }


def confusion(y_true, y_pred) -> np.ndarray:
    """2x2 matrix, rows = actual, cols = predicted, order [Not at Risk, At Risk]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def confusion_df(cm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        cm,
        index=[f"Actual: {c}" for c in CLASS_NAMES],
        columns=[f"Predicted: {c}" for c in CLASS_NAMES],
    )


def class_report_df(y_true, y_pred) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s.astype(int)}, index=CLASS_NAMES
    )


def comparison_table(metrics_by_model: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """One row per model; columns are metrics for the At Risk class + macro/weighted F1."""
    df = pd.DataFrame(metrics_by_model).T
    df = df[["accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1"]]
    df.index.name = "model"
    return df


def plot_confusion_matrix(cm: np.ndarray, title: str) -> Figure:
    """Polished annotated heat-map with contextual cell roles and high-contrast typography."""
    fig = Figure(figsize=(3.8, 3.4), dpi=120)
    ax = fig.subplots()
    
    # Modern clean heatmap
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not at Risk", "At Risk"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not at Risk", "At Risk"], fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=9.5, labelpad=6, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=9.5, labelpad=6, fontweight="bold")
    ax.set_title(title, fontsize=10.5, pad=8, fontweight="bold", color="#1E293B")
    
    # Subtle borders
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.0)
    ax.tick_params(colors="#475569", width=0.8)

    cell_labels = [
        ["TN (Safe)", "FP (False Alarm)"],
        ["FN (Missed)", "TP (At Risk)"]
    ]
    
    top = cm.max() if cm.max() > 0 else 1
    total = cm.sum() if cm.sum() > 0 else 1
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            pct = val / total * 100
            txt_color = "#FFFFFF" if val > top * 0.55 else "#0F172A"
            sub_color = "#E2E8F0" if val > top * 0.55 else "#64748B"
            ax.text(j, i - 0.12, f"{val}", ha="center", va="center",
                    color=txt_color, fontsize=13, fontweight="bold")
            ax.text(j, i + 0.16, f"{cell_labels[i][j]}\n({pct:.1f}%)", ha="center", va="center",
                    color=sub_color, fontsize=7.5)
            
    fig.tight_layout()
    return fig

