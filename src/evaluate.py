from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .runtime import prepare_runtime

prepare_runtime()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import ensure_dir


def compute_metrics(y_true, y_pred, labels: Iterable[str]) -> dict:
    report = classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "report": report,
    }


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(metrics, indent=2))


def save_confusion_matrix(y_true, y_pred, labels: list[str], title: str, output_path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def comparison_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(by=["accuracy", "macro_f1"], ascending=False)
