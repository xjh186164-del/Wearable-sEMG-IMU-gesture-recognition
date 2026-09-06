"""Dependency-free multiclass classification metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
) -> dict:
    """Return JSON-safe confusion, accuracy, balanced accuracy, and Macro-F1."""
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    class_count = len(class_labels)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.size == 0:
        raise ValueError("true and predicted labels must be nonempty one-dimensional arrays")
    if truth.shape != prediction.shape:
        raise ValueError("true and predicted labels must have the same shape")
    if class_count < 2 or len(set(class_labels)) != class_count:
        raise ValueError("class labels must contain at least two unique values")
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("true and predicted labels must be integers")
    if (
        int(truth.min()) < 0
        or int(prediction.min()) < 0
        or int(truth.max()) >= class_count
        or int(prediction.max()) >= class_count
    ):
        raise ValueError("true or predicted label is outside the class range")

    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (truth.astype(np.int64), prediction.astype(np.int64)), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(class_count, dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(class_count, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    per_class = [
        {
            "label": str(class_labels[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(class_count)
    ]
    return {
        "accuracy": float(true_positive.sum() / truth.size),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
