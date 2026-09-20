"""Pixel-level binary change-detection metrics for the tutorial and its constant baseline.

The upstream evaluation (`confuse_matrix.py`) pools one 2 × 2 confusion matrix over all scored pixels with
`pred > 0.5` and `target > 0.5` and reports recall, precision, F1 and IoU of the *change* class; this module
reproduces those four numbers on the same pooled matrix and adds the overall pixel accuracy, the change fraction of
the labels and of the predictions, and the confusion counts. All numbers are pooled over the labelled pixels of
the pairs scored together (the ignore index excluded); nothing here estimates dispersion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def confusion_counts(predictions: Sequence[Any], labels: Sequence[Any], *, ignore_index: int) -> dict[str, int]:
    """TP / FP / FN / TN of the change class over every labelled pixel of the given pairs."""
    import numpy as np

    tp = fp = fn = tn = 0
    for pred, label in zip(predictions, labels, strict=True):
        pred = np.asarray(pred).reshape(-1)
        label = np.asarray(label).reshape(-1)
        if pred.shape != label.shape:
            raise ValueError(f"prediction shape {pred.shape} != label shape {label.shape}")
        valid = label != ignore_index
        p = pred[valid].astype(bool)
        t = label[valid].astype(bool)
        tp += int(np.sum(p & t))
        fp += int(np.sum(p & ~t))
        fn += int(np.sum(~p & t))
        tn += int(np.sum(~p & ~t))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    total = tp + fp + fn + tn
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = (
        2 * recall * precision / (recall + precision)
        if recall is not None and precision is not None and recall + precision
        else 0.0
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    rnd = lambda v: None if v is None else round(float(v), 4)  # noqa: E731
    return {
        "pixels": total,
        "f1": rnd(f1),
        "iou": rnd(iou),
        "precision": rnd(precision),
        "recall": rnd(recall),
        "accuracy": rnd((tp + tn) / total) if total else None,
        "change_fraction_label": rnd((tp + fn) / total) if total else None,
        "change_fraction_predicted": rnd((tp + fp) / total) if total else None,
        "confusion": dict(counts),
    }


def change_metrics(predictions: Sequence[Any], labels: Sequence[Any], *, ignore_index: int = -1) -> dict[str, Any]:
    """F1, IoU, precision and recall of the change class (upstream's protocol), overall accuracy and the change
    fractions, pooled over the labelled pixels of the pairs."""
    return metrics_from_counts(confusion_counts(predictions, labels, ignore_index=ignore_index))


def unchanged_baseline(labels: Sequence[Any], *, ignore_index: int = -1) -> dict[str, Any]:
    """The constant predictor that calls every pixel unchanged — the baseline any change detector must beat —
    scored on the same pixels as the model: its accuracy is the unchanged fraction and its F1 and IoU are 0."""
    import numpy as np

    predictions = [np.zeros(np.asarray(label).shape, dtype=bool) for label in labels]
    report = change_metrics(predictions, labels, ignore_index=ignore_index)
    report["note"] = "predicts 'unchanged' everywhere; accuracy equals the unchanged fraction, F1 and IoU are 0"
    return report
