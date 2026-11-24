"""Accumulate semantic segmentation metrics over labeled pixels."""

import numpy as np

from .data import CLASS_NAMES, NUM_CLASSES, VOID


def confusion(truth, prediction):
    if truth.shape != prediction.shape:
        raise ValueError("The prediction and ground truth must have matching dimensions.")
    valid = truth != VOID
    pairs = truth[valid].astype(np.int64) * NUM_CLASSES + prediction[valid]
    return np.bincount(pairs, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def summarize(matrix):
    true_positive = matrix.diagonal()
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - true_positive
    represented = union > 0
    if not represented.any():
        raise ValueError("No labeled pixels were available for evaluation.")
    iou = np.divide(true_positive, union, out=np.zeros(NUM_CLASSES), where=represented)
    return {
        "mean_iou": float(iou[represented].mean()),
        "pixel_accuracy": float(true_positive.sum() / matrix.sum()),
        "per_class_iou": {
            name: float(iou[index]) if represented[index] else None
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
    }
