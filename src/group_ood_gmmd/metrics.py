"""Grouping and OOD metrics shared by the reproduction scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def nonoverlapping_groups(
    values: np.ndarray,
    size: int,
    *,
    permutation_seed: int,
) -> np.ndarray:
    """Permute values and return arithmetic means of disjoint groups."""
    values = np.asarray(values)
    rng = np.random.default_rng(permutation_seed)
    shuffled = rng.permutation(values)
    count = len(shuffled) // int(size)
    return shuffled[: count * size].reshape(count, size).mean(axis=1)


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    """Return AUROC and FPR at the first ROC point whose TPR reaches 95%."""
    labels = np.concatenate(
        [
            np.zeros(len(id_scores), dtype=np.int8),
            np.ones(len(ood_scores), dtype=np.int8),
        ]
    )
    scores = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(labels, scores) * 100.0)
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = float(fpr[np.argmax(tpr >= 0.95)] * 100.0)
    return {"auroc": auroc, "fpr95": fpr95}


def threshold_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    validation_scores: np.ndarray,
    *,
    quantile: float = 0.9,
) -> dict[str, float]:
    """Return validation-calibrated sensitivity, specificity, and balance."""
    threshold = float(np.quantile(validation_scores, quantile, method="linear"))
    sensitivity = float(np.mean(np.asarray(ood_scores) > threshold))
    specificity = float(np.mean(np.asarray(id_scores) <= threshold))
    return {
        "threshold": threshold,
        "sensitivity": sensitivity * 100.0,
        "specificity": specificity * 100.0,
        "balanced_accuracy": (sensitivity + specificity) * 50.0,
    }
