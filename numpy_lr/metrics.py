"""
numpy_lr.metrics
================
Classification evaluation metrics implemented with NumPy only.

Functions
---------
accuracy_score        — fraction of correct predictions
confusion_matrix      — K×K count matrix (rows = true, cols = predicted)
classification_report — per-class precision, recall, F1-score, support
"""

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of predictions that match the true labels."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int | None = None,
) -> np.ndarray:
    """
    Build a K×K confusion matrix where entry [i, j] counts how many
    samples with true class i were predicted as class j.

    The diagonal entries are correct predictions; off-diagonal entries are
    misclassifications.

    Parameters
    ----------
    y_true     : true integer labels
    y_pred     : predicted integer labels
    n_classes  : number of classes (inferred from data if None)
    """
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: list[str] | None = None,
) -> str:
    """
    Return a text table with precision, recall, F1-score, and support per
    class, plus macro-averaged summary row.

    Precision = TP / (TP + FP)   — how often the model is right when it
                                    predicts a given class.
    Recall    = TP / (TP + FN)   — how often the model finds all instances
                                    of a given class.
    F1        = 2·P·R / (P + R)  — harmonic mean; penalises imbalance
                                    between precision and recall.
    """
    classes   = np.unique(y_true)
    n_classes = len(classes)
    cm        = confusion_matrix(y_true, y_pred, n_classes=n_classes)

    if target_names is None:
        target_names = [str(c) for c in classes]

    rows: list[str] = []
    header = f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    rows.append(header)
    rows.append("─" * len(header))

    precisions, recalls, f1s = [], [], []
    for idx, c in enumerate(classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp

        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = cm[idx, :].sum()

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        rows.append(
            f"{target_names[idx]:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10}"
        )

    rows.append("─" * len(header))
    rows.append(
        f"{'macro avg':<20} {np.mean(precisions):>10.4f} {np.mean(recalls):>10.4f}"
        f" {np.mean(f1s):>10.4f} {len(y_true):>10}"
    )
    return "\n".join(rows)
