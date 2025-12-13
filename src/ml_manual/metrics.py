# src/ml_manual/metrics.py
from __future__ import annotations
import numpy as np

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true y y_pred deben tener mismo largo.")
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[idx[yt], idx[yp]] += 1
    return cm

def classification_report(y_true, y_pred, labels=None, digits=4):
    """
    Reporte estilo sklearn (simplificado).
    Devuelve string.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # por clase: TP, FP, FN
    lines = []
    header = f"{'class':<15}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    lines.append(header)

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        lines.append(
            f"{str(lab):<15}{prec:>10.{digits}f}{rec:>10.{digits}f}{f1:>10.{digits}f}{support:>10d}"
        )

    supports = np.asarray(supports, dtype=float)
    total = int(supports.sum())

    macro_p = float(np.mean(precisions)) if len(precisions) else 0.0
    macro_r = float(np.mean(recalls)) if len(recalls) else 0.0
    macro_f = float(np.mean(f1s)) if len(f1s) else 0.0

    weighted_p = float(np.sum(np.asarray(precisions) * supports) / total) if total > 0 else 0.0
    weighted_r = float(np.sum(np.asarray(recalls) * supports) / total) if total > 0 else 0.0
    weighted_f = float(np.sum(np.asarray(f1s) * supports) / total) if total > 0 else 0.0

    acc = accuracy_score(y_true, y_pred)

    lines.append("")
    lines.append(f"{'accuracy':<15}{acc:>30.{digits}f}{total:>10d}")
    lines.append(f"{'macro avg':<15}{macro_p:>10.{digits}f}{macro_r:>10.{digits}f}{macro_f:>10.{digits}f}{total:>10d}")
    lines.append(f"{'weighted avg':<15}{weighted_p:>10.{digits}f}{weighted_r:>10.{digits}f}{weighted_f:>10.{digits}f}{total:>10d}")

    return "\n".join(lines)
