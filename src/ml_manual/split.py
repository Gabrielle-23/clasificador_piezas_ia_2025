# src/ml_manual/split.py
from __future__ import annotations
import numpy as np

def train_test_split(X, y, test_size=0.3, random_state=42, stratify=None):
    """
    Split básico estilo sklearn.
    - X: array-like (N,D) o DataFrame (si es DF, conviene pasar .values)
    - y: array-like (N,)
    - stratify: si no es None, se estratifica por etiquetas (típico en clasificación)
    Devuelve: X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y deben tener la misma cantidad de filas.")
    N = X.shape[0]
    rng = np.random.default_rng(random_state)

    n_test = int(round(N * float(test_size)))
    n_test = max(1, min(n_test, N - 1))

    if stratify is None:
        idx = np.arange(N)
        rng.shuffle(idx)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
    else:
        strat = np.asarray(stratify)
        if strat.shape[0] != N:
            raise ValueError("stratify debe tener largo N.")

        train_idx = []
        test_idx = []
        for cls in np.unique(strat):
            cls_idx = np.where(strat == cls)[0]
            rng.shuffle(cls_idx)
            cls_test = int(round(len(cls_idx) * float(test_size)))
            cls_test = max(1, min(cls_test, len(cls_idx) - 1)) if len(cls_idx) > 1 else 0
            test_idx.extend(cls_idx[:cls_test].tolist())
            train_idx.extend(cls_idx[cls_test:].tolist())

        train_idx = np.array(train_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
