# src/ml_manual/scaler.py
from __future__ import annotations
import numpy as np

class StandardScaler:
    """
    Reimplementación simple del StandardScaler:
    x_scaled = (x - mean) / (std + eps)

    API similar a sklearn:
      - fit(X)
      - transform(X)
      - fit_transform(X)
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)
        self.mean_ = None
        self.scale_ = None  # std

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("StandardScaler.fit espera X 2D (n_samples, n_features).")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0)
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler no está fitteado. Llamá fit() primero.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / (self.scale_ + self.eps)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
