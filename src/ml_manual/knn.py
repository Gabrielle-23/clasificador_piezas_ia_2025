# src/ml_manual/knn.py
from __future__ import annotations
import numpy as np

class KNeighborsClassifier:
    """
    KNN manual con:
      - metric euclidean
      - weights: 'uniform' o 'distance' (tu caso usa distance :contentReference[oaicite:3]{index=3})
    API similar:
      - fit(X,y)
      - predict(X)
      - predict_proba(X)
      - classes_
    """
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "euclidean", eps: float = 1e-12):
        self.n_neighbors = int(n_neighbors)
        self.weights = str(weights)
        self.metric = str(metric)
        self.eps = float(eps)

        self.X_ = None
        self.y_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("KNN.fit espera X 2D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener la misma cantidad de muestras.")
        if self.metric != "euclidean":
            raise ValueError("Solo se soporta metric='euclidean'.")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors debe ser >= 1.")

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    @staticmethod
    def _euclidean_distances(X, Y):
        # X: (M,D), Y:(N,D) -> dist: (M,N)
        # dist^2 = (x-y)^2 sum
        diff = X[:, None, :] - Y[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        return np.sqrt(dist2)

    def _kneighbors(self, X):
        if self.X_ is None:
            raise RuntimeError("KNN no está fitteado.")
        X = np.asarray(X, dtype=float)
        dist = self._euclidean_distances(X, self.X_)  # (M,N)
        k = min(self.n_neighbors, self.X_.shape[0])
        idx = np.argpartition(dist, kth=k-1, axis=1)[:, :k]  # vecinos sin ordenar
        # Ordenar esos k por distancia
        row = np.arange(X.shape[0])[:, None]
        idx_sorted = idx[row, np.argsort(dist[row, idx], axis=1)]
        dist_sorted = dist[row, idx_sorted]
        return dist_sorted, idx_sorted

    def predict_proba(self, X):
        dist, idx = self._kneighbors(X)  # (M,k)
        M, k = idx.shape
        proba = np.zeros((M, self.classes_.shape[0]), dtype=float)

        for i in range(M):
            neigh_y = self.y_[idx[i]]
            neigh_d = dist[i]

            if self.weights == "uniform":
                w = np.ones(k, dtype=float)
            elif self.weights == "distance":
                w = 1.0 / (neigh_d + self.eps)
            else:
                raise ValueError("weights soporta 'uniform' o 'distance'.")

            # acumular pesos por clase
            for cls_index, cls in enumerate(self.classes_):
                proba[i, cls_index] = np.sum(w[neigh_y == cls])

            s = proba[i].sum()
            if s > 0:
                proba[i] /= s

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
