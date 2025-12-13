# src/ml_manual/kmeans.py
from __future__ import annotations
import numpy as np

class KMeans:
    """
    KMeans "manual" con API parecida a sklearn:
      - fit(X)
      - predict(X)
      - labels_
      - cluster_centers_

    Soporta init:
      - 'random'
      - ndarray (K, D) con centros iniciales (tu caso con prototipos)
    """
    def __init__(
        self,
        n_clusters: int = 2,
        init="random",
        n_init=1,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    @staticmethod
    def _squared_distances(X, C):
        # X: (N,D), C: (K,D) -> dist2: (N,K)
        # ||x-c||^2 = sum((x-c)^2)
        diff = X[:, None, :] - C[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _init_centers(self, X, rng):
        if isinstance(self.init, str):
            if self.init != "random":
                raise ValueError("init soporta 'random' o ndarray(K,D).")
            # random: elegir K muestras como centros iniciales
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[idx].copy()

        # init como array de centros
        C = np.asarray(self.init, dtype=float)
        if C.shape != (self.n_clusters, X.shape[1]):
            raise ValueError(f"init ndarray debe ser shape ({self.n_clusters},{X.shape[1]}).")
        return C.copy()

    def _run_once(self, X, rng):
        C = self._init_centers(X, rng)

        prev_inertia = None
        for _ in range(self.max_iter):
            dist2 = self._squared_distances(X, C)
            labels = np.argmin(dist2, axis=1)
            inertia = float(np.sum(dist2[np.arange(X.shape[0]), labels]))

            # Recalcular centros
            C_new = C.copy()
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    C_new[k] = X[mask].mean(axis=0)
                else:
                    # cluster vacío -> resembrar con un punto aleatorio
                    C_new[k] = X[rng.integers(0, X.shape[0])]

            # criterio de parada: mejora de inercia o movimiento pequeño
            move = np.linalg.norm(C_new - C)
            C = C_new
            if prev_inertia is not None:
                if abs(prev_inertia - inertia) <= self.tol * (prev_inertia + 1e-12):
                    break
            if move <= self.tol:
                break

            prev_inertia = inertia

        # labels finales
        dist2 = self._squared_distances(X, C)
        labels = np.argmin(dist2, axis=1)
        inertia = float(np.sum(dist2[np.arange(X.shape[0]), labels]))
        return C, labels, inertia

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("KMeans.fit espera X 2D (n_samples, n_features).")
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples debe ser >= n_clusters.")

        best = None
        base_rng = np.random.default_rng(self.random_state)

        # Si init viene como ndarray (prototipos), n_init suele ser 1 (como tu nodo1/nodo2) :contentReference[oaicite:2]{index=2}
        for i in range(max(1, self.n_init)):
            rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
            C, labels, inertia = self._run_once(X, rng)
            if best is None or inertia < best[2]:
                best = (C, labels, inertia)

        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans no está fitteado. Llamá fit() primero.")
        X = np.asarray(X, dtype=float)
        dist2 = self._squared_distances(X, self.cluster_centers_)
        return np.argmin(dist2, axis=1)
