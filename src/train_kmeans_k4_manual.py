"""
Entrenamiento KMeans con k=4 (MANUAL, sin sklearn) usando features.csv.

- NO pisa tu sistema jerárquico actual.
- Usa src.ml_manual (StandardScaler, KMeans) como en kmeans_manual.py
- Guarda todo en: data/models_k4_manual/

Requisitos:
- Tener features.csv generado (mismas columnas que usás en el proyecto).
"""

from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
import joblib

# ====== IMPORTS MANUALES (sin sklearn) ======
from src.ml_manual.scaler import StandardScaler
from src.ml_manual.kmeans import KMeans

# Ajustá si tu ruta real es otra:
CSV_PATH = Path("data/features/features.csv")

# Carpeta NUEVA (no pisa nada)
MODELS_DIR = Path("data/models_k4_manual")

# Subset recomendado (k=4). Si alguna columna no existe, se ignora automáticamente.
K4_FEATURES_CANDIDATAS = [
    "aspect_ratio",
    "extent",
    "solidity",
    "circularity",
    "num_holes",
    "hole_area_ratio",
    "num_vertices_rdp",
    "vertices_perimeter_ratio",
    "hu1",
    "hu2",
    "hu3",
]

CLASSES_ORDEN = ["arandela", "clavo", "tornillo", "tuerca"]


def _seleccionar_features_existentes(df: pd.DataFrame, candidatas):
    feats = [f for f in candidatas if f in df.columns]
    faltantes = [f for f in candidatas if f not in df.columns]
    if faltantes:
        print("[WARN] Features no encontradas (se ignoran):", faltantes)
    if len(feats) < 4:
        raise ValueError(
            f"Muy pocas features disponibles ({len(feats)}). "
            f"Revisá tu features.csv o la lista K4_FEATURES_CANDIDATAS."
        )
    return feats


def _mapear_clusters_a_clases(y_true, labels_pred):
    """Devuelve mapping cluster(int) -> clase(str) por mayoría."""
    mapping = {}
    for c in np.unique(labels_pred):
        clases = y_true[labels_pred == c]
        if len(clases) == 0:
            mapping[int(c)] = "desconocida"
        else:
            mapping[int(c)] = Counter(clases).most_common(1)[0][0]
    return mapping


def _evaluar(y_true, y_pred):
    """Matriz de confusión y métricas básicas (sin sklearn)."""
    clases = CLASSES_ORDEN
    idx = {c: i for i, c in enumerate(clases)}
    cm = np.zeros((len(clases), len(clases)), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt not in idx or yp not in idx:
            continue
        cm[idx[yt], idx[yp]] += 1

    acc = (np.array(y_true) == np.array(y_pred)).mean()

    report = {}
    for c in clases:
        i = idx[c]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(cm[i, :].sum())
        report[c] = dict(precision=precision, recall=recall, f1=f1, support=support)

    return cm, report, acc


def entrenar_k4(csv_path: Path = CSV_PATH, models_dir: Path = MODELS_DIR, random_state: int = 42):
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de features en: {csv_path}")

    df = pd.read_csv(csv_path)
    if "clase_real" not in df.columns:
        raise ValueError("Tu CSV debe tener la columna 'clase_real'.")

    df = df[df["clase_real"] != "desconocida"].reset_index(drop=True)

    features = _seleccionar_features_existentes(df, K4_FEATURES_CANDIDATAS)
    print("Features usadas (k=4):", features)
    print("Total muestras:", len(df))

    X = df[features].values.astype(np.float64)
    y = df["clase_real"].values.astype(object)

    # 1) Escalado manual
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 2) KMeans manual k=4
    kmeans = KMeans(
        n_clusters=4,
        init="random",   # si tu implementación no lo soporta, cambiá a "random"
        n_init=20,
        max_iter=300,
        tol=1e-4,
        random_state=random_state,
    )
    kmeans.fit(Xs)
    labels = kmeans.labels_.astype(int)

    # 3) Mapping cluster -> clase por mayoría
    cluster_to_class = _mapear_clusters_a_clases(y, labels)

    # 4) Predicción en train (para evaluar)
    pred = [cluster_to_class[int(l)] for l in labels]

    cm, report, acc = _evaluar(y, pred)

    # 5) Guardar sin pisar nada
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, models_dir / "k4_scaler.joblib")
    joblib.dump(kmeans, models_dir / "k4_kmeans.joblib")

    info = {
        "features": features,
        "cluster_to_class": {str(int(k)): v for k, v in cluster_to_class.items()},
        "classes_orden": CLASSES_ORDEN,
        "csv_path": str(csv_path),
        "random_state": random_state,
        "nota": "Modelo KMeans k=4 manual (sin sklearn).",
    }
    with open(models_dir / "k4_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    pd.DataFrame(cm, index=CLASSES_ORDEN, columns=CLASSES_ORDEN).to_csv(models_dir / "confusion_matrix.csv")

    with open(models_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("Matriz de confusión (filas=real, columnas=pred):\n")
        f.write(str(cm) + "\n\n")
        f.write("Reporte:\n")
        for c in CLASSES_ORDEN:
            r = report[c]
            f.write(
                f"{c:10s}  precision={r['precision']:.2f}  recall={r['recall']:.2f}  "
                f"f1={r['f1']:.2f}  support={r['support']}\n"
            )
        f.write(f"\naccuracy={acc:.4f}\n")

    print("\n✔ Modelo guardado en:", models_dir)
    print("✔ accuracy train:", round(acc, 4))
    print("✔ cluster_to_class:", cluster_to_class)


if __name__ == "__main__":
    entrenar_k4()
