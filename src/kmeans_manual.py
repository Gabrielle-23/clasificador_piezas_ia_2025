# src/kmeans_manual.py
"""
Versión MANUAL (sin sklearn) del trainer jerárquico de 3 nodos KMeans.
Replica el comportamiento de src/kmeans.py, pero:
- Usa src.ml_manual (StandardScaler, KMeans)
- Guarda en data/models_manual/ (no pisa data/models/)
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

CSV_PATH = Path("data/features/features.csv")
MODELS_DIR = Path("data/models_manual")  # <-- carpeta separada

# ---------- FEATURES POR NODO ----------
NODE1_FEATURES = ["num_holes", "hole_area_ratio", "extent", "circularity"]
NODE2_FEATURES = ["hole_area_ratio", "num_vertices_rdp", "circularity", "hu2"]
NODE3_FEATURES = ["aspect_ratio", "extent", "circularity", "num_vertices_rdp", "hu2"]

AR_CLASSES = {"arandela", "tuerca"}
STICK_CLASSES = {"clavo", "tornillo"}


def _seleccionar_prototipos_centroide(X: np.ndarray, y: np.ndarray, class_labels, k: int = 1):
    """
    Para cada clase en class_labels:
    - calcula su centroide
    - devuelve índices (en X) de las k muestras más cercanas a ese centroide
    """
    indices = []
    for clase in class_labels:
        idxs = np.where(y == clase)[0]
        if len(idxs) == 0:
            raise ValueError(f"No hay muestras de clase '{clase}' para seleccionar prototipos.")

        Xc = X[idxs]
        centroide = Xc.mean(axis=0)
        dist = np.linalg.norm(Xc - centroide, axis=1)

        ordenados = idxs[np.argsort(dist)]
        indices.extend(ordenados[:k])
    return indices


def _fit_kmeans_2clusters(
    X_scaled: np.ndarray,
    y_labels: np.ndarray,
    class_labels,
    indices_protos,
    description: str,
):
    """
    Entrena KMeans(n_clusters=2) inicializando centros con prototipos (como en tu versión sklearn).
    Devuelve: (kmeans, mapping_cluster_a_clase, labels)
    """
    print(f"\n=== {description} ===")
    prot_centers = X_scaled[indices_protos]  # (2, D) si k=1 por clase

    kmeans = KMeans(
        n_clusters=2,
        init=prot_centers,
        n_init=1,
        max_iter=300,
        tol=1e-4,
        random_state=42,
    )
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # mapping cluster -> clase por mayoría
    mapping = {}
    for cl in [0, 1]:
        clases_en_cluster = y_labels[labels == cl]
        if len(clases_en_cluster) == 0:
            mapping[cl] = class_labels[cl]  # fallback
        else:
            mapping[cl] = Counter(clases_en_cluster).most_common(1)[0][0]

    print("Mapa cluster → etiqueta:")
    for c in mapping:
        print(f"  cluster {c} → {mapping[c]}")

    return kmeans, mapping, labels


# ---------- NODO 1: ARO vs PALO ----------
def entrenar_nodo1_aro_palo(df: pd.DataFrame, models_dir: Path):
    sub = df.copy()
    X = sub[NODE1_FEATURES].values
    y_real = sub["clase_real"].values

    # rama: aro si arandela/tuerca; palo si clavo/tornillo
    y_branch = np.array(["aro" if c in AR_CLASSES else "palo" for c in y_real], dtype=object)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_labels = ["aro", "palo"]
    indices_protos = _seleccionar_prototipos_centroide(X_scaled, y_branch, class_labels, k=1)

    kmeans, cluster_to_branch, _ = _fit_kmeans_2clusters(
        X_scaled,
        y_branch,
        class_labels=class_labels,
        indices_protos=indices_protos,
        description="Nodo 1 (ARO vs PALO)",
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "nodo1_aro_palo_scaler.joblib"
    modelo_path = models_dir / "nodo1_aro_palo_kmeans.joblib"
    info_path = models_dir / "nodo1_aro_palo_info.json"

    joblib.dump(scaler, scaler_path)
    joblib.dump(kmeans, modelo_path)

    info = {
        "features": NODE1_FEATURES,
        "cluster_to_branch": {int(k): v for k, v in cluster_to_branch.items()},
        "class_labels": class_labels,
        "prototype_selection": "closest_to_branch_centroid",
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n✔ Nodo 1 guardado en:\n  {scaler_path}\n  {modelo_path}\n  {info_path}")
    return scaler, kmeans, cluster_to_branch


# ---------- NODO 2: ARANDELA vs TUERCA ----------
def entrenar_nodo2_arandela_tuerca(df: pd.DataFrame, models_dir: Path):
    sub = df[df["clase_real"].isin(AR_CLASSES)].reset_index(drop=True)
    X = sub[NODE2_FEATURES].values
    y = sub["clase_real"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_labels = ["arandela", "tuerca"]
    indices_protos = _seleccionar_prototipos_centroide(X_scaled, y, class_labels, k=1)

    kmeans, cluster_to_class, _ = _fit_kmeans_2clusters(
        X_scaled,
        y,
        class_labels=class_labels,
        indices_protos=indices_protos,
        description="Nodo 2 (ARANDELA vs TUERCA)",
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "nodo2_arandela_tuerca_scaler.joblib"
    modelo_path = models_dir / "nodo2_arandela_tuerca_kmeans.joblib"
    info_path = models_dir / "nodo2_arandela_tuerca_info.json"

    joblib.dump(scaler, scaler_path)
    joblib.dump(kmeans, modelo_path)

    info = {
        "features": NODE2_FEATURES,
        "cluster_to_class": {int(k): v for k, v in cluster_to_class.items()},
        "class_labels": class_labels,
        "prototype_selection": "closest_to_class_centroid",
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n✔ Nodo 2 guardado en:\n  {scaler_path}\n  {modelo_path}\n  {info_path}")
    return scaler, kmeans, cluster_to_class


# ---------- NODO 3: CLAVO vs TORNILLO ----------
def entrenar_nodo3_clavo_tornillo(df: pd.DataFrame, models_dir: Path, k_protos_por_clase: int = 3):
    sub = df[df["clase_real"].isin(STICK_CLASSES)].reset_index(drop=True)

    def seleccionar_protos_para(clase: str):
        dfc = sub[sub["clase_real"] == clase].copy()
        Xc = dfc[NODE3_FEATURES].values
        centroide = Xc.mean(axis=0)
        dist = np.linalg.norm(Xc - centroide, axis=1)
        dfc["distancia"] = dist
        return dfc.sort_values("distancia").head(k_protos_por_clase)

    prot_clavos = seleccionar_protos_para("clavo")
    prot_tornillos = seleccionar_protos_para("tornillo")
    df_prot = pd.concat([prot_clavos, prot_tornillos], axis=0).reset_index(drop=True)

    print("\n=== Prototipos seleccionados para nodo 3 ===")
    print(df_prot[["filename", "clase_real"] + NODE3_FEATURES])

    X = df_prot[NODE3_FEATURES].values
    y = df_prot["clase_real"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Nodo 3 en tu versión usa init random (en sklearn n_init="auto") :contentReference[oaicite:4]{index=4}
    kmeans = KMeans(n_clusters=2, init="random", n_init=10, random_state=42)
    kmeans.fit(Xs)

    labels = kmeans.labels_
    mapping = {}
    for cluster in [0, 1]:
        clases_en_cluster = df_prot.iloc[labels == cluster]["clase_real"]
        clase_mayoritaria = clases_en_cluster.value_counts().idxmax()
        mapping[cluster] = clase_mayoritaria

    print("\n[Nodo 3] Mapa cluster → clase:")
    for c in mapping:
        print(f"  cluster {c} → {mapping[c]}")

    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "nodo3_clavo_tornillo_scaler.joblib"
    modelo_path = models_dir / "nodo3_clavo_tornillo_kmeans.joblib"
    info_path = models_dir / "nodo3_clavo_tornillo_info.json"

    joblib.dump(scaler, scaler_path)
    joblib.dump(kmeans, modelo_path)

    info = {
        "features": NODE3_FEATURES,
        "cluster_to_class": {int(k): v for k, v in mapping.items()},
        "class_labels": ["clavo", "tornillo"],
        "num_prototipos": len(df_prot),
        "prototype_selection": f"{k_protos_por_clase}_closest_to_class_centroid",
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n✔ Nodo 3 guardado en:\n  {scaler_path}\n  {modelo_path}\n  {info_path}")
    return scaler, kmeans, mapping


def evaluar_jerarquico_en_memoria(df: pd.DataFrame,
                                 scaler1, k1, map1,
                                 scaler2, k2, map2,
                                 scaler3, k3, map3):
    df_eval = df[df["clase_real"] != "desconocida"].reset_index(drop=True)

    preds = []
    for _, row in df_eval.iterrows():
        x1 = row[NODE1_FEATURES].values.astype(float).reshape(1, -1)
        x1s = scaler1.transform(x1)
        cl1 = int(k1.predict(x1s)[0])
        rama = map1[cl1]

        if rama == "aro":
            x2 = row[NODE2_FEATURES].values.astype(float).reshape(1, -1)
            x2s = scaler2.transform(x2)
            cl2 = int(k2.predict(x2s)[0])
            clase = map2[cl2]
        else:
            x3 = row[NODE3_FEATURES].values.astype(float).reshape(1, -1)
            x3s = scaler3.transform(x3)
            cl3 = int(k3.predict(x3s)[0])
            clase = map3[cl3]

        preds.append(clase)

    df_eval["pred_jerarquica"] = preds

    print("\n=== MATRIZ GLOBAL: clase_real vs pred_jerarquica ===")
    print(pd.crosstab(df_eval["clase_real"], df_eval["pred_jerarquica"]))

    acc = (df_eval["clase_real"] == df_eval["pred_jerarquica"]).mean()
    print(f"\nExactitud jerárquica global: {acc*100:.2f}%")

    return df_eval, acc


def entrenar_y_evaluar_jerarquico(csv_path: Path = CSV_PATH, models_dir: Path = MODELS_DIR):
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de features en {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["clase_real"] != "desconocida"].reset_index(drop=True)

    print(f"Total de piezas para entrenamiento: {len(df)}")

    scaler1, k1, map1 = entrenar_nodo1_aro_palo(df, models_dir)
    scaler2, k2, map2 = entrenar_nodo2_arandela_tuerca(df, models_dir)
    scaler3, k3, map3 = entrenar_nodo3_clavo_tornillo(df, models_dir, k_protos_por_clase=3)

    return evaluar_jerarquico_en_memoria(df, scaler1, k1, map1, scaler2, k2, map2, scaler3, k3, map3)


if __name__ == "__main__":
    entrenar_y_evaluar_jerarquico()
