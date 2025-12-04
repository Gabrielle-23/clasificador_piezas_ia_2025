# src/kmeans.py
"""
Script unificado para entrenar los 3 nodos KMeans (jerárquico)
y mostrar la matriz de confusión global.

- Lee los features desde: data/features/features.csv
- Entrena:
    Nodo 1: aro vs palo
    Nodo 2: arandela vs tuerca
    Nodo 3: clavo vs tornillo (con prototipos centrales, propuesta 1)
- Guarda los modelos en: data/models/
- Imprime la matriz clase_real vs pred_jerarquica y la exactitud global.

Se asume que el CSV tiene al menos estas columnas:
    filename, clase_real,
    num_holes, hole_area_ratio, extent, circularity,
    num_vertices_rdp, aspect_ratio, hu2
"""

from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


CSV_PATH = Path("data/features/features.csv")
MODELS_DIR = Path("data/models")

# ---------- FEATURES POR NODO ----------

# Nodo 1: ARO vs PALO (todas las piezas)
NODE1_FEATURES = [
    "num_holes",
    "hole_area_ratio",
    "extent",
    "circularity",
]

# Nodo 2: dentro de ARO (arandela vs tuerca)
NODE2_FEATURES = [
    "hole_area_ratio",
    "num_vertices_rdp",
    "circularity",
    "hu2",
]

# Nodo 3: dentro de PALO (clavo vs tornillo) - versión con prototipos centrales
NODE3_FEATURES = [
    "aspect_ratio",
    "extent",
    "circularity",
    "num_vertices_rdp",
    "hu2",
]

AR_CLASSES = {"arandela", "tuerca"}
STICK_CLASSES = {"clavo", "tornillo"}


# ---------- UTILIDADES GENERALES ----------

def _seleccionar_prototipos_centroide(X: np.ndarray, y: np.ndarray, class_labels, k: int = 1):
    """
    Para cada clase en class_labels:
    - calcula su centroide
    - devuelve los índices (en X) de las k muestras más cercanas a ese centroide
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
        # Nos quedamos con los primeros k (los más representativos)
        indices.extend(ordenados[:k])

    return indices


def _fit_kmeans_2clusters(
    X: np.ndarray,
    y: np.ndarray,
    class_labels,
    indices_protos,
    description: str,
):
    """
    Entrena un KMeans(k=2) con prototipos dados, devuelve (kmeans, cluster_to_class, labels).
    - X: matriz ya escalada
    - y: etiquetas reales
    - class_labels: nombres de las 2 clases, ej: ['arandela','tuerca']
    - indices_protos: índices en X de los prototipos en el mismo orden que class_labels
    """
    n_clusters = 2
    init_centers = X[indices_protos, :]

    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        random_state=0,
    )
    kmeans.fit(X)
    labels = kmeans.labels_

    # Mapeo cluster -> clase por mayoría
    cluster_to_class = {}
    for c in range(n_clusters):
        clases_en_cluster = y[labels == c]
        if len(clases_en_cluster) == 0:
            cluster_to_class[c] = None
            continue

        conteo = Counter(clases_en_cluster)
        clase_mas_comun, _ = conteo.most_common(1)[0]
        cluster_to_class[c] = clase_mas_comun

    print(f"\n[{description}] Matriz clase_real vs cluster:")
    crosstab = pd.crosstab(pd.Series(y, name="clase_real"), pd.Series(labels, name="cluster"))
    print(crosstab)

    print(f"\n[{description}] Mapeo cluster → clase:")
    for c in range(n_clusters):
        print(f"  cluster {c} → {cluster_to_class[c]}")

    return kmeans, cluster_to_class, labels


# ---------- NODO 1: ARO vs PALO ----------

def entrenar_nodo1_aro_palo(df: pd.DataFrame, models_dir: Path):
    """
    Nodo 1: usa TODAS las piezas, pero las agrupa en dos clases:
    - 'aro'  si clase_real es arandela o tuerca
    - 'palo' si clase_real es clavo o tornillo
    """
    sub = df.copy().reset_index(drop=True)

    # Etiquetas de rama: aro vs palo
    y_branch = np.where(sub["clase_real"].isin(AR_CLASSES), "aro", "palo")

    X = sub[NODE1_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_labels = ["aro", "palo"]
    # Seleccionamos un prototipo por rama (más cercano al centroide)
    indices_protos = _seleccionar_prototipos_centroide(X_scaled, y_branch, class_labels, k=1)

    kmeans, cluster_to_branch, labels = _fit_kmeans_2clusters(
        X_scaled,
        y_branch,
        class_labels=class_labels,
        indices_protos=indices_protos,
        description="Nodo 1 (ARO vs PALO)",
    )

    # Guardar scaler y modelo
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

    # Para cada clase, buscamos las muestras más cercanas al centroide
    indices_protos = _seleccionar_prototipos_centroide(X_scaled, y, class_labels, k=1)

    kmeans, cluster_to_class, labels = _fit_kmeans_2clusters(
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


# ---------- NODO 3: CLAVO vs TORNILLO (prototipos centrales, propuesta 1) ----------

def entrenar_nodo3_clavo_tornillo(df: pd.DataFrame, models_dir: Path, k_protos_por_clase: int = 3):
    sub = df[df["clase_real"].isin(STICK_CLASSES)].reset_index(drop=True)

    # Seleccionamos k_protos_por_clase prototipos centrales usando las NODE3_FEATURES
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
    y = df_prot["clase_real"].map({"clavo": 0, "tornillo": 1}).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
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

    # Guardar
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


# ---------- EVALUACIÓN JERÁRQUICA CON LOS MODELOS ENTRENADOS EN MEMORIA ----------

def evaluar_jerarquico_en_memoria(df: pd.DataFrame,
                                  scaler1, k1, map1,
                                  scaler2, k2, map2,
                                  scaler3, k3, map3):
    """
    Recibe el DataFrame de features y los 3 nodos ya entrenados,
    devuelve la matriz de confusión y la exactitud global.
    """

    df_eval = df[df["clase_real"] != "desconocida"].reset_index(drop=True)

    preds = []

    for _, row in df_eval.iterrows():
        # Nodo 1: aro vs palo
        x1 = row[NODE1_FEATURES].values.astype(float).reshape(1, -1)
        x1s = scaler1.transform(x1)
        cl1 = int(k1.predict(x1s)[0])
        rama = map1[cl1]   # 'aro' o 'palo'

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


# ---------- FUNCIÓN PRINCIPAL ----------

def entrenar_y_evaluar_jerarquico(
    csv_path: Path = CSV_PATH,
    models_dir: Path = MODELS_DIR,
):
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de features en {csv_path}")

    df = pd.read_csv(csv_path)

    # Filtramos cualquier fila sin clase conocida
    df = df[df["clase_real"] != "desconocida"].reset_index(drop=True)

    print(f"Total de piezas para entrenamiento: {len(df)}")

    # Entrenamos los 3 nodos
    scaler1, k1, map1 = entrenar_nodo1_aro_palo(df, models_dir)
    scaler2, k2, map2 = entrenar_nodo2_arandela_tuerca(df, models_dir)
    scaler3, k3, map3 = entrenar_nodo3_clavo_tornillo(df, models_dir, k_protos_por_clase=3)

    # Evaluamos con los modelos recién entrenados
    df_eval, acc = evaluar_jerarquico_en_memoria(
        df,
        scaler1, k1, map1,
        scaler2, k2, map2,
        scaler3, k3, map3,
    )

    return df_eval, acc


if __name__ == "__main__":
    entrenar_y_evaluar_jerarquico()
