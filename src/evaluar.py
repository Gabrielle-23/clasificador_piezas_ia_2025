# src/evaluar_jerarquico_modelos.py

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib


CSV_PATH = Path("data/features/features.csv")
MODELS_DIR = Path("data/models")


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_nodo1(models_dir: Path):
    scaler = joblib.load(models_dir / "nodo1_aro_palo_scaler.joblib")
    kmeans = joblib.load(models_dir / "nodo1_aro_palo_kmeans.joblib")
    info = _load_json(models_dir / "nodo1_aro_palo_info.json")

    feats = info["features"]
    # En el JSON las claves quedan como strings
    cluster_to_branch = {int(k): v for k, v in info["cluster_to_branch"].items()}

    return feats, scaler, kmeans, cluster_to_branch


def cargar_nodo2(models_dir: Path):
    scaler = joblib.load(models_dir / "nodo2_arandela_tuerca_scaler.joblib")
    kmeans = joblib.load(models_dir / "nodo2_arandela_tuerca_kmeans.joblib")
    info = _load_json(models_dir / "nodo2_arandela_tuerca_info.json")

    feats = info["features"]
    cluster_to_class = {int(k): v for k, v in info["cluster_to_class"].items()}

    return feats, scaler, kmeans, cluster_to_class


def cargar_nodo3(models_dir: Path):
    """
    Carga el nodo 3 actual (da igual si viene del
    train_kmeans_hierarchical o del script de prototipos).
    """
    scaler = joblib.load(models_dir / "nodo3_clavo_tornillo_scaler.joblib")
    kmeans = joblib.load(models_dir / "nodo3_clavo_tornillo_kmeans.joblib")
    info = _load_json(models_dir / "nodo3_clavo_tornillo_info.json")

    feats = info["features"]
    cluster_to_class = {int(k): v for k, v in info["cluster_to_class"].items()}

    return feats, scaler, kmeans, cluster_to_class


def evaluar_jerarquico(csv_path: Path = CSV_PATH, models_dir: Path = MODELS_DIR):
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de features: {csv_path}")

    df = pd.read_csv(csv_path)
    if "clase_real" not in df.columns:
        raise ValueError("El CSV no tiene la columna 'clase_real'.")

    # Sacamos cualquier fila "desconocida"
    df = df[df["clase_real"] != "desconocida"].reset_index(drop=True)

    # Cargar modelos ya entrenados desde disco
    f1, scaler1, km1, cluster_to_branch = cargar_nodo1(models_dir)
    f2, scaler2, km2, cluster_to_at = cargar_nodo2(models_dir)
    f3, scaler3, km3, cluster_to_ct = cargar_nodo3(models_dir)

    preds = []

    for _, row in df.iterrows():
        # ----- Nodo 1: aro vs palo -----
        x1 = row[f1].values.astype(float).reshape(1, -1)
        x1s = scaler1.transform(x1)
        cl1 = int(km1.predict(x1s)[0])
        branch = cluster_to_branch[cl1]   # 'aro' o 'palo'

        # ----- Nodo 2 o Nodo 3 según rama -----
        if branch == "aro":
            x2 = row[f2].values.astype(float).reshape(1, -1)
            x2s = scaler2.transform(x2)
            cl2 = int(km2.predict(x2s)[0])
            clase = cluster_to_at[cl2]
        else:  # 'palo'
            x3 = row[f3].values.astype(float).reshape(1, -1)
            x3s = scaler3.transform(x3)
            cl3 = int(km3.predict(x3s)[0])
            clase = cluster_to_ct[cl3]

        preds.append(clase)

    df["pred_jerarquica"] = preds

    print("\n=== MATRIZ GLOBAL: clase_real vs pred_jerarquica (modelos actuales) ===")
    print(pd.crosstab(df["clase_real"], df["pred_jerarquica"]))

    acc = (df["clase_real"] == df["pred_jerarquica"]).mean()
    print(f"\nExactitud jerárquica global (modelos actuales): {acc*100:.2f}%")

    return df


if __name__ == "__main__":
    evaluar_jerarquico()
