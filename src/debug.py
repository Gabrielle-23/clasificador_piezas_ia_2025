# src/debug_pred_vs_csv.py
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import joblib

from src.binario import binarizar
from src.features import extraer_features_desde_mask

CSV_PATH = Path("data/features/features.csv")   # <-- NUEVO
COLOR_DIR = Path("data/raso")                  # <-- donde están las fotos .jpg originales
MODELS_DIR = Path("data/models")


def _cargar_info_json(path_json: Path):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _lookup(mapping: dict, key_int: int):
    key_str = str(key_int)
    if key_str in mapping:
        return mapping[key_str]
    if key_int in mapping:
        return mapping[key_int]
    raise KeyError(f"Clave {key_int} no encontrada en mapping {mapping}.")


def cargar_modelos():
    scaler1 = joblib.load(MODELS_DIR / "nodo1_aro_palo_scaler.joblib")
    kmeans1 = joblib.load(MODELS_DIR / "nodo1_aro_palo_kmeans.joblib")
    info1 = _cargar_info_json(MODELS_DIR / "nodo1_aro_palo_info.json")

    scaler2 = joblib.load(MODELS_DIR / "nodo2_arandela_tuerca_scaler.joblib")
    kmeans2 = joblib.load(MODELS_DIR / "nodo2_arandela_tuerca_kmeans.joblib")
    info2 = _cargar_info_json(MODELS_DIR / "nodo2_arandela_tuerca_info.json")

    scaler3 = joblib.load(MODELS_DIR / "nodo3_clavo_tornillo_scaler.joblib")
    kmeans3 = joblib.load(MODELS_DIR / "nodo3_clavo_tornillo_kmeans.joblib")
    info3 = _cargar_info_json(MODELS_DIR / "nodo3_clavo_tornillo_info.json")

    return (scaler1, kmeans1, info1,
            scaler2, kmeans2, info2,
            scaler3, kmeans3, info3)


def clasificar_desde_dict(feats: dict, modelos):
    (scaler1, k1, info1,
     scaler2, k2, info2,
     scaler3, k3, info3) = modelos

    # Nodo 1
    f1 = info1["features"]
    x1 = np.array([[feats[f] for f in f1]], dtype=np.float64)
    x1s = scaler1.transform(x1)
    c1 = int(k1.predict(x1s)[0])
    rama = _lookup(info1["cluster_to_branch"], c1)

    if rama == "aro":
        f2 = info2["features"]
        x2 = np.array([[feats[f] for f in f2]], dtype=np.float64)
        x2s = scaler2.transform(x2)
        c2 = int(k2.predict(x2s)[0])
        clase = _lookup(info2["cluster_to_class"], c2)
        nodo_final = 2
        cluster_final = c2
    else:
        f3 = info3["features"]
        x3 = np.array([[feats[f] for f in f3]], dtype=np.float64)
        x3s = scaler3.transform(x3)
        c3 = int(k3.predict(x3s)[0])
        clase = _lookup(info3["cluster_to_class"], c3)
        nodo_final = 3
        cluster_final = c3

    return {
        "rama": rama,
        "nodo_final": nodo_final,
        "cluster_final": cluster_final,
        "clase_predicha": clase,
    }


def debug_pieza(nombre_base: str):
    """
    nombre_base: sin extensión, ej. 'tuerca20', 'clavo20'
    Usa:
      - 'nombre_base.png' para buscar en el CSV
      - 'nombre_base.jpg' para leer imagen color en data/raso
    """
    print("=" * 60)
    print("DEBUG PARA:", nombre_base)

    df = pd.read_csv(CSV_PATH)

    filename_bin = nombre_base + ".png"
    fila = df[df["filename"] == filename_bin]

    if fila.empty:
        print("⚠ No encontré", filename_bin, "en el CSV.")
        return

    fila = fila.iloc[0]
    clase_real = fila["clase_real"]
    feats_csv = fila.to_dict()

    print("\n▶ Fila CSV:", filename_bin)
    print("  clase_real:", clase_real)
    for k in ["num_holes", "hole_area_ratio", "aspect_ratio",
              "extent", "circularity", "hu2"]:
        if k in feats_csv:
            print(f"  {k}: {feats_csv[k]}")

    modelos = cargar_modelos()
    res_csv = clasificar_desde_dict(feats_csv, modelos)
    print("\n▶ Clasificación usando SOLO features del CSV:")
    print("  rama:", res_csv["rama"])
    print("  nodo_final:", res_csv["nodo_final"])
    print("  clase_predicha:", res_csv["clase_predicha"])

    # --- Recalcular desde la imagen color ---
    ruta_color = COLOR_DIR / (nombre_base + ".jpg")
    img = cv2.imread(str(ruta_color), cv2.IMREAD_COLOR)
    if img is None:
        print("\n⚠ No pude leer la imagen color:", ruta_color)
        return

    mask = binarizar(img)

    lista_feats = extraer_features_desde_mask(
        mask,
        min_area_rel=0.001,
        min_hole_area_rel=0.0005,
        epsilon_factor=0.01,
    )
    if not lista_feats:
        print("\n⚠ No se encontraron objetos válidos en la máscara nueva.")
        return

    feats_new = lista_feats[0]

    print("\n▶ Features recalculados desde imagen color:")
    for k in ["num_holes", "hole_area_ratio", "aspect_ratio",
              "extent", "circularity", "hu2"]:
        if k in feats_new:
            print(f"  {k}: {feats_new[k]}")

    res_new = clasificar_desde_dict(feats_new, modelos)
    print("\n▶ Clasificación usando features recalculados:")
    print("  rama:", res_new["rama"])
    print("  nodo_final:", res_new["nodo_final"])
    print("  clase_predicha:", res_new["clase_predicha"])


if __name__ == "__main__":
    # probá con alguna pieza del dataset, por ejemplo:
    debug_pieza("tuerca20")
    debug_pieza("clavo20")
